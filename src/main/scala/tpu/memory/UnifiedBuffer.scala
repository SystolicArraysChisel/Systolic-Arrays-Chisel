package tpu.memory

import chisel3._
import chisel3.util._

import ifaces._

/**
  * Banked on-chip memory holding the matrices A, B and the results C.
  */
class UnifiedBuffer(c: SAConfig) extends Module {
    val io = IO(new Bundle {
        // CommandDecoder <-> UnifiedBuffer interface
        val hostWr = Flipped(Decoupled(new BufWritePort(c)))
        val hostRdReq = Flipped(Decoupled(new BufReadReq(c)))
        val hostRdResp = Decoupled(c.accT)

        // DataMover <-> UnifiedBuffer interface
        val mvRdReq = Flipped(Decoupled(new BufRowReq(c)))
        val mvRdResp = Decoupled(Vec(c.banks, c.inT))

        // Accumulator <-> UnifiedBuffer interface
        val resWr = Flipped(Decoupled(new BufResultWrite(c)))
    })



    /**
      * Split an address into a bank/column selection and 
      * a row address inside that bank/column.
      */
    def splitAddr(addr: UInt, bits: Int): (UInt, UInt) = {
        if (bits > 0) {
            (addr(bits - 1, 0), addr(addr.getWidth - 1, bits))
        } else {
            (0.U, addr)
        }
    }



    /**
      * Input memory for A and B. Written element by element by 
      * the host (hostWr, always ready), read one whole row per 
      * cycle by the data mover.
      */
    val inputMem = Seq.fill(c.banks) {
        SyncReadMem(c.inDepth / c.banks, c.inT)
    }

    /**
      * Write logic for the input memory.
      * The host can always write to the input memory.
      */
    io.hostWr.ready := true.B

    val (hostWrBank, hostWrRow) = splitAddr(io.hostWr.bits.addr, c.bankBits)

    when (io.hostWr.fire) {
        for (b <- 0 until c.banks) {
            when (hostWrBank === b.U) {
                inputMem(b).write(
                    hostWrRow, 
                    io.hostWr.bits.data)
            }
        }
    }

    /**
      * Read logic for the input memory. 
      * The data mover can always read from it.
      */
    io.mvRdReq.ready := true.B

    val mvRdRespVec = Wire(Vec(c.banks, c.inT))
    for (b <- 0 until c.banks) {
        mvRdRespVec(b) := inputMem(b).read(
            io.mvRdReq.bits.rowAddr, 
            io.mvRdReq.valid)
    }
    io.mvRdResp.bits := mvRdRespVec
    // The valid signal is delayed by one cycle to account for the read latency of SyncReadMem
    io.mvRdResp.valid := RegNext(io.mvRdReq.valid, false.B)

    assert(!io.mvRdResp.valid || io.mvRdResp.ready, 
        "UnifiedBuffer: mvRdResp not accepted, DataMover must always be ready")



    /**
      * Result memory for C. Written one whole row per cycle by the 
      * accumulators (resWr), read element by element by the host 
      * (hostRdReq/hostRdResp).
      */
    val resultMem = Seq.fill(c.cols) {
        SyncReadMem(c.outDepth / c.cols, c.accT)
    }

    /**
      * Write logic for the result memory. 
      */
    io.resWr.ready := true.B

    when (io.resWr.fire) {
        for (col <- 0 until c.cols) {
            resultMem(col).write(
                io.resWr.bits.rowAddr, 
                io.resWr.bits.data(col))
        }
    }

    /**
      * Read logic for the result memory. 
      */
    val (hostRdCol, hostRdRow) = splitAddr(io.hostRdReq.bits.addr, c.colBits)

    val rdFireD = RegNext(io.hostRdReq.fire, false.B)
    val rdColD = RegEnable(hostRdCol, io.hostRdReq.fire)
    val respValid = RegInit(false.B)
    val respData = Reg(c.accT)

    val hostRdRespVec = Wire(Vec(c.cols, c.accT))

    for (col <- 0 until c.cols) {
        hostRdRespVec(col) := resultMem(col).read(
            hostRdRow, 
            io.hostRdReq.fire && (hostRdCol === col.U))
    }

    val rdRaw = if (c.cols == 1) {
        hostRdRespVec(0)
    } else {
        hostRdRespVec(rdColD)
    }

    when (rdFireD) {
        respData := rdRaw
        respValid := true.B
    }
    when (io.hostRdResp.fire) {
        respValid := false.B
    }

    io.hostRdReq.ready := !rdFireD && !respValid
    io.hostRdResp.bits := respData
    io.hostRdResp.valid := respValid
}
