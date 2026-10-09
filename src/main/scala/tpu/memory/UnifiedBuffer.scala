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

    val hostWrBank = Wire(UInt(c.bankBits.W))
    val hostWrRow = Wire(UInt((c.addrW - c.bankBits).W))

    if (c.bankBits > 0) {
        hostWrBank := io.hostWr.bits.addr(c.bankBits - 1, 0)
        hostWrRow := io.hostWr.bits.addr(c.addrW - 1, c.bankBits)
    } else {
        hostWrBank := 0.U
        hostWrRow := io.hostWr.bits.addr(c.addrW - 1, 0)
    }

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
      * The host can always read from the result memory.
      */
    io.hostRdReq.ready := true.B
    
    val hostRdBank = Wire(UInt(c.bankBits.W))
    val hostRdRow = Wire(UInt((c.outAddrW - c.bankBits).W))

    if (c.bankBits > 0) {
        hostRdBank := io.hostRdReq.bits.addr(c.bankBits - 1, 0)
        hostRdRow := io.hostRdReq.bits.addr(c.outAddrW - 1, c.bankBits)
    } else {
        hostRdBank := 0.U
        hostRdRow := io.hostRdReq.bits.addr(c.outAddrW - 1, 0)
    }

    val hostRdRespVec = Wire(Vec(c.cols, c.accT))

    for (col <- 0 until c.cols) {
        hostRdRespVec(col) := resultMem(col).read(
            hostRdRow, 
            io.hostRdReq.valid && (hostRdBank === col.U))
    }

    // The valid signal is delayed by one cycle to account for the read latency of SyncReadMem
    io.hostRdResp.valid := RegNext(io.hostRdReq.valid, false.B)
    // The bank selection is also delayed by one cycle to match the valid signal
    val delayedBank = RegEnable(hostRdBank, io.hostRdReq.valid)
    io.hostRdResp.bits := hostRdRespVec(delayedBank)
}
