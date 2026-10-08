package tpu.memory

import chisel3._
import chisel3.util._

import ifaces._

/**
  * Banked on-chip memory holding the matrices A, B and the results C.
  */
class UnifiedBuffer(c: SAConfig) extends Module {
    val io = IO(new Bundle {
        val hostWr = Flipped(Decoupled(new BufWritePort(c)))
        val hostRdReq = Flipped(Decoupled(new BufReadReq(c)))
        val hostRdResp = Decoupled(c.accT)

        val mvRdReq = Flipped(Decoupled(new BufRowReq(c)))
        val mvRdResp = Decoupled(Vec(c.banks, c.inT))

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
      * Write logic for the input memory. The host can always write to it, 
      * and the data mover can always read from it.
      */
    io.hostWr.ready := true.B // The host can always write to the input memory

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
    io.mvRdResp.valid := RegNext(io.mvRdReq.valid, false.B)



    /**
      * Result memory for C. Written one whole row per cycle by the 
      * accumulators (resWr), read element by element by the host 
      * (hostRdReq/hostRdResp).
      */
    val resultMem = Seq.fill(c.cols) {
        SyncReadMem(c.outDepth / c.cols, c.accT)
    }

    io.resWr.ready := true.B

    when (io.resWr.fire) {
        for (col <- 0 until c.cols) {
            resultMem(col).write(
                io.resWr.bits.rowAddr, 
                io.resWr.bits.data(col))
        }
    }
}

