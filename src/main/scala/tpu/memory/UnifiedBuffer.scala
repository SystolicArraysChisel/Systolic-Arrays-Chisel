package tpu.memory

import chisel3._
import chisel3.util._

import ifaces._

/**
  * Banked on-chip memory holding the matrices A, B and the results C
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

    io.hostWr.ready := true.B // The host can always write to the input memory



    /**
      * Input memory for A and B. Written element by element by 
      * the host (hostWr, always ready), read one whole row per 
      * cycle by the data mover
      */
    val inputMem = Seq.fill(c.banks) {
        SyncReadMem(c.inDepth / c.banks, c.inT)
    }

    val hostWrBank = Wire(UInt(c.bankBits.W))
    val hostWrRow = Wire(UInt((c.addrW - c.bankBits).W))
    val mvRdBank = Wire(UInt(c.bankBits.W))
    val mvRdRow = Wire(UInt((c.addrW - c.bankBits).W))

    if (c.bankBits > 0) {
        val hostWrBank = io.hostWr.bits.addr(c.bankBits - 1, 0)
        val hostWrRow = io.hostWr.bits.addr(c.addrW - 1, c.bankBits)

        val mvRdBank = io.mvRdReq.bits.rowAddr(c.bankBits - 1, 0)
        val mvRdRow = io.mvRdReq.bits.rowAddr(c.addrW - 1, c.bankBits)
    } else {
        val hostWrBank = 0.U
        val hostWrRow = io.hostWr.bits.addr(c.addrW - 1, 0)

        val mvRdBank = 0.U
        val mvRdRow = io.mvRdReq.bits.rowAddr(c.addrW - 1, 0)
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
      * Result memory for C. Written one whole row per cycle by the 
      * accumulators (resWr), read element by element by the host 
      * (hostRdReq/hostRdResp).
      */
    val resultMem = Seq.fill(c.cols) {
        SyncReadMem(c.outDepth / c.cols, c.accT)
    }
}

