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

    val inputMem = Seq.fill(c.banks) {
        SyncReadMem(c.inDepth / c.banks, c.inT)
    }

    val resultMem = Seq.fill(c.cols) {
        SyncReadMem(c.outDepth / c.cols, c.accT)
    }
}

