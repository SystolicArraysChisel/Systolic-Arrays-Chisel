package ifaces

import chisel3._
import chisel3.util._

/**
  * Asks the input memory for one full row
  * (one value per bank) in a single cycle.
  */
class BufRowReq(c: SAConfig) extends Bundle {
  val rowAddr = UInt(c.rowAddrW.W)
}

/**
  * Writes one finished row of C (one value per
  * PE column) into the result memory in a
  * single cycle.
  */
class BufResultWrite(c: SAConfig) extends Bundle {
  val rowAddr = UInt((c.outAddrW - log2Ceil(c.cols)).W)
  val data = Vec(c.cols, c.accT)
}