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
  * PE column) in a single cycle: into the result
  * memory, or into the input memory if toInput
  * (then only the low inWidth bits of each value
  * are kept, one value per bank).
  */
class BufResultWrite(c: SAConfig) extends Bundle {
  val rowAddr = UInt(c.resRowAddrW.W)
  val data = Vec(c.cols, c.accT)
  val toInput = Bool()
}