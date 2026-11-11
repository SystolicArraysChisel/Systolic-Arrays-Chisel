package ifaces

import chisel3._
import chisel3.util._

class AccCtrl(c: SAConfig) extends Bundle {
  val first = Bool()
  val last = Bool()

  val cAddr = UInt(c.resAddrW.W)
  val ldC = UInt(c.dimW.W)
  val m = UInt(c.dimW.W)

  val relu = Bool()

  val requant = Bool()
  val shift = UInt(c.shiftW.W)

  val toInput = Bool() // write the finished rows into the input memory
}
