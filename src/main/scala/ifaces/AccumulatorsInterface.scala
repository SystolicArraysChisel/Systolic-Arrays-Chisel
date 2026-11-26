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
  val leakShift = UInt(3.W)

  val requant = Bool()
  val shift = UInt(c.shiftW.W)

  val toInput = Bool() // write the finished rows into the input memory

  val bias = Bool() // the first K-tile starts from the bias memory
  val biasAddr = UInt(c.biasAddrW.W) // bias entry of this tile's first column
}
