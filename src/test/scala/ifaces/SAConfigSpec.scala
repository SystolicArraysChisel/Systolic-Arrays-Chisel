package ifaces

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SAConfigTester extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "SAConfig"

  it should "give the MVP configuration by default" in {
    val c = SAConfig()
    assert(c.rows == 4 && c.cols == 4)
    assert(c.banks == 4 && c.bankBits == 2)
    assert(c.addrW == 12 && c.rowAddrW == 10)
    assert(c.outAddrW == 12 && c.shiftW == 5)
    assert(c.dimW == 16)
    assert(c.arrayLatency == 7)
    assert(c.dspCount == 16)
  }
}
