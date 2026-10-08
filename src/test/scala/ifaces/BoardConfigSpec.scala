package ifaces

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class BoardConfigTester extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "BoardConfig"

  it should "match the Nexys4 DDR by default" in {
    val b = BoardConfig()
    assert(b.cyclesPerBit == 868)
    assert(b.byteCycles == 8680)
    assert(b.baudError < 0.001)
    assert(b.timeoutW == 21)
  }
}
