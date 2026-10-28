package tpu.compute.array

import chisel3._
import chiseltest._
import ifaces._
import org.scalatest.flatspec.AnyFlatSpec

class ProcessingElementSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "ProcessingElement"

  private val config = SAConfig(
    rows = 1,
    cols = 1,
    inDepth = 64,
    outDepth = 64,
    maxM = 4
  )

  private def loadWeight(dut: ProcessingElement, w: Int): Unit = {
    dut.io.wIn.poke(w.S)
    dut.io.wEn.poke(true.B)
    dut.clock.step()
    dut.io.wEn.poke(false.B)
  }

  it should "multiply-accumulate with negative values" in {
    test(new ProcessingElement(config)) { dut =>
      loadWeight(dut, -3)
      dut.io.aIn.poke(5.S)
      dut.io.psumIn.poke(-7.S)
      dut.clock.step()
      dut.io.psumOut.expect(-22.S)
      dut.io.aOut.expect(5.S)
    }
  }

  it should "pass the weight down and hold it while wEn is low" in {
    test(new ProcessingElement(config)) { dut =>
      loadWeight(dut, 4)
      dut.io.wOut.expect(4.S)

      // wEn stays low, so this value must be ignored.
      dut.io.wIn.poke(9.S)
      dut.clock.step(3)
      dut.io.wOut.expect(4.S)

      dut.io.aIn.poke(2.S)
      dut.io.psumIn.poke(0.S)
      dut.clock.step()
      dut.io.psumOut.expect(8.S)
    }
  }

  it should "wrap around on accumulator overflow" in {
    test(new ProcessingElement(config)) { dut =>
      loadWeight(dut, 1)
      dut.io.aIn.poke(1.S)
      dut.io.psumIn.poke(Int.MaxValue.S)
      dut.clock.step()
      dut.io.psumOut.expect(Int.MinValue.S)
    }
  }
}
