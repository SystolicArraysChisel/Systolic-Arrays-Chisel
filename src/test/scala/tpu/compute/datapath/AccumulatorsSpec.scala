package tpu.compute.datapath

import chisel3._
import chiseltest._
import ifaces._
import org.scalatest.flatspec.AnyFlatSpec

class AccumulatorsSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "Accumulators"

  private val config = SAConfig(
    rows = 2,
    cols = 2,
    inDepth = 64,
    outDepth = 64,
    maxM = 4
  )

  private def initialize(dut: Accumulators): Unit = {
    dut.io.in.valid.poke(false.B)
    dut.io.ctrl.valid.poke(false.B)
    dut.io.resWr.ready.poke(true.B)

    // Reset the DUT.
    dut.reset.poke(true.B)
    dut.clock.step()
    dut.reset.poke(false.B)
  }

  private def sendControl(
      dut: Accumulators,

      first: Boolean,
      last: Boolean,
      cAddr: Int,
      ldC: Int,
      m: Int,
      requant: Boolean = false,
      shift: Int = 0,
      relu: Boolean = false
  ): Unit = {
    dut.io.ctrl.bits.first.poke(first.B)
    dut.io.ctrl.bits.last.poke(last.B)
    dut.io.ctrl.bits.cAddr.poke(cAddr.U)
    dut.io.ctrl.bits.ldC.poke(ldC.U)
    dut.io.ctrl.bits.m.poke(m.U)
    dut.io.ctrl.bits.requant.poke(requant.B)
    dut.io.ctrl.bits.shift.poke(shift.U)
    dut.io.ctrl.bits.relu.poke(relu.B)

    dut.io.ctrl.valid.poke(true.B)
    dut.io.ctrl.ready.expect(true.B)

    // The control is accepted in one cycle, so we 
    // can deassert valid in the next cycle.
    dut.clock.step()
    dut.io.ctrl.valid.poke(false.B)
  }

  private def sendRows(
      dut: Accumulators,
      rows: Seq[Seq[Int]],
      last: Boolean
  ): Unit = {
    rows.zipWithIndex.foreach { case (row, rowIndex) =>
      row.zipWithIndex.foreach { case (value, col) =>
        dut.io.in.bits(col).poke(value.S)
      }
      dut.io.in.valid.poke(true.B)

      // done.valid comes with the last row, unless the tile is the last
      // one: then it comes with the last write.
      dut.io.done.valid.expect((!last && rowIndex == rows.size - 1).B)

      dut.clock.step()
      dut.io.in.valid.poke(false.B)
    }
  }

  private def readWrites(
      dut: Accumulators,
      expected: Seq[(Int, Seq[Int])]
  ): Unit = {
    expected.foreach { case (rowAddr, values) =>
      dut.io.resWr.valid.expect(true.B)
      dut.io.done.valid.expect(((rowAddr, values) == expected.last).B)

      dut.io.resWr.bits.rowAddr.expect(rowAddr.U)
      values.zipWithIndex.foreach { case (value, col) =>
        dut.io.resWr.bits.data(col).expect(value.S)
      }
      dut.clock.step()
    }
    // After all writes, resWr.valid should be deasserted.
    dut.io.resWr.valid.expect(false.B)
  }

  it should "write a single first and last tile" in {
    test(new Accumulators(config)) { dut =>
      initialize(dut)
      
      // Row addresses are (cAddr / cols) = (4 / 2) = 2 
      // and (cAddr + ldC) / cols = (4 + 4) / 2 = 4.
      sendControl(
        dut, first = true, last = true, 
        cAddr = 4, ldC = 4, m = 2)
      sendRows(dut, Seq(Seq(1, -2), Seq(3, 4)), last = true)
      readWrites(dut, Seq(2 -> Seq(1, -2), 4 -> Seq(3, 4)))
    }
  }

  it should "accumulate two tiles before writing the result" in {
    test(new Accumulators(config)) { dut =>
      initialize(dut)

      // Row addresses are (cAddr / cols) = (0 / 2) = 0
      // and (cAddr + ldC) / cols = (0 + 4) / 2 = 2.
      sendControl(
        dut, first = true, last = false, 
        cAddr = 0, ldC = 4, m = 2)
      sendRows(dut, Seq(Seq(1, 2), Seq(3, 4)), last = false)

      sendControl(
        dut, first = false, last = true, 
        cAddr = 0, ldC = 4, m = 2)
      sendRows(dut, Seq(Seq(10, 20), Seq(-3, 6)), last = true)

      // The expected result is the sum of the two tiles.
      readWrites(dut, Seq(0 -> Seq(11, 22), 2 -> Seq(0, 10)))
    }
  }

  it should "saturate requantized results to the int8 symmetric range" in {
    test(new Accumulators(config)) { dut =>
      initialize(dut)

      // Row addresses are (cAddr / cols) = (0 / 2) = 0
      // and (cAddr + ldC) / cols = (0 + 2) / 2 = 1. 
      // The tile is requantized with a shift of 1.
      sendControl(
        dut, first = true, last = true,
        cAddr = 0, ldC = 2, m = 1,
        requant = true, shift = 1
      )
      sendRows(dut, Seq(Seq(1000, -1000)), last = true)

      // The expected result is the requantized values,
      // which are saturated to the int8 symmetric range.
      readWrites(dut, Seq(0 -> Seq(127, -127)))
    }
  }
}

/**
  * TODO: More tests.
  */