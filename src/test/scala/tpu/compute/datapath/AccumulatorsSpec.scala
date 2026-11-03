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
    while (!dut.io.resWr.valid.peekBoolean()) {
      dut.clock.step()
    }
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

  it should "apply ReLU to negative values without requantization" in {
    test(new Accumulators(config)) { dut =>
      initialize(dut)

      sendControl(
        dut, first = true, last = true,
        cAddr = 0, ldC = 2, m = 2,
        relu = true, requant = false
      )
      sendRows(dut, Seq(Seq(15, -30), Seq(-10, 45)), last = true)
      readWrites(dut, Seq(0 -> Seq(15, 0), 1 -> Seq(0, 45)))
    }
  }

  it should "apply ReLU with requantization and scaling" in {
    test(new Accumulators(config)) { dut =>
      initialize(dut)

      // shift = 2: rounding is +2 (1 << 1)
      // 40: (40 + 2) >> 2 = 10 -> ReLU(10) = 10
      // -40: (-40 + 2) >> 2 = -10 -> ReLU(-10) = 0
      // -1: (-1 + 2) >> 2 = 0 -> ReLU(0) = 0
      // 100: (100 + 2) >> 2 = 25 -> ReLU(25) = 25
      sendControl(
        dut, first = true, last = true,
        cAddr = 0, ldC = 2, m = 2,
        requant = true, shift = 2, relu = true
      )
      sendRows(dut, Seq(Seq(40, -40), Seq(-1, 100)), last = true)
      readWrites(dut, Seq(0 -> Seq(10, 0), 1 -> Seq(0, 25)))
    }
  }

  it should "round ties toward positive infinity during requantization" in {
    test(new Accumulators(config)) { dut =>
      initialize(dut)

      // shift = 1: rounding is +1 (1 << 0)
      // 1: (1 + 1) >> 1 = 1 (tie: 0.5 rounds up to 1)
      // 2: (2 + 1) >> 1 = 1 (1.0 is 1)
      // -1: (-1 + 1) >> 1 = 0 (tie: -0.5 rounds up to 0)
      // -2: (-2 + 1) >> 1 = -1 (-1.0 is -1)
      // -3: (-3 + 1) >> 1 = -1 (tie: -1.5 rounds up to -1)
      // 3: (3 + 1) >> 1 = 2 (tie: 1.5 rounds up to 2)
      sendControl(
        dut, first = true, last = true,
        cAddr = 0, ldC = 2, m = 3,
        requant = true, shift = 1, relu = false
      )
      sendRows(dut, Seq(Seq(1, 2), Seq(-1, -2), Seq(-3, 3)), last = true)
      readWrites(dut, Seq(0 -> Seq(1, 1), 1 -> Seq(0, -1), 2 -> Seq(-1, 2)))
    }
  }

  it should "accumulate three K-tiles before writing" in {
    test(new Accumulators(config)) { dut =>
      initialize(dut)

      // Tile 1: first = true, last = false
      sendControl(
        dut, first = true, last = false,
        cAddr = 0, ldC = 4, m = 2
      )
      sendRows(dut, Seq(Seq(1, 2), Seq(3, 4)), last = false)

      // Tile 2: first = false, last = false
      sendControl(
        dut, first = false, last = false,
        cAddr = 0, ldC = 4, m = 2
      )
      sendRows(dut, Seq(Seq(10, 20), Seq(30, 40)), last = false)

      // Tile 3: first = false, last = true
      sendControl(
        dut, first = false, last = true,
        cAddr = 0, ldC = 4, m = 2
      )
      sendRows(dut, Seq(Seq(100, 200), Seq(300, 400)), last = true)

      // Expected: sum of all 3 tiles (111, 222), (333, 444)
      readWrites(dut, Seq(0 -> Seq(111, 222), 2 -> Seq(333, 444)))
    }
  }

  it should "handle backpressure on the result write interface" in {
    test(new Accumulators(config)) { dut =>
      initialize(dut)

      sendControl(
        dut, first = true, last = true,
        cAddr = 0, ldC = 2, m = 4
      )
      sendRows(dut, Seq(Seq(5, 6), Seq(7, 8), Seq(9, 10), Seq(11, 12)), last = true)

      // Wait until resWr is valid with row 0.
      while (!dut.io.resWr.valid.peekBoolean()) {
        dut.clock.step()
      }

      // Drop ready on the first cycle resWr.valid is high.
      // With m >= 3, pipeP has row 0, pipeRnd has row 1, pipeR has row 2 (pipeR_valid is true),
      // and row 2's data is waiting on memRdData.
      dut.io.resWr.ready.poke(false.B)
      for (_ <- 0 until 5) {
        dut.io.resWr.valid.expect(true.B)
        dut.io.resWr.bits.rowAddr.expect(0.U)
        dut.io.resWr.bits.data(0).expect(5.S)
        dut.io.resWr.bits.data(1).expect(6.S)
        dut.io.done.valid.expect(false.B)
        dut.clock.step()
      }

      // Accept row 0.
      dut.io.resWr.ready.poke(true.B)
      dut.clock.step()

      // Assert backpressure again for row 1.
      dut.io.resWr.ready.poke(false.B)
      for (_ <- 0 until 5) {
        dut.io.resWr.valid.expect(true.B)
        dut.io.resWr.bits.rowAddr.expect(1.U)
        dut.io.resWr.bits.data(0).expect(7.S)
        dut.io.resWr.bits.data(1).expect(8.S)
        dut.io.done.valid.expect(false.B)
        dut.clock.step()
      }

      // Accept row 1.
      dut.io.resWr.ready.poke(true.B)
      dut.clock.step()

      // Assert backpressure for row 2 (which had to survive the memory stall in Stage R).
      dut.io.resWr.ready.poke(false.B)
      for (_ <- 0 until 5) {
        dut.io.resWr.valid.expect(true.B)
        dut.io.resWr.bits.rowAddr.expect(2.U)
        dut.io.resWr.bits.data(0).expect(9.S)
        dut.io.resWr.bits.data(1).expect(10.S)
        dut.io.done.valid.expect(false.B)
        dut.clock.step()
      }

      // Accept row 2.
      dut.io.resWr.ready.poke(true.B)
      dut.clock.step()

      // Assert backpressure for row 3 (final row).
      dut.io.resWr.ready.poke(false.B)
      for (_ <- 0 until 5) {
        dut.io.resWr.valid.expect(true.B)
        dut.io.resWr.bits.rowAddr.expect(3.U)
        dut.io.resWr.bits.data(0).expect(11.S)
        dut.io.resWr.bits.data(1).expect(12.S)
        dut.io.done.valid.expect(false.B)
        dut.clock.step()
      }

      // Accept row 3.
      dut.io.resWr.ready.poke(true.B)
      dut.io.done.valid.expect(true.B)
      dut.clock.step()

      dut.io.resWr.valid.expect(false.B)
    }
  }

  it should "accumulate and write the maximum number of rows (m = maxM)" in {
    test(new Accumulators(config)) { dut =>
      initialize(dut)

      val maxM = config.maxM
      val rows = (0 until maxM).map(r => Seq(r * 2 + 1, r * 2 + 2))

      sendControl(
        dut, first = true, last = true,
        cAddr = 0, ldC = 4, m = maxM
      )
      sendRows(dut, rows, last = true)

      // Expected row addresses: r * (ldC / cols) = r * (4 / 2) = r * 2.
      val expected = rows.zipWithIndex.map { case (vals, r) =>
        (r * 2) -> vals
      }
      readWrites(dut, expected)
    }
  }
}
