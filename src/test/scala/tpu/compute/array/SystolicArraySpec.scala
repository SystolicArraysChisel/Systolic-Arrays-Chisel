package tpu.compute.array

import chisel3._
import chiseltest._
import ifaces._
import org.scalatest.flatspec.AnyFlatSpec

import ArrayTestUtil._

class SystolicArraySpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "SystolicArray"

  private def cfg(n: Int) = SAConfig(
    rows = n,
    cols = n,
    inDepth = 64,
    outDepth = 64,
    maxM = 8
  )

  it should "compute the WorkloadSplit.md example (two K-tiles, back to back)" in {
    val c = cfg(2)
    test(new SystolicArray(c)) { dut =>
      val a0 = Seq(Seq(1, 2), Seq(5, 6), Seq(1, 0)).map(Some(_))
      val b0 = Seq(Seq(1, 0), Seq(0, 1))
      assert(runTile(dut, c, b0, a0) == expected(c, b0, a0))

      val a1 = Seq(Seq(3, 4), Seq(7, 8), Seq(-1, 2)).map(Some(_))
      val b1 = Seq(Seq(1, 1), Seq(2, -1))
      val p1 = runTile(dut, c, b1, a1)
      assert(p1 == expected(c, b1, a1))
      assert(p1.map(_._2) == Seq(Seq(11, -1), Seq(23, -1), Seq(3, -3)).map(_.map(BigInt(_))))
    }
  }

  it should "keep gaps between aIn rows and emit out arrayLatency cycles later" in {
    val c = cfg(4)
    test(new SystolicArray(c)) { dut =>
      val b = Seq.tabulate(4, 4)((r, j) => r - j)
      val a = Seq(Some(Seq(1, 2, 3, 4)), None, None, Some(Seq(-5, 6, -7, 8)), None, Some(Seq(9, 9, 9, 9)))
      assert(runTile(dut, c, b, a) == expected(c, b, a))
    }
  }

  it should "work as a 1x1 single-MAC baseline" in {
    val c = cfg(1)
    test(new SystolicArray(c)) { dut =>
      val b = Seq(Seq(-2))
      val a = Seq(Some(Seq(3)), Some(Seq(-4)), None, Some(Seq(127)))
      assert(runTile(dut, c, b, a) == expected(c, b, a))
    }
  }
}
