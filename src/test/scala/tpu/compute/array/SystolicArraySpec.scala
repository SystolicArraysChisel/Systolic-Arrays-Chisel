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

  it should "load the next tile while computing and run both tiles without a gap" in {
    val c = cfg(4)
    val rnd = new scala.util.Random(7)
    def int8() = rnd.nextInt(256) - 128
    val b0 = Seq.fill(4, 4)(int8())
    val b1 = Seq.fill(4, 4)(int8())
    // Tightest schedule: b1 shifts in at [arrayLatency, arrayLatency + rows - 1],
    // and tile 1 starts the cycle right after.
    val m0 = c.arrayLatency + c.rows
    val a0 = Seq.fill(m0, 4)(int8())
    val a1 = Seq.fill(5, 4)(int8())
    val load = c.arrayLatency until c.arrayLatency + c.rows
    val aRows = a0 ++ a1

    test(new SystolicArray(c)) { dut =>
      dut.io.aIn.valid.poke(false.B)
      dut.io.aFirst.poke(false.B)
      for (r <- b0.indices.reverse) {
        dut.io.wIn.valid.poke(true.B)
        b0(r).zipWithIndex.foreach { case (v, j) => dut.io.wIn.bits(j).poke(v.S) }
        dut.clock.step()
      }

      val seen = scala.collection.mutable.ArrayBuffer[(Int, Seq[BigInt])]()
      for (t <- 0 until aRows.length + c.arrayLatency + 1) {
        val w = load.indexOf(t)
        dut.io.wIn.valid.poke((w >= 0).B)
        if (w >= 0)
          b1(c.rows - 1 - w).zipWithIndex.foreach { case (v, j) => dut.io.wIn.bits(j).poke(v.S) }
        dut.io.aIn.valid.poke((t < aRows.length).B)
        dut.io.aFirst.poke((t == 0 || t == m0).B)
        aRows.lift(t).foreach(_.zipWithIndex.foreach { case (v, i) => dut.io.aIn.bits(i).poke(v.S) })
        if (dut.io.out.valid.peek().litToBoolean)
          seen += ((t, dut.io.out.bits.map(_.peek().litValue)))
        dut.clock.step()
      }

      val want = (matmul(a0, b0) ++ matmul(a1, b1)).zipWithIndex.map { case (r, t) =>
        (t + c.arrayLatency, r)
      }
      assert(seen.toSeq == want)
    }
  }
}
