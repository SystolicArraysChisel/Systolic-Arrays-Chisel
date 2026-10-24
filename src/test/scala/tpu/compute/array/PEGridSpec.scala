package tpu.compute.array

import chisel3._
import chiseltest._
import ifaces._
import org.scalatest.flatspec.AnyFlatSpec

import ArrayTestUtil._

class PEGridSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "PEGrid"

  private def cfg(n: Int) = SAConfig(
    rows = n,
    cols = n,
    inDepth = 64,
    outDepth = 64,
    maxM = 8
  )

  /** Loads tile b (last row first), then feeds a already skewed: at cycle t,
    * PE row i gets A[t - i][i]. Checks that column j of the bottom row shows
    * the result of A row m at cycle m + rows + j.
    */
  private def check(dut: PEGrid, c: SAConfig, b: Seq[Seq[Int]], a: Seq[Seq[Int]]): Unit = {
    for (r <- b.indices.reverse) {
      dut.io.wIn.valid.poke(true.B)
      b(r).zipWithIndex.foreach { case (v, j) => dut.io.wIn.bits(j).poke(v.S) }
      dut.clock.step()
    }
    dut.io.wIn.valid.poke(false.B)

    val expected = matmul(a, b)
    for (t <- 0 until a.length + c.rows + c.cols) {
      for (i <- 0 until c.rows) {
        val v = a.lift(t - i).map(_(i)).getOrElse(0)
        dut.io.aIn(i).poke(v.S)
      }
      for (j <- 0 until c.cols) {
        val m = t - c.rows - j
        if (m >= 0 && m < a.length)
          dut.io.psumOut(j).expect(expected(m)(j).S, s"row $m col $j at cycle $t")
      }
      dut.clock.step()
    }
  }

  it should "compute the WorkloadSplit.md example (two K-tiles, back to back)" in {
    val c = cfg(2)
    test(new PEGrid(c)) { dut =>
      // P0 = [[1,2],[5,6],[1,0]]
      check(dut, c, Seq(Seq(1, 0), Seq(0, 1)), Seq(Seq(1, 2), Seq(5, 6), Seq(1, 0)))
      // P1 = [[11,-1],[23,-1],[3,-3]]
      check(dut, c, Seq(Seq(1, 1), Seq(2, -1)), Seq(Seq(3, 4), Seq(7, 8), Seq(-1, 2)))
    }
  }

  for (n <- Seq(1, 4)) {
    it should s"match the reference on random signed ${n}x$n tiles" in {
      val c = cfg(n)
      val rnd = new scala.util.Random(n)
      def int8() = rnd.nextInt(256) - 128
      test(new PEGrid(c)) { dut =>
        check(dut, c, Seq.fill(n, n)(int8()), Seq.fill(6, n)(int8()))
      }
    }
  }
}
