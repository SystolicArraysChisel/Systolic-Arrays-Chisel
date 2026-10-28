package tpu.compute.array

import chisel3._
import chiseltest._
import ifaces._

import scala.collection.mutable.ArrayBuffer

object ArrayTestUtil {

  // shortcut: local matmul, swap for the shared reference model once #7 lands.
  def matmul(a: Seq[Seq[Int]], b: Seq[Seq[Int]]): Seq[Seq[BigInt]] =
    a.map(row =>
      b.head.indices.map(j => BigInt(row.indices.map(r => row(r) * b(r)(j)).sum))
    )

  /** Loads tile b (last row first), then streams aRows into aIn, where None
    * is a cycle with valid = 0. Returns (cycle, row) for every valid out,
    * with cycle 0 = the cycle aRows(0) is presented.
    */
  def runTile(
      dut: SystolicArray,
      c: SAConfig,
      b: Seq[Seq[Int]],
      aRows: Seq[Option[Seq[Int]]]
  ): Seq[(Int, Seq[BigInt])] = {
    dut.io.aIn.valid.poke(false.B)
    for (r <- b.indices.reverse) {
      dut.io.wIn.valid.poke(true.B)
      b(r).zipWithIndex.foreach { case (v, j) => dut.io.wIn.bits(j).poke(v.S) }
      dut.clock.step()
    }
    dut.io.wIn.valid.poke(false.B)

    val seen = ArrayBuffer[(Int, Seq[BigInt])]()
    for (t <- 0 until aRows.length + c.arrayLatency + 1) {
      val row = aRows.lift(t).flatten
      dut.io.aIn.valid.poke(row.isDefined.B)
      row.foreach(_.zipWithIndex.foreach { case (v, i) => dut.io.aIn.bits(i).poke(v.S) })
      if (dut.io.out.valid.peek().litToBoolean)
        seen += ((t, dut.io.out.bits.map(_.peek().litValue)))
      dut.clock.step()
    }
    dut.io.aIn.valid.poke(false.B)
    seen.toSeq
  }

  /** Every valid aIn row comes out exactly arrayLatency cycles later. */
  def expected(
      c: SAConfig,
      b: Seq[Seq[Int]],
      aRows: Seq[Option[Seq[Int]]]
  ): Seq[(Int, Seq[BigInt])] =
    aRows.zipWithIndex.collect { case (Some(a), t) =>
      (t + c.arrayLatency, matmul(Seq(a), b).head)
    }
}
