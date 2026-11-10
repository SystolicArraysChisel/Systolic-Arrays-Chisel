package tpu.compute.datapath

import chiseltest._
import chisel3._
import ifaces._
import org.scalatest.flatspec.AnyFlatSpec

import scala.collection.mutable.ArrayBuffer

class DataMoverSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "DataMover"

  private val c = SAConfig(
    rows = 4,
    cols = 4,
    inDepth = 64,
    outDepth = 64,
    maxM = 16
  )
  private val L = c.arrayLatency
  private val R = c.rows

  case class Tile(aAddr: Int, bAddr: Int, m: Int, waitAcc: Boolean = false, ldA: Int = 4, ldB: Int = 4) {
    def wRows: Seq[Int] = (R - 1 to 0 by -1).map(r => (bAddr + r * ldB) >> c.bankBits)
    def aRows: Seq[Int] = (0 until m).map(i => (aAddr + i * ldA) >> c.bankBits)
  }

  /** What reached the array: (cycle, row) of every weight and activation
    * beat, and the cycles with aFirst. Every bank of input row r holds r + bank,
    * so element 0 of a beat names its row.
    */
  case class Trace(w: Seq[(Int, Int)], a: Seq[(Int, Int)], first: Seq[Int])

  /** Feeds tiles into cmd as soon as they are accepted, and models both
    * buffer ports with one cycle of latency. Port A is stalled (as by a host
    * write) in the cycles of `stalls`; accFree is pulsed in the cycles of
    * `accFree`.
    */
  private def run(
      dut: DataMover,
      tiles: Seq[Tile],
      cycles: Int,
      stalls: Int => Boolean = _ => false,
      accFree: Set[Int] = Set()
  ): Trace = {
    val w = ArrayBuffer[(Int, Int)]()
    val a = ArrayBuffer[(Int, Int)]()
    val first = ArrayBuffer[Int]()
    var pendA: Option[Int] = None
    var pendW: Option[Int] = None
    var next = 0

    def drive(valid: Bool, bits: Vec[SInt], row: Option[Int]): Unit = {
      valid.poke(row.isDefined.B)
      row.foreach(r => bits.zipWithIndex.foreach { case (b, i) => b.poke((r + i).S) })
    }

    for (t <- 0 until cycles) {
      drive(dut.io.rdResp.valid, dut.io.rdResp.bits, pendA)
      drive(dut.io.wRdResp.valid, dut.io.wRdResp.bits, pendW)
      dut.io.rdReq.ready.poke((!stalls(t)).B)
      dut.io.wRdReq.ready.poke(true.B)
      dut.io.accFree.poke(accFree(t).B)

      val cmdValid = next < tiles.length
      dut.io.cmd.valid.poke(cmdValid.B)
      if (cmdValid) {
        val tl = tiles(next)
        dut.io.cmd.bits.aAddr.poke(tl.aAddr.U)
        dut.io.cmd.bits.bAddr.poke(tl.bAddr.U)
        dut.io.cmd.bits.ldA.poke(tl.ldA.U)
        dut.io.cmd.bits.ldB.poke(tl.ldB.U)
        dut.io.cmd.bits.m.poke(tl.m.U)
        dut.io.cmd.bits.waitAcc.poke(tl.waitAcc.B)
      }

      if (dut.io.wOut.valid.peek().litToBoolean)
        w += ((t, dut.io.wOut.bits(0).peek().litValue.toInt))
      if (dut.io.aOut.valid.peek().litToBoolean)
        a += ((t, dut.io.aOut.bits(0).peek().litValue.toInt))
      if (dut.io.aFirst.peek().litToBoolean) first += t

      pendA =
        if (dut.io.rdReq.valid.peek().litToBoolean && !stalls(t))
          Some(dut.io.rdReq.bits.rowAddr.peek().litValue.toInt)
        else None
      pendW =
        if (dut.io.wRdReq.valid.peek().litToBoolean)
          Some(dut.io.wRdReq.bits.rowAddr.peek().litValue.toInt)
        else None
      if (cmdValid && dut.io.cmd.ready.peek().litToBoolean) next += 1

      dut.clock.step()
    }
    Trace(w.toSeq, a.toSeq, first.toSeq)
  }

  /** Rows in order, aFirst on the first row of every tile, and every weight
    * beat of tile k+1 within [t0(k) + arrayLatency, t0(k+1) - 1].
    */
  private def check(tiles: Seq[Tile], tr: Trace): Unit = {
    assert(tr.w.map(_._2) == tiles.flatMap(_.wRows), "weight rows")
    assert(tr.a.map(_._2) == tiles.flatMap(_.aRows), "activation rows")

    val live = tiles.zipWithIndex.filter(_._1.m > 0)
    val starts = live.map(_._1.aRows.head)
    val t0 = tr.first
    assert(t0.length == live.length, "one aFirst per tile with rows")
    assert(t0.map(t => tr.a.find(_._1 == t).map(_._2).get) == starts, "aFirst on each tile's first row")

    val wBeats = tr.w.map(_._1).grouped(R).toSeq
    live.zip(t0).sliding(2).foreach {
      case Seq(((_, _), tPrev), ((_, j), tNext)) =>
        wBeats(j).foreach { tw =>
          assert(tw >= tPrev + L && tw <= tNext - 1, s"tile $j weight beat at $tw outside [${tPrev + L}, ${tNext - 1}]")
        }
      case _ =>
    }
    live.headOption.foreach { case (_, j) => assert(wBeats(j).last < t0.head) }
  }

  it should "overlap the next tile's weights with the current activations, without gaps" in {
    for (m <- Seq(3, 12)) {
      test(new DataMover(c)) { dut =>
        val tiles = Seq(Tile(0, 32, m), Tile(4, 36, m), Tile(8, 40, m))
        val tr = run(dut, tiles, 80)
        check(tiles, tr)
        // Steady state: one tile every max(m, arrayLatency + rows) cycles.
        tr.first.sliding(2).foreach { case Seq(p, q) => assert(q - p == (m max (L + R)), s"m = $m") }
      }
    }
  }

  it should "keep the order and the weight window under random port-A stalls" in {
    val rnd = new scala.util.Random(3)
    val stall = Seq.fill(400)(rnd.nextInt(4) == 0)
    test(new DataMover(c)) { dut =>
      val tiles = Seq(Tile(0, 32, 5), Tile(16, 36, 1), Tile(4, 40, 9), Tile(8, 44, 2), Tile(12, 48, 6))
      check(tiles, run(dut, tiles, 400, stall))
    }
  }

  it should "hold a waitAcc tile's activations until accFree, but preload its weights" in {
    test(new DataMover(c)) { dut =>
      val tiles = Seq(Tile(0, 32, 4), Tile(4, 36, 4, waitAcc = true))
      val tr = run(dut, tiles, 80, accFree = Set(50))
      check(tiles, tr)
      assert(tr.first(1) > 50)
      assert(tr.w.last._1 < 50, "weights loaded while waiting")
    }
  }

  it should "not wait again when accFree came before the tile that needs it" in {
    test(new DataMover(c)) { dut =>
      val tiles = Seq(Tile(0, 32, 12), Tile(4, 36, 12, waitAcc = true))
      val tr = run(dut, tiles, 60, accFree = Set(2))
      check(tiles, tr)
      assert(tr.first(1) - tr.first(0) == 12)
    }
  }

  it should "skip a tile with no activation rows" in {
    test(new DataMover(c)) { dut =>
      val tiles = Seq(Tile(0, 32, 3), Tile(4, 36, 0), Tile(8, 40, 2))
      check(tiles, run(dut, tiles, 60))
    }
  }
}
