package tpu.ref

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.io.Source

import ifaces.SAConfig
import chisel3._
import tpu.io.{BoardConfig, CommandDecoder, DecoderBench, DecoderWithSequencer}

/** Checks the Scala model and the CommandDecoder against the vectors generated
  * by the former Python host, also checked by the C++ host, so all sides agree
  * bit for bit. Line format: host/README.md.
  */
class RefVectorsSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "Scala reference model vs Python host"

  val lines: Seq[Array[String]] = {
    val src = Source.fromResource("ref_vectors.txt")
    try src.getLines().filterNot(l => l.isEmpty || l.startsWith("#")).map(_.split(" ")).toVector
    finally src.close()
  }
  def kind(k: String): Seq[Array[String]] = lines.filter(_.head == k)

  it should "load the vectors" in {
    for (k <- Seq("Q", "S", "P", "K", "M", "FM", "FW", "FR")) assert(kind(k).nonEmpty, s"no $k vectors")
  }

  it should "quantize like Python" in {
    for (Array(_, x, s, q) <- kind("Q"))
      assert(Quant.quantize(x.toDouble, s.toDouble) == q.toInt, s"quantize($x, $s)")
  }

  it should "choose the same scale as Python" in {
    for (l <- kind("S"))
      assert(Quant.chooseScale(l.drop(2).map(_.toDouble)) == l(1).toDouble, l.mkString(" "))
  }

  it should "post-process like Python" in {
    for (Array(_, acc, rq, relu, shift, y) <- kind("P"))
      assert(RefModel.postProcess(acc.toLong, rq == "1", relu == "1", shift.toInt) == y.toLong,
        s"acc $acc requant $rq relu $relu shift $shift")
  }

  it should "choose the same shift as Python" in {
    for (l <- kind("K"))
      assert(Quant.chooseShift(l.drop(2).map(_.toLong)) == l(1).toInt, l.mkString(" "))
  }

  it should "compute the same MATMUL results as Python" in {
    for (l <- kind("M")) {
      val Array(head, a, b, c) = l.mkString(" ").split(" \\| ")
      val Array(_, m, k, n, rq, relu, shift) = head.split(" ")
      def mat(s: String, rows: Int, cols: Int) = s.split(" ").map(_.toLong).toSeq.grouped(cols).toSeq
      val got = RefModel.matmul(mat(a, m.toInt, k.toInt), mat(b, k.toInt, n.toInt), rq == "1", relu == "1", shift.toInt)
      assert(got == mat(c, m.toInt, n.toInt), head)
    }
  }

  // ---- frames built by Python, fed to the real decoder ----

  val c = SAConfig()
  val b = BoardConfig(clockHz = 400_000, baud = 100_000, frameTimeout = 200)
  def hex(s: String): Seq[Int] = s.grouped(2).map(Integer.parseInt(_, 16)).toSeq

  /** MATMUL fields as the decoder hands them to the sequencer. */
  case class SeenInstr(
      aAddr: Int, bAddr: Int, cAddr: Int, m: Int, k: Int, n: Int,
      ldA: Int, ldB: Int, ldC: Int, requant: Boolean, relu: Boolean, shift: Int
  )

  // The vectors use random sizes the real Sequencer cannot run, so the
  // decoder is tested alone, with instr always ready.
  it should "decode Python's MATMUL frames into the right instruction" in {
    test(new CommandDecoder(c, b)) { dut =>
      dut.io.instr.ready.poke(true.B)
      dut.io.tx.ready.poke(true.B)
      val bytes = kind("FM").flatMap(l => hex(l(1)))
      val expected = kind("FM").map { l =>
        val f = l.drop(2).map(_.toInt)
        SeenInstr(f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8), f(9) == 1, f(10) == 1, f(11))
      }
      val seen = scala.collection.mutable.ArrayBuffer[SeenInstr]()
      var i = 0
      var cycles = 0
      while ((i < bytes.length || cycles < 10) && cycles < 10000) {
        dut.io.rx.valid.poke((i < bytes.length).B)
        dut.io.rx.bits.poke((if (i < bytes.length) bytes(i) else 0).U)
        if (i < bytes.length && dut.io.rx.ready.peekBoolean()) i += 1
        if (dut.io.instr.valid.peekBoolean()) {
          val x = dut.io.instr.bits
          seen += SeenInstr(
            x.aAddr.peekInt().toInt, x.bAddr.peekInt().toInt, x.cAddr.peekInt().toInt,
            x.m.peekInt().toInt, x.k.peekInt().toInt, x.n.peekInt().toInt,
            x.ldA.peekInt().toInt, x.ldB.peekInt().toInt, x.ldC.peekInt().toInt,
            x.requant.peekBoolean(), x.relu.peekBoolean(), x.shift.peekInt().toInt
          )
        }
        dut.clock.step()
        if (i >= bytes.length) cycles += 1
      }
      assert(seen.toSeq == expected)
      assert(!dut.io.error.peekBoolean())
    }
  }

  it should "decode Python's WRITE and READ frames" in {
    test(new DecoderWithSequencer(c, b)) { dut =>
      val t = new DecoderBench(dut)
      val expWrites = kind("FW").flatMap { l =>
        t.send(hex(l(1)))
        val addr = l(2).toInt
        l.drop(3).map(_.toInt).zipWithIndex.map { case (d, i) => (addr + i, d) }
      }
      var replyLen = 0
      val expReads = kind("FR").flatMap { l =>
        t.send(hex(l(1)))
        val Array(addr, count, size): Array[Int] = l.drop(2).map(_.toInt)
        replyLen += count * size
        (0 until count).map(addr + _)
      }
      t.drain()
      assert(t.writes == expWrites)
      assert(t.readReqs == expReads)
      assert(t.received.length == replyLen)
      assert(!t.status().error)
    }
  }
}
