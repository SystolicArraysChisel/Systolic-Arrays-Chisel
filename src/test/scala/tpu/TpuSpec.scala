package tpu

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable
import scala.util.Random

import ifaces.SAConfig
import tpu.io.{BoardConfig, HostFrames}
import tpu.io.UartTestUtil.frameBits
import tpu.ref.RefModel
import tpu.ref.RefModel.Mat

/** The Python host's side of the link, in Scala: sends frames, waits for the
  * replies, and runs whole MATMULs. Subclasses decide how bytes reach the TPU.
  */
abstract class Host(c: SAConfig) {
  val received = mutable.Queue[Int]() // reply bytes not yet consumed
  var cycle = 0L

  /** One clock cycle: drive the next input, collect any output byte. */
  def step(): Unit

  /** Queue `bytes` for the TPU, back to back. */
  def send(bytes: Seq[Int]): Unit

  /** True while queued bytes have not all reached the TPU. */
  def sending: Boolean

  /** Step until everything queued is sent, then wait for `n` reply bytes. */
  def receive(n: Int, limit: Long = 200000): Seq[Int] = {
    val start = cycle
    while (sending || received.length < n) {
      assert(cycle - start < limit, s"no reply: ${received.length} of $n bytes after $limit cycles")
      step()
    }
    Seq.fill(n)(received.dequeue())
  }

  def status(): HostFrames.StatusReply = {
    send(HostFrames.status)
    HostFrames.parseStatus(receive(9))
  }

  def write(addr: Int, data: Seq[Long]): Unit = {
    send(HostFrames.write(addr, data.map(_.toInt)))
    receive(0)
  }

  /** READ `count` elements of `size` bytes, as signed values. */
  def read(addr: Int, count: Int, size: Int): Seq[Long] = {
    send(HostFrames.read(addr, count, size))
    receive(count * size).grouped(size).map { bs =>
      val v = bs.zipWithIndex.map { case (x, i) => x.toLong << (8 * i) }.sum
      val bits = 8 * size
      if (v >= (1L << (bits - 1))) v - (1L << bits) else v
    }.toSeq
  }

  private def roundUp(x: Int, mult: Int): Int = (x + mult - 1) / mult * mult

  /** C = A . B the way the Python host does it: zero-pad K and N to a multiple
    * of banks, WRITE A then B, MATMUL, poll STATUS until idle, READ C. Returns
    * the m x n result and the last STATUS.
    */
  def matmul(a: Mat, b: Mat, requant: Boolean = false, relu: Boolean = false,
             shift: Int = 0): (Mat, HostFrames.StatusReply) = {
    val (m, k, n) = (a.length, b.length, b.head.length)
    val (kp, np) = (roundUp(k, c.banks), roundUp(n, c.banks))
    val aFlat = a.flatMap(_.padTo(kp, 0L))
    val bFlat = b.map(_.padTo(np, 0L)).padTo(kp, Seq.fill(np)(0L)).flatten
    val aAddr = 0
    val bAddr = roundUp(aFlat.length, c.banks)
    write(aAddr, aFlat)
    write(bAddr, bFlat)

    val flags = (if (requant) 1 else 0) | (if (relu) 2 else 0)
    send(HostFrames.Matmul(aAddr, bAddr, 0, m, kp, np, kp, np, np, flags, shift).bytes)
    var st = status()
    while (st.busy) st = status()

    val cFlat = read(0, m * np, if (requant) 1 else 4)
    (cFlat.grouped(np).map(_.take(n)).toSeq, st)
  }
}

/** Talks to TpuCore one byte per cycle, the way RxFifo and UartTx would. */
class ByteHost(dut: TpuCore, c: SAConfig) extends Host(c) {
  private val toSend = mutable.Queue[Int]()
  dut.io.rxOverrun.poke(false.B)
  dut.io.tx.ready.poke(true.B)

  def send(bytes: Seq[Int]): Unit = toSend ++= bytes
  def sending: Boolean = toSend.nonEmpty

  def step(): Unit = {
    dut.io.rx.valid.poke(toSend.nonEmpty.B)
    dut.io.rx.bits.poke(toSend.headOption.getOrElse(0).U)
    val rxFire = toSend.nonEmpty && dut.io.rx.ready.peekBoolean()
    if (dut.io.tx.valid.peekBoolean()) received.enqueue(dut.io.tx.bits.peekInt().toInt)
    dut.clock.step()
    cycle += 1
    if (rxFire) toSend.dequeue()
  }
}

/** Talks to Tpu through its pins: drives rxd with UART frames and decodes txd
  * the way a PC's UART would, sampling each bit in its middle.
  */
class SerialHost(dut: Tpu, c: SAConfig, b: BoardConfig) extends Host(c) {
  private val cpb = b.cyclesPerBit
  private val line = mutable.Queue[Boolean]() // rxd levels, one per cycle

  // txd receiver: -1 = idle, otherwise cycles since the start bit's edge
  private var rxT = -1
  private var rxByte = 0
  private var prevTxd = true

  def send(bytes: Seq[Int]): Unit = line ++= bytes.flatMap(frameBits(_).flatMap(Seq.fill(cpb)(_)))
  def sending: Boolean = line.nonEmpty

  def step(): Unit = {
    dut.io.rxd.poke((if (line.nonEmpty) line.dequeue() else true).B)
    val txd = dut.io.txd.peekBoolean()
    if (rxT < 0) {
      if (prevTxd && !txd) { rxT = 0; rxByte = 0 }
    } else {
      rxT += 1
      if ((rxT - cpb / 2) % cpb == 0) {
        val bit = (rxT - cpb / 2) / cpb // 0 = start, 1..8 = data, 9 = stop
        if (bit >= 1 && bit <= 8 && txd) rxByte |= 1 << (bit - 1)
        if (bit == 9) {
          assert(txd, s"cycle $cycle: stop bit of a byte from the TPU is 0")
          received.enqueue(rxByte)
          rxT = -1
        }
      }
    }
    prevTxd = txd
    dut.clock.step()
    cycle += 1
  }
}

class TpuSpec extends AnyFlatSpec with ChiselScalatestTester {

  val c = SAConfig()
  val b = BoardConfig(clockHz = 400_000, baud = 100_000) // 4 cycles per bit

  val example: (Mat, Mat, Mat) = (
    Seq(Seq(1, 2, 3, 4), Seq(5, 6, 7, 8), Seq(1, 0, -1, 2)).map(_.map(_.toLong)),
    Seq(Seq(1, 0), Seq(0, 1), Seq(1, 1), Seq(2, -1)).map(_.map(_.toLong)),
    Seq(Seq(12, 1), Seq(28, 5), Seq(4, -3)).map(_.map(_.toLong))
  )

  /** Runs `body` on a fresh TpuCore, then checks that no error was flagged. */
  def withCore(body: Host => Unit): Unit =
    test(new TpuCore(c, b)) { dut =>
      dut.clock.setTimeout(0)
      val host = new ByteHost(dut, c)
      body(host)
      val st = host.status()
      assert(!st.error && !st.overrun, s"final STATUS: $st")
      dut.io.error.expect(false.B)
    }

  def check(host: Host, a: Mat, bm: Mat, requant: Boolean = false, relu: Boolean = false,
            shift: Int = 0): HostFrames.StatusReply = {
    val (got, st) = host.matmul(a, bm, requant, relu, shift)
    assert(got == RefModel.matmul(a, bm, requant, relu, shift),
      s"${a.length}x${bm.length}x${bm.head.length} requant=$requant relu=$relu shift=$shift")
    assert(!st.error && !st.overrun, s"STATUS after MATMUL: $st")
    st
  }

  behavior of "TpuCore"

  it should "compute the worked example of the workload document" in {
    withCore { host =>
      val (a, bm, want) = example
      assert(host.matmul(a, bm)._1 == want)
    }
  }

  it should "report m * k * n useful PE-cycles and a plausible cycle count" in {
    withCore { host =>
      val rnd = new Random(1)
      val st = check(host, RefModel.randomMatrix(rnd, 4, 8), RefModel.randomMatrix(rnd, 8, 8))
      assert(st.busyPe == 4L * 8 * 8)
      // 4 tiles of at least: weight load + m rows + array latency
      assert(st.cycles >= 4L * (c.rows + 4 + c.arrayLatency), s"cycles = ${st.cycles}")
    }
  }

  it should "tile over K and N with a tall A, still being written back when the next tile starts" in {
    withCore { host =>
      val rnd = new Random(2)
      check(host, RefModel.randomMatrix(rnd, 20, 8), RefModel.randomMatrix(rnd, 8, 12))
    }
  }

  it should "use every row of the accumulators (m = maxM)" in {
    withCore { host =>
      val rnd = new Random(3)
      check(host, RefModel.randomMatrix(rnd, c.maxM, 4), RefModel.randomMatrix(rnd, 4, 8))
    }
  }

  it should "zero-pad K and N that are not multiples of the array size" in {
    withCore { host =>
      val rnd = new Random(4)
      check(host, RefModel.randomMatrix(rnd, 5, 7), RefModel.randomMatrix(rnd, 7, 6))
    }
  }

  it should "requantize, apply ReLU, and run several MATMULs in a row" in {
    withCore { host =>
      val rnd = new Random(5)
      check(host, RefModel.randomMatrix(rnd, 6, 8), RefModel.randomMatrix(rnd, 8, 4), requant = true, shift = 9)
      check(host, RefModel.randomMatrix(rnd, 6, 8), RefModel.randomMatrix(rnd, 8, 4), requant = true,
        relu = true, shift = 7)
      check(host, RefModel.randomMatrix(rnd, 3, 4), RefModel.randomMatrix(rnd, 4, 4))
    }
  }

  behavior of "Tpu"

  it should "compute the worked example through the serial pins" in {
    test(new Tpu(c, b)) { dut =>
      dut.clock.setTimeout(0)
      val host = new SerialHost(dut, c, b)
      val (a, bm, want) = example
      val (got, st) = host.matmul(a, bm)
      assert(got == want)
      assert(!st.error && !st.overrun, s"STATUS after MATMUL: $st")
      dut.io.debug.error.expect(false.B)
      dut.io.debug.overrun.expect(false.B)
      dut.io.debug.busy.expect(false.B)
    }
  }
}
