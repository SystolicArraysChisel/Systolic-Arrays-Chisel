package tpu.io

import chisel3._
import chisel3.util._
import chiseltest._
import scala.collection.mutable
import scala.util.Random

import ifaces._
import tpu.compute.control.Sequencer
import tpu.io.HostProtocol._

/** Host-side frame builders and reply parsers, as the Python host would do it. */
object HostFrames {
  def le16(v: Int): Seq[Int] = Seq(v & 0xff, (v >> 8) & 0xff)

  def write(addr: Int, data: Seq[Int]): Seq[Int] =
    Seq(Op.Write) ++ le16(addr) ++ le16(data.length) ++ data.map(_ & 0xff)

  def read(addr: Int, count: Int, size: Int): Seq[Int] =
    Seq(Op.Read) ++ le16(addr) ++ le16(count) ++ Seq(size)

  val status: Seq[Int] = Seq(Op.Status)

  case class Matmul(
      aAddr: Int = 0,
      bAddr: Int = 0,
      cAddr: Int = 0,
      m: Int = 4,
      k: Int = 4,
      n: Int = 4,
      ldA: Int = 4,
      ldB: Int = 4,
      ldC: Int = 4,
      flags: Int = 0,
      shift: Int = 0
  ) {
    def bytes: Seq[Int] =
      Seq(Op.Matmul) ++ Seq(aAddr, bAddr, cAddr, m, k, n, ldA, ldB, ldC).flatMap(le16) ++
        Seq(flags, shift)
  }

  /** The 9 bytes of a STATUS reply. */
  case class StatusReply(busy: Boolean, error: Boolean, overrun: Boolean, cycles: Long, busyPe: Long)

  def parseStatus(r: Seq[Int]): StatusReply = {
    require(r.length == Status.replyBytes, s"STATUS reply has ${r.length} bytes")
    def le32(off: Int) = (0 until 4).map(i => r(off + i).toLong << (8 * i)).sum
    StatusReply(
      busy = ((r(0) >> Status.BusyBit) & 1) == 1,
      error = ((r(0) >> Status.ErrorBit) & 1) == 1,
      overrun = ((r(0) >> Status.OverrunBit) & 1) == 1,
      cycles = le32(1),
      busyPe = le32(5)
    )
  }
}

/** Tile commands as the real Sequencer issues them, one pair per tile. */
case class SeenTileCmd(aAddr: Int, bAddr: Int, ldA: Int, ldB: Int, m: Int)
case class SeenAccCtrl(
    first: Boolean, last: Boolean, cAddr: Int, ldC: Int, m: Int,
    relu: Boolean, requant: Boolean, shift: Int
)

object Tiles {

  /** Tiles a MATMUL must produce, K first inside each N tile (address fields
    * wrap at their width, as in hardware).
    */
  def expected(mm: HostFrames.Matmul, c: SAConfig): Seq[(SeenTileCmd, SeenAccCtrl)] =
    for {
      nOff <- 0 until mm.n by c.cols
      kOff <- 0 until mm.k by c.rows
    } yield (
      SeenTileCmd(
        (mm.aAddr + kOff) % (1 << c.addrW),
        (mm.bAddr + kOff * mm.ldB + nOff) % (1 << c.addrW),
        mm.ldA, mm.ldB, mm.m
      ),
      SeenAccCtrl(
        first = kOff == 0, last = kOff + c.rows == mm.k,
        (mm.cAddr + nOff) % (1 << c.outAddrW), mm.ldC, mm.m,
        relu = (mm.flags & 2) != 0, requant = (mm.flags & 1) != 0, mm.shift
      )
    )
}

/** CommandDecoder wired to the real Sequencer. The sequencer's status is also
  * brought out, so the test can compare STATUS replies with it.
  */
class DecoderWithSequencer(c: SAConfig, b: BoardConfig) extends Module {
  val io = IO(new Bundle {
    val rx = Flipped(Decoupled(UInt(8.W)))
    val tx = Decoupled(UInt(8.W))
    val rxOverrun = Input(Bool())
    val error = Output(Bool())
    val bufWr = Decoupled(new BufWritePort(c))
    val bufRdReq = Decoupled(new BufReadReq(c))
    val bufRdResp = Flipped(Decoupled(c.accT))
    val tileCmd = Decoupled(new TileCmd(c))
    val accCtrl = Decoupled(new AccCtrl(c))
    val tileDone = Flipped(Valid(Bool()))
    val status = Output(new SeqStatus)
  })
  val dec = Module(new CommandDecoder(c, b))
  val seq = Module(new Sequencer(c))

  dec.io.rx <> io.rx
  io.tx <> dec.io.tx
  dec.io.rxOverrun := io.rxOverrun
  io.error := dec.io.error
  io.bufWr <> dec.io.bufWr
  io.bufRdReq <> dec.io.bufRdReq
  dec.io.bufRdResp <> io.bufRdResp

  seq.io.instr <> dec.io.instr
  dec.io.status := seq.io.status
  io.status := seq.io.status
  io.tileCmd <> seq.io.tileCmd
  io.accCtrl <> seq.io.accCtrl
  seq.io.tileDone := io.tileDone
}

/** Cycle-by-cycle environment of a CommandDecoder and the real Sequencer: a
  * host that sends bytes, a UartTx that takes them, a buffer, and a data
  * mover/accumulators pair that takes tile commands and reports each tile done,
  * all modelled in Scala. Every cycle it also checks the ready/valid sender
  * rule on the decoder's outputs: once valid, an item stays valid and unchanged
  * until it is taken.
  */
class DecoderBench(dut: DecoderWithSequencer, seed: Int = 0, randomReady: Boolean = false) {
  private val rnd = new Random(seed)

  val toSend = mutable.Queue[Int]() // bytes from the host, waiting in "RxFifo"
  val received = mutable.ArrayBuffer[Int]() // bytes sent back on tx
  val writes = mutable.ArrayBuffer[(Int, Int)]() // (addr, signed data) on bufWr
  val readReqs = mutable.ArrayBuffer[Int]() // addresses on bufRdReq
  val resultMem = mutable.Map[Int, BigInt]().withDefaultValue(BigInt(0))

  // data mover / accumulators model
  val tileCmds = mutable.ArrayBuffer[SeenTileCmd]()
  val accCtrls = mutable.ArrayBuffer[SeenAccCtrl]()
  def tiles: Seq[(SeenTileCmd, SeenAccCtrl)] = tileCmds.zip(accCtrls).toSeq
  var holdTiles = false // never report a tile done: the sequencer stays busy
  var tileLatency = 3 // cycles from a tile's commands to its tileDone pulse
  var tilesDone = 0
  private var tileTimer = 0

  /** Length of the last run of cycles with busy high, as seen on the status. */
  var lastBusyRun = 0L
  private var wasBusy = false

  var overrunNext = false // drive rxOverrun for the next cycle only

  var readLatency = 2
  private var pendingRead: Option[(Int, Int)] = None // (addr, cycles left)
  private var respValid = false
  private var respData = BigInt(0)

  var cycle = 0
  var quiet = 0 // cycles since anything happened on any port

  private def maybe(p: Boolean): Boolean = p && (!randomReady || rnd.nextInt(3) != 0)

  // previous cycle's offered-but-not-taken items, for the sender-rule check
  private var heldTx: Option[BigInt] = None
  private var heldWr: Option[(BigInt, BigInt)] = None
  private var heldRd: Option[BigInt] = None

  private def peekTileCmd(): SeenTileCmd = {
    val t = dut.io.tileCmd.bits
    SeenTileCmd(
      t.aAddr.peekInt().toInt, t.bAddr.peekInt().toInt,
      t.ldA.peekInt().toInt, t.ldB.peekInt().toInt, t.m.peekInt().toInt
    )
  }

  private def peekAccCtrl(): SeenAccCtrl = {
    val a = dut.io.accCtrl.bits
    SeenAccCtrl(
      a.first.peekBoolean(), a.last.peekBoolean(), a.cAddr.peekInt().toInt,
      a.ldC.peekInt().toInt, a.m.peekInt().toInt,
      a.relu.peekBoolean(), a.requant.peekBoolean(), a.shift.peekInt().toInt
    )
  }

  private def checkHeld[T](name: String, held: Option[T], valid: Boolean, now: => T): Unit =
    held.foreach { h =>
      assert(valid, s"cycle $cycle: $name.valid dropped before the transfer")
      assert(now == h, s"cycle $cycle: $name.bits changed before the transfer: $h -> $now")
    }

  def step(): Unit = {
    // ---- drive inputs ----
    dut.io.rx.valid.poke(toSend.nonEmpty.B)
    dut.io.rx.bits.poke(toSend.headOption.getOrElse(0).U)
    val txReady = maybe(true)
    val wrReady = maybe(true)
    val rdReqReady = maybe(pendingRead.isEmpty && !respValid)
    val tcReady = maybe(true)
    val acReady = maybe(true)
    dut.io.tx.ready.poke(txReady.B)
    dut.io.bufWr.ready.poke(wrReady.B)
    dut.io.bufRdReq.ready.poke(rdReqReady.B)
    dut.io.bufRdResp.valid.poke(respValid.B)
    dut.io.bufRdResp.bits.poke(respData.S)
    dut.io.tileCmd.ready.poke(tcReady.B)
    dut.io.accCtrl.ready.poke(acReady.B)
    // a tile whose two commands were both taken is done tileLatency cycles later
    val tilePending = (tileCmds.length min accCtrls.length) > tilesDone
    val tileDone = tilePending && !holdTiles && tileTimer >= tileLatency
    dut.io.tileDone.valid.poke(tileDone.B)
    dut.io.tileDone.bits.poke(true.B)
    if (tileDone) { tilesDone += 1; tileTimer = 0 }
    else if (tilePending) tileTimer += 1
    dut.io.rxOverrun.poke(overrunNext.B)
    overrunNext = false

    // ---- observe outputs ----
    var active = tileDone || (tilePending && !holdTiles)
    val rxFire = toSend.nonEmpty && dut.io.rx.ready.peekBoolean()

    val txValid = dut.io.tx.valid.peekBoolean()
    checkHeld("tx", heldTx, txValid, dut.io.tx.bits.peekInt())
    heldTx = None
    if (txValid && txReady) { received += dut.io.tx.bits.peekInt().toInt; active = true }
    else if (txValid) heldTx = Some(dut.io.tx.bits.peekInt())

    val wrValid = dut.io.bufWr.valid.peekBoolean()
    def wrBits = (dut.io.bufWr.bits.addr.peekInt(), dut.io.bufWr.bits.data.peekInt())
    checkHeld("bufWr", heldWr, wrValid, wrBits)
    heldWr = None
    if (wrValid && wrReady) { writes += ((wrBits._1.toInt, wrBits._2.toInt)); active = true }
    else if (wrValid) heldWr = Some(wrBits)

    val rdValid = dut.io.bufRdReq.valid.peekBoolean()
    checkHeld("bufRdReq", heldRd, rdValid, dut.io.bufRdReq.bits.addr.peekInt())
    heldRd = None
    val rdFire = rdValid && rdReqReady
    if (rdFire) { readReqs += dut.io.bufRdReq.bits.addr.peekInt().toInt; active = true }
    else if (rdValid) heldRd = Some(dut.io.bufRdReq.bits.addr.peekInt())

    val respFire = respValid && dut.io.bufRdResp.ready.peekBoolean()

    val tcValid = dut.io.tileCmd.valid.peekBoolean()
    if (tcValid && tcReady) tileCmds += peekTileCmd()
    val acValid = dut.io.accCtrl.valid.peekBoolean()
    if (acValid && acReady) accCtrls += peekAccCtrl()
    if (tcValid || acValid) active = true

    val busy = dut.io.status.busy.peekBoolean()
    if (busy) lastBusyRun = if (wasBusy) lastBusyRun + 1 else 1
    wasBusy = busy

    dut.clock.step()
    cycle += 1

    // ---- update models ----
    if (rxFire) { toSend.dequeue(); active = true }
    if (respFire) { respValid = false; active = true }
    // a request taken in cycle t is answered in cycle t + readLatency (>= 1)
    pendingRead = pendingRead.map { case (a, n) => (a, n - 1) }
    if (rdFire) pendingRead = Some((readReqs.last, readLatency - 1))
    pendingRead.foreach { case (a, n) =>
      if (n == 0) { respValid = true; respData = resultMem(a); pendingRead = None }
    }
    if (respValid || pendingRead.nonEmpty) active = true
    quiet = if (active) 0 else quiet + 1
  }

  def send(bytes: Seq[Int]): Unit = toSend ++= bytes

  def run(n: Int): Unit = for (_ <- 0 until n) step()

  /** Step until all host bytes are taken and nothing has moved for a while. */
  def drain(limit: Int = 20000): Unit = {
    val start = cycle
    quiet = 0 // wait for a fresh quiet period, even right after another drain
    while (toSend.nonEmpty || quiet < 30) {
      assert(cycle - start < limit, s"decoder stuck: ${toSend.length} host bytes left")
      step()
    }
  }

  /** Send STATUS, wait for the reply and return it (removed from `received`). */
  def status(): HostFrames.StatusReply = {
    val before = received.length
    send(HostFrames.status)
    drain()
    val reply = received.drop(before).toSeq
    received.remove(before, reply.length)
    HostFrames.parseStatus(reply)
  }
}
