package tpu.io

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import ifaces._
import tpu.memory.UnifiedBuffer
import HostFrames._

/** CommandDecoder wired to the real UnifiedBuffer, with the buffer's other
  * ports (data mover and accumulators) brought out so the test can fill and
  * check the memories.
  */
class DecoderWithBuffer(c: SAConfig, b: BoardConfig) extends Module {
  val io = IO(new Bundle {
    val rx = Flipped(Decoupled(UInt(8.W)))
    val tx = Decoupled(UInt(8.W))
    val mvRdReq = Flipped(Decoupled(new BufRowReq(c)))
    val mvRdResp = Decoupled(Vec(c.banks, c.inT))
    val resWr = Flipped(Decoupled(new BufResultWrite(c)))
  })
  val dec = Module(new CommandDecoder(c, b))
  val buf = Module(new UnifiedBuffer(c))

  dec.io.rx <> io.rx
  io.tx <> dec.io.tx
  buf.io.hostWr <> dec.io.bufWr
  buf.io.hostRdReq <> dec.io.bufRdReq
  dec.io.bufRdResp <> buf.io.hostRdResp
  buf.io.mvRdReq <> io.mvRdReq
  io.mvRdResp <> buf.io.mvRdResp
  buf.io.resWr <> io.resWr

  // no sequencer: MATMUL is not used here
  dec.io.instr.ready := false.B
  dec.io.status := 0.U.asTypeOf(new SeqStatus)
  dec.io.rxOverrun := false.B
}

class DecoderBufferSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "CommandDecoder + UnifiedBuffer"

  val c = SAConfig() // 4 banks, 4 columns
  val b = BoardConfig(clockHz = 400_000, baud = 100_000, frameTimeout = 200)

  /** Push host bytes in and collect the reply, with tx always ready. */
  def exchange(dut: DecoderWithBuffer, bytes: Seq[Int], replyLen: Int): Seq[Int] = {
    var toSend = bytes
    var reply = Vector.empty[Int]
    var cycles = 0
    var extra = 0 // cycles run after the expected reply, to catch extra bytes
    dut.io.tx.ready.poke(true.B)
    while (extra < 20) {
      assert(cycles < 10000, s"stuck: ${toSend.length} bytes left, reply so far $reply")
      dut.io.rx.valid.poke(toSend.nonEmpty.B)
      dut.io.rx.bits.poke(toSend.headOption.getOrElse(0).U)
      val fire = toSend.nonEmpty && dut.io.rx.ready.peekBoolean()
      if (dut.io.tx.valid.peekBoolean()) reply :+= dut.io.tx.bits.peekInt().toInt
      dut.clock.step()
      cycles += 1
      if (fire) toSend = toSend.tail
      if (toSend.isEmpty && reply.length >= replyLen) extra += 1
    }
    dut.io.rx.valid.poke(false.B)
    reply
  }

  def init(dut: DecoderWithBuffer): Unit = {
    dut.io.mvRdReq.valid.poke(false.B)
    dut.io.mvRdResp.ready.poke(true.B)
    dut.io.resWr.valid.poke(false.B)
  }

  /** Read one row of the input memory as the data mover does. */
  def readRow(dut: DecoderWithBuffer, row: Int): Seq[Int] = {
    dut.io.mvRdReq.valid.poke(true.B)
    dut.io.mvRdReq.bits.rowAddr.poke(row.U)
    dut.clock.step()
    dut.io.mvRdReq.valid.poke(false.B)
    dut.io.mvRdResp.valid.expect(true.B)
    dut.io.mvRdResp.bits.map(_.peekInt().toInt)
  }

  /** Write one row of the result memory as the accumulators do. */
  def writeResultRow(dut: DecoderWithBuffer, row: Int, data: Seq[BigInt]): Unit = {
    dut.io.resWr.valid.poke(true.B)
    dut.io.resWr.bits.rowAddr.poke(row.U)
    for ((d, v) <- dut.io.resWr.bits.data.zip(data)) d.poke(v.S)
    dut.clock.step()
    dut.io.resWr.valid.poke(false.B)
  }

  it should "use the buffer's element addresses for WRITE and READ" in {
    test(new DecoderWithBuffer(c, b)) { dut =>
      dut.clock.setTimeout(0)
      init(dut)

      // WRITE: elements 8..19 are rows 2..4 of the input memory
      val data = (0 until 12).map(i => (i * 37 - 100) & 0xff)
      assert(exchange(dut, write(8, data), 0).isEmpty)
      for (r <- 0 until 3)
        assert(readRow(dut, 2 + r) == data.slice(4 * r, 4 * r + 4).map(_.toByte.toInt), s"row ${2 + r}")

      // READ: elements 20..27 are rows 5 and 6 of the result memory
      val rows = Seq(
        Seq(BigInt(1), BigInt(-1), BigInt(0x7fffffff), BigInt(Int.MinValue)),
        Seq(BigInt(1000), BigInt(-1000), BigInt(127), BigInt(-127))
      )
      for ((r, i) <- rows.zipWithIndex) writeResultRow(dut, 5 + i, r)
      def le(v: BigInt) = (0 until 4).map(i => (((v & 0xffffffffL) >> (8 * i)) & 0xff).toInt)
      // start mid-row to cross a row boundary
      assert(exchange(dut, read(22, 5, 4), 20) == rows.flatten.slice(2, 7).flatMap(le))
    }
  }
}
