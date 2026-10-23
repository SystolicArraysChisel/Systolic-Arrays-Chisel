package tpu.io

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

import ifaces.SAConfig
import HostFrames._

class CommandDecoderSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "CommandDecoder"

  val c = SAConfig() // addrW = outAddrW = 12
  val b = BoardConfig(clockHz = 400_000, baud = 100_000, frameTimeout = 200)

  def withBench(randomReady: Boolean = false)(body: (DecoderWithSequencer, DecoderBench) => Unit): Unit =
    test(new DecoderWithSequencer(c, b)) { dut =>
      dut.clock.setTimeout(0)
      body(dut, new DecoderBench(dut, seed = 1, randomReady))
    }

  /** Signed value of a byte, as stored in the input memory. */
  def s8(v: Int): Int = (v.toByte).toInt

  /** Little-endian bytes of the low `size` bytes of a result. */
  def leBytes(v: BigInt, size: Int): Seq[Int] = {
    val u = v & ((BigInt(1) << 32) - 1)
    (0 until size).map(i => ((u >> (8 * i)) & 0xff).toInt)
  }

  // ---- one test per command ----

  it should "write the elements of a WRITE in order" in {
    withBench() { (_, t) =>
      val data = Seq(0x00, 0x01, 0x7f, 0x80, 0xff, 0x42)
      t.send(write(0x10, data))
      t.drain()
      assert(t.writes == data.zipWithIndex.map { case (d, i) => (0x10 + i, s8(d)) })
      assert(t.received.isEmpty)
    }
  }

  it should "answer a READ with little-endian elements of 4 or 1 bytes" in {
    withBench() { (_, t) =>
      val values = Seq(BigInt(0x12345678), BigInt(-2), BigInt(Int.MinValue), BigInt(127))
      for ((v, i) <- values.zipWithIndex) t.resultMem(0x20 + i) = v
      t.send(read(0x20, values.length, 4) ++ read(0x20, values.length, 1))
      t.drain()
      assert(t.readReqs == (values.indices ++ values.indices).map(0x20 + _))
      assert(t.received == values.flatMap(leBytes(_, 4)) ++ values.flatMap(leBytes(_, 1)))
    }
  }

  it should "start a MATMUL on the sequencer with every field" in {
    withBench() { (_, t) =>
      // 2 N tiles of 2 K tiles each; every field shows up in the tile commands
      val mm = Matmul(aAddr = 0x100, bAddr = 0x204, cAddr = 0x308, m = 33, k = 8, n = 8,
        ldA = 0x1234, ldB = 0x0abc, ldC = 0xfff0, flags = 0x03, shift = 31)
      t.send(mm.bytes)
      t.drain()
      assert(t.tiles == Tiles.expected(mm, c))
      assert(t.tilesDone == 4)
    }
  }

  it should "reply to STATUS with the sequencer's busy flag and counters" in {
    withBench() { (_, t) =>
      assert(t.status() == StatusReply(busy = false, error = false, overrun = false, 0, 0))

      t.holdTiles = true
      t.send(Matmul(m = 5, k = 8, n = 4).bytes)
      t.run(50)
      val during = t.status()
      assert(during.busy && !during.error && during.busyPe == 0)
      assert(during.cycles > 0 && during.cycles < t.lastBusyRun, "the count so far, at capture time")

      t.holdTiles = false
      t.drain()
      assert(t.status() == StatusReply(busy = false, error = false, overrun = false,
        cycles = t.lastBusyRun, busyPe = 5 * 8 * 4))
    }
  }

  // ---- edge cases with their own logic in the decoder ----

  it should "accept a WRITE or READ of zero elements without doing anything" in {
    withBench() { (_, t) =>
      t.send(write(5, Seq()) ++ read(5, 0, 4))
      t.drain()
      assert(t.writes.isEmpty && t.readReqs.isEmpty && t.received.isEmpty)
      assert(!t.status().error)
    }
  }

  it should "drop an invalid frame, report an error, and keep working" in {
    val invalid = Seq(
      "unknown opcode" -> Seq(0x99),
      "READ size 2" -> read(0, 3, 2),
      "MATMUL shift 32" -> Matmul(shift = 32).bytes,
      "MATMUL flag bit 2" -> Matmul(flags = 0x04).bytes
    )
    for ((name, frame) <- invalid) withBench() { (dut, t) =>
      dut.io.error.expect(false.B)
      t.send(frame ++ write(1, Seq(7))) // the next valid frame still goes through
      t.drain()
      assert(t.writes == Seq((1, 7)), name)
      assert(t.readReqs.isEmpty && t.tileCmds.isEmpty && t.received.isEmpty, name)
      assert(t.status().error, name)
      assert(!t.status().error, s"$name: cleared by the previous STATUS")
      dut.io.error.expect(true.B, s"$name: the LED output stays on until reset")
      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.io.error.expect(false.B)
    }
  }

  it should "keep overrun sticky until STATUS and let a new event win over the clear" in {
    withBench() { (_, t) =>
      t.overrunNext = true
      t.run(3)
      assert(t.status().overrun)
      assert(!t.status().overrun)
      // overrun pulse in the very cycle the STATUS opcode is accepted
      t.send(status)
      t.overrunNext = true
      t.drain()
      assert(!parseStatus(t.received.toSeq).overrun, "the event is after the snapshot")
      t.received.clear()
      assert(t.status().overrun, "but it is not lost")
    }
  }

  it should "drop a frame after more than frameTimeout idle cycles, not after exactly frameTimeout" in {
    withBench() { (_, t) =>
      for (byte <- write(0x40, Seq(1, 2))) {
        t.send(Seq(byte))
        t.step() // the byte is taken in this cycle
        t.run(b.frameTimeout)
      }
      t.drain()
      assert(t.writes == Seq((0x40, 1), (0x41, 2)))
      assert(!t.status().error)

      t.send(write(0x60, Seq(1, 2, 3)).take(6)) // opcode, header, first data byte
      while (t.toSend.nonEmpty) t.step()
      t.run(b.frameTimeout + 1)
      t.send(write(0x50, Seq(9))) // read as a new frame, not as the rest of the old one
      t.drain()
      assert(t.writes.drop(2) == Seq((0x60, 1), (0x50, 9)))
      assert(t.status().error)
    }
  }

  // ---- flow control ----

  it should "read no bytes while a MATMUL waits for the busy sequencer" in {
    withBench() { (_, t) =>
      t.holdTiles = true
      t.send(Matmul(m = 1).bytes ++ Matmul(m = 2).bytes ++ status)
      t.run(500)
      assert(t.tileCmds.map(_.m) == Seq(1)) // the second MATMUL is not accepted
      assert(t.toSend == Seq(HostProtocol.Op.Status)) // STATUS still waiting in the FIFO
      t.holdTiles = false
      t.drain()
      assert(t.tileCmds.map(_.m) == Seq(1, 2))
      assert(!parseStatus(t.received.toSeq).error, "the wait is not a timeout")
    }
  }

  it should "run every command back to back under random backpressure" in {
    withBench(randomReady = true) { (_, t) =>
      t.holdTiles = true // keeps the sequencer busy, so the first STATUS is known
      t.resultMem(0) = BigInt(-2); t.resultMem(1) = BigInt(0x12345678)
      val mm = Matmul(m = 5)
      t.send(write(0x10, Seq(1, 0xff)) ++ read(0, 2, 4) ++ mm.bytes ++ status ++ read(1, 1, 1))
      t.drain()
      assert(t.writes == Seq((0x10, 1), (0x11, -1)))
      val reads = leBytes(BigInt(-2), 4) ++ leBytes(BigInt(0x12345678), 4)
      assert(t.received.take(8) == reads && t.received.last == 0x78 && t.received.length == 8 + 9 + 1)
      val during = parseStatus(t.received.slice(8, 17).toSeq)
      assert(during.busy && !during.error && during.busyPe == 0 && during.cycles <= t.lastBusyRun)

      t.received.clear()
      t.holdTiles = false
      t.drain()
      assert(t.tiles == Tiles.expected(mm, c))
      assert(t.status() == StatusReply(busy = false, error = false, overrun = false,
        cycles = t.lastBusyRun, busyPe = 5 * 4 * 4))
    }
  }
}
