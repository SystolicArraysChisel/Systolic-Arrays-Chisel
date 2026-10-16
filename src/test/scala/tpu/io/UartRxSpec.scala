package tpu.io

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import UartTestUtil._

class UartRxSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "UartRx"

  /** Watches the outputs of a UartRx every cycle and checks the pulse rules. */
  class Monitor(dut: UartRx) {
    val bytes = Seq.newBuilder[Int]
    var framingErrs = 0
    private var lastValid = false

    def sample(): Unit = {
      val valid = dut.io.out.valid.peekBoolean()
      assert(!(valid && lastValid), "out.valid held for more than one cycle")
      if (valid) bytes += dut.io.out.bits.peekInt().toInt
      if (dut.io.framingErr.peekBoolean()) framingErrs += 1
      lastValid = valid
    }

    /** Idle line for `n` cycles while sampling. */
    def idle(n: Int): Unit = {
      dut.io.rxd.poke(true.B)
      for (_ <- 0 until n) { dut.clock.step(); sample() }
    }
  }

  def withRx(b: BoardConfig)(body: (UartRx, Monitor) => Unit): Unit =
    test(new UartRx(b)) { dut =>
      dut.clock.setTimeout(0)
      val m = new Monitor(dut)
      m.idle(2 * b.cyclesPerBit)
      body(dut, m)
    }

  for (b <- fastConfigs) {
    val cpb = b.cyclesPerBit

    it should s"stay silent on an idle line ($cpb cycles/bit)" in {
      withRx(b) { (_, m) =>
        m.idle(20 * cpb)
        assert(m.bytes.result().isEmpty && m.framingErrs == 0)
      }
    }

    it should s"receive every byte value ($cpb cycles/bit)" in {
      withRx(b) { (dut, m) =>
        for (byte <- 0 until 256) {
          sendFrame(dut.io.rxd, dut.clock, byte, cpb)(m.sample())
          m.idle(cpb) // one idle bit between frames
        }
        m.idle(2 * cpb) // let the last byte come out
        assert(m.bytes.result() == (0 until 256))
        assert(m.framingErrs == 0)
      }
    }

    it should s"receive back-to-back frames with no idle time ($cpb cycles/bit)" in {
      withRx(b) { (dut, m) =>
        val bytes = Seq(0x00, 0xff, 0x80, 0x01, 0x55, 0xaa, 0x00, 0x00)
        for (byte <- bytes) sendFrame(dut.io.rxd, dut.clock, byte, cpb)(m.sample())
        m.idle(2 * cpb)
        assert(m.bytes.result() == bytes)
      }
    }

    it should s"flag a zero stop bit as a framing error and drop the byte ($cpb cycles/bit)" in {
      withRx(b) { (dut, m) =>
        sendFrame(dut.io.rxd, dut.clock, 0x3c, cpb, stopBit = false)(m.sample())
        m.idle(2 * cpb) // line back high: next frame must be found again
        assert(m.framingErrs == 1)
        assert(m.bytes.result().isEmpty)
        sendFrame(dut.io.rxd, dut.clock, 0xc3, cpb)(m.sample())
        m.idle(2 * cpb)
        assert(m.bytes.result() == Seq(0xc3))
        assert(m.framingErrs == 1)
      }
    }

    it should s"ignore a low glitch shorter than half a bit ($cpb cycles/bit)" in {
      withRx(b) { (dut, m) =>
        val glitch = cpb / 2 - 1 // low for less than half a bit: gone by the mid-start sample
        dut.io.rxd.poke(false.B)
        for (_ <- 0 until glitch) { dut.clock.step(); m.sample() }
        m.idle(12 * cpb)
        assert(m.bytes.result().isEmpty && m.framingErrs == 0)
        sendFrame(dut.io.rxd, dut.clock, 0x5a, cpb)(m.sample())
        m.idle(2 * cpb)
        assert(m.bytes.result() == Seq(0x5a))
      }
    }
  }

  /** A sender whose bits are `ratio` times as long as the receiver expects. */
  def sendSkewed(dut: UartRx, m: Monitor, byte: Int, cpb: Int, ratio: Double): Unit = {
    val bits = frameBits(byte)
    val total = math.round(10 * cpb * ratio).toInt
    for (t <- 0 until total) {
      dut.io.rxd.poke(bits(math.min(9, (t / (cpb * ratio)).toInt)).B)
      dut.clock.step()
      m.sample()
    }
  }

  for (ratio <- Seq(0.985, 1.015)) {
    it should f"tolerate a ${(ratio - 1) * 100}%+.1f%% baud mismatch" in {
      val b = BoardConfig(clockHz = 1_600_000, baud = 100_000) // 16 cycles/bit
      withRx(b) { (dut, m) =>
        val bytes = Seq(0x00, 0xff, 0x55, 0xaa, 0x0f, 0xf0)
        for (byte <- bytes) {
          sendSkewed(dut, m, byte, b.cyclesPerBit, ratio)
          m.idle(b.cyclesPerBit)
        }
        m.idle(2 * b.cyclesPerBit)
        assert(m.bytes.result() == bytes)
        assert(m.framingErrs == 0)
      }
    }
  }

  it should "discard a partly received byte on reset" in {
    val b = fastConfigs.head
    val cpb = b.cyclesPerBit
    withRx(b) { (dut, m) =>
      // first 5 bits of a frame, then reset with the line idle
      for (bit <- frameBits(0x00).take(5); _ <- 0 until cpb) {
        dut.io.rxd.poke(bit.B)
        dut.clock.step()
        m.sample()
      }
      dut.io.rxd.poke(true.B)
      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.io.out.valid.expect(false.B)
      m.idle(12 * cpb)
      assert(m.bytes.result().isEmpty && m.framingErrs == 0)
      sendFrame(dut.io.rxd, dut.clock, 0x99, cpb)(m.sample())
      m.idle(2 * cpb)
      assert(m.bytes.result() == Seq(0x99))
    }
  }

  it should "receive a byte at the real board's 115200 baud" in {
    val b = BoardConfig.nexys
    withRx(b) { (dut, m) =>
      sendFrame(dut.io.rxd, dut.clock, 0xa7, b.cyclesPerBit)(m.sample())
      m.idle(2 * b.cyclesPerBit)
      assert(m.bytes.result() == Seq(0xa7))
    }
  }
}
