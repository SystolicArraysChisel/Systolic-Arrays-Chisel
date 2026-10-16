package tpu.io

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import UartTestUtil._

class UartTxSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "UartTx"

  /** Offer `byte` for one cycle (the transmitter must be idle), then record
    * txd for every cycle until the transmitter is ready again.
    */
  def sendAndRecord(dut: UartTx, byte: Int): Seq[Boolean] = {
    dut.io.in.ready.expect(true.B)
    dut.io.in.valid.poke(true.B)
    dut.io.in.bits.poke(byte.U)
    dut.clock.step()
    dut.io.in.valid.poke(false.B)
    val wave = Seq.newBuilder[Boolean]
    while (!dut.io.in.ready.peekBoolean()) {
      wave += dut.io.txd.peekBoolean()
      dut.clock.step()
    }
    wave.result()
  }

  /** The wave a correct transmitter produces: each frame bit held cyclesPerBit cycles. */
  def expectedWave(byte: Int, cpb: Int): Seq[Boolean] =
    frameBits(byte).flatMap(Seq.fill(cpb)(_))

  for (b <- fastConfigs) {
    val cpb = b.cyclesPerBit

    it should s"keep the line idle and be ready after reset ($cpb cycles/bit)" in {
      test(new UartTx(b)) { dut =>
        for (_ <- 0 until 3 * cpb) {
          dut.io.txd.expect(true.B)
          dut.io.in.ready.expect(true.B)
          dut.clock.step()
        }
      }
    }

    it should s"send a frame with exact bit timing ($cpb cycles/bit)" in {
      test(new UartTx(b)) { dut =>
        for (byte <- Seq(0x00, 0xff, 0x55, 0xaa, 0x01, 0x80, 0xa5)) {
          assert(sendAndRecord(dut, byte) == expectedWave(byte, cpb), f"byte 0x$byte%02x")
          dut.io.txd.expect(true.B) // back to idle
        }
      }
    }

    // ready rises the cycle after the stop bit ends, so back-to-back frames are
    // separated by one extra idle cycle (a stop bit of cyclesPerBit + 1 cycles).
    it should s"send back-to-back bytes with a one-cycle gap ($cpb cycles/bit)" in {
      test(new UartTx(b)) { dut =>
        val bytes = Seq(0x12, 0x34, 0xfe)
        dut.io.in.valid.poke(true.B) // a sender with a new byte always waiting
        val wave = Seq.newBuilder[Boolean]
        var queue = bytes
        var cycles = 0
        while (queue.nonEmpty || !dut.io.in.ready.peekBoolean()) {
          if (queue.nonEmpty) dut.io.in.bits.poke(queue.head.U)
          val fire = dut.io.in.ready.peekBoolean() && queue.nonEmpty
          if (queue.isEmpty) dut.io.in.valid.poke(false.B)
          if (cycles > 0) wave += dut.io.txd.peekBoolean() // skip the cycle before the first start bit
          dut.clock.step()
          cycles += 1
          if (fire) queue = queue.tail
        }
        val expected = bytes.map(expectedWave(_, cpb)).reduce(_ ++ Seq(true) ++ _)
        assert(wave.result() == expected)
      }
    }

    it should s"ignore bits while busy ($cpb cycles/bit)" in {
      test(new UartTx(b)) { dut =>
        dut.io.in.valid.poke(true.B)
        dut.io.in.bits.poke(0x0f.U)
        dut.clock.step()
        dut.io.in.bits.poke(0xf0.U) // changing bits while busy must not affect the frame
        val wave = Seq.newBuilder[Boolean]
        for (_ <- 0 until 10 * cpb) {
          dut.io.in.ready.expect(false.B)
          wave += dut.io.txd.peekBoolean()
          dut.clock.step()
        }
        assert(wave.result() == expectedWave(0x0f, cpb))
        dut.io.in.ready.expect(true.B)
      }
    }
  }

  it should "return to idle on reset in the middle of a byte" in {
    val b = fastConfigs.head
    test(new UartTx(b)) { dut =>
      dut.io.in.valid.poke(true.B)
      dut.io.in.bits.poke(0x00.U)
      dut.clock.step()
      dut.io.in.valid.poke(false.B)
      dut.clock.step(3 * b.cyclesPerBit)
      dut.io.txd.expect(false.B) // inside the data bits of 0x00
      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.io.txd.expect(true.B)
      dut.io.in.ready.expect(true.B)
    }
  }

  it should "take 10 * cyclesPerBit cycles per byte on the real board" in {
    val b = BoardConfig.nexys
    test(new UartTx(b)) { dut =>
      dut.clock.setTimeout(0)
      assert(sendAndRecord(dut, 0x5a).length == b.byteCycles)
    }
  }
}
