package tpu.io

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import UartTestUtil._

/** UartTx wired straight into UartRx, for testing the two together. */
class UartLoopback(b: BoardConfig) extends Module {
  val io = IO(new Bundle {
    val in = Flipped(chisel3.util.Decoupled(UInt(8.W)))
    val out = Output(chisel3.util.Valid(UInt(8.W)))
    val framingErr = Output(Bool())
  })
  val tx = Module(new UartTx(b))
  val rx = Module(new UartRx(b))
  tx.io.in <> io.in
  rx.io.rxd := tx.io.txd
  io.out := rx.io.out
  io.framingErr := rx.io.framingErr
}

class UartLoopbackSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "UartTx -> UartRx"

  for (b <- fastConfigs) {
    it should s"carry every byte value back-to-back (${b.cyclesPerBit} cycles/bit)" in {
      test(new UartLoopback(b)) { dut =>
        dut.clock.setTimeout(0)
        var toSend = (0 until 256).toList
        var received = Vector.empty[Int]
        val limit = 300 * b.byteCycles // 256 bytes plus some margin
        var cycles = 0
        while (received.length < 256 && cycles < limit) {
          dut.io.in.valid.poke(toSend.nonEmpty.B)
          if (toSend.nonEmpty) dut.io.in.bits.poke(toSend.head.U)
          val fire = toSend.nonEmpty && dut.io.in.ready.peekBoolean()
          if (dut.io.out.valid.peekBoolean()) received :+= dut.io.out.bits.peekInt().toInt
          dut.io.framingErr.expect(false.B)
          dut.clock.step()
          cycles += 1
          if (fire) toSend = toSend.tail
        }
        assert(received == (0 until 256))
      }
    }
  }
}
