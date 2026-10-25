package tpu.board

import chisel3._
import chisel3.stage.ChiselStage
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

/** Smallest core BoardTop can wrap: txd is rxd delayed by one register (idle
  * high, so it stays high while in reset), and the debug flags are constants.
  */
class StubCore(busy: Boolean = false, error: Boolean = false, overrun: Boolean = false)
    extends CoreModule {
  val io = IO(new CoreIO)
  io.txd := RegNext(io.rxd, true.B)
  io.debug.busy := busy.B
  io.debug.error := error.B
  io.debug.overrun := overrun.B
}

/** BoardTop driven by the test clock, with its pins as ports. */
class BoardHarness(core: => CoreModule) extends Module {
  val io = IO(new Bundle {
    val resetn = Input(Bool())
    val txdIn = Input(Bool())
    val rxdOut = Output(Bool())
    val led = Output(UInt(16.W))
  })
  val top = Module(new BoardTop(core))
  top.CLK100MHZ := clock
  top.CPU_RESETN := io.resetn
  top.UART_TXD_IN := io.txdIn
  io.rxdOut := top.UART_RXD_OUT
  io.led := top.LED
}

class BoardTopSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "BoardTop"

  def pressReset(dut: BoardHarness): Unit = {
    dut.io.txdIn.poke(true.B)
    dut.io.resetn.poke(false.B)
    dut.clock.step(3)
    dut.io.resetn.poke(true.B)
    dut.clock.step(3) // released two edges after the button
  }

  it should "connect the UART pins through the core" in {
    test(new BoardHarness(new StubCore)) { dut =>
      pressReset(dut)
      val rnd = new Random(0)
      val line = Seq.fill(64)(rnd.nextBoolean())
      val seen = line.map { bit =>
        dut.io.txdIn.poke(bit.B)
        dut.clock.step()
        dut.io.rxdOut.peekBoolean()
      }
      assert(seen == line, "UART_RXD_OUT must be UART_TXD_IN one cycle later")
    }
  }

  it should "hold the core in reset while the button is pressed, and release it two edges after" in {
    test(new BoardHarness(new StubCore)) { dut =>
      pressReset(dut)
      dut.io.txdIn.poke(false.B)
      dut.io.resetn.poke(false.B)
      for (_ <- 0 until 20) {
        dut.clock.step()
        dut.io.rxdOut.expect(true.B) // in reset: the register keeps the idle level
      }
      dut.io.resetn.poke(true.B)
      val highFor = (0 until 6).map { _ =>
        dut.clock.step()
        dut.io.rxdOut.peekBoolean()
      }.takeWhile(identity).length
      assert(highFor == 2, s"core left reset after $highFor edges, expected 2")
    }
  }

  it should "show busy, error and overrun on LED 0, 1 and 2, with the other LEDs off" in {
    for (flags <- 0 until 8) {
      val (busy, error, overrun) = ((flags & 1) != 0, (flags & 2) != 0, (flags & 4) != 0)
      test(new BoardHarness(new StubCore(busy, error, overrun))) { dut =>
        pressReset(dut)
        dut.io.led.expect(flags.U)
      }
    }
  }

  it should "emit Verilog with the pin names of the .xdc file" in {
    @annotation.nowarn("cat=deprecation")
    val v = (new ChiselStage).emitVerilog(new BoardTop(new StubCore),
      Array("--target-dir", "test_run_dir/BoardTopVerilog"))
    for (pin <- Seq("CLK100MHZ", "CPU_RESETN", "UART_TXD_IN", "UART_RXD_OUT"))
      assert(s"(input|output)\\s+$pin\\b".r.findFirstIn(v).isDefined, s"port $pin")
    assert("""output\s+\[15:0\]\s+LED\b""".r.findFirstIn(v).isDefined, "port LED[15:0]")
  }
}
