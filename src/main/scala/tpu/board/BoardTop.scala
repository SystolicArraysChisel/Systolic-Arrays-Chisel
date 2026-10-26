package tpu.board

import chisel3._
import chisel3.util._

/** Maps a core to the pins of the Nexys4 DDR
  *
  * A RawModule: it has no implicit clock or reset. The core runs on the board
  * clock, or, when coreHz is not 100 MHz, on a clock made from it by an MMCM
  * (ClockGen). Its reset comes from the CPU_RESETN button: active-low and
  * asynchronous on the pin, it is inverted and passed through two flip-flops,
  * so the core sees a reset that starts as soon as the button is pressed and
  * ends on a clock edge, two cycles after it is let go. With an MMCM, reset is
  * also held until the MMCM has locked.
  */
class BoardTop(core: => CoreModule, coreHz: Int = 100_000_000) extends RawModule {
  val CLK100MHZ = IO(Input(Clock())) // 100 MHz board clock
  val CPU_RESETN = IO(Input(Bool())) // reset button, active-low
  val UART_TXD_IN = IO(Input(Bool())) // from the PC (named from the PC's side)
  val UART_RXD_OUT = IO(Output(Bool())) // to the PC
  val LED = IO(Output(UInt(16.W))) // LED(0) busy, LED(1) error, LED(2) overrun

  val (coreClock, locked) =
    if (coreHz == 100_000_000) (CLK100MHZ, true.B)
    else {
      val gen = Module(new ClockGen(coreHz))
      gen.clkIn := CLK100MHZ
      (gen.clkOut, gen.locked)
    }

  val coreReset = withClockAndReset(coreClock, (!CPU_RESETN || !locked).asAsyncReset) {
    RegNext(RegNext(false.B, true.B), true.B)
  }

  withClockAndReset(coreClock, coreReset) {
    val c = Module(core)
    c.io.rxd := UART_TXD_IN
    UART_RXD_OUT := c.io.txd
    LED := Cat(0.U(13.W), c.io.debug.overrun, c.io.debug.error, c.io.debug.busy)
  }
}
