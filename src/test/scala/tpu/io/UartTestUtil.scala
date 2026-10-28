package tpu.io

import chisel3._
import chiseltest._

/** Helpers shared by the UART tests. */
object UartTestUtil {

  /** Small board configs so tests run fast: 4 (the minimum), 7 (odd) and 16
    * cycles per bit.
    */
  val fastConfigs: Seq[BoardConfig] = Seq(
    BoardConfig(clockHz = 400_000, baud = 100_000),
    BoardConfig(clockHz = 700_000, baud = 100_000),
    BoardConfig(clockHz = 1_600_000, baud = 100_000)
  )

  /** Bits of one UART frame in wire order: start (0), 8 data bits LSB first,
    * stop (1).
    */
  def frameBits(byte: Int, stopBit: Boolean = true): Seq[Boolean] =
    Seq(false) ++ (0 until 8).map(i => ((byte >> i) & 1) == 1) ++ Seq(stopBit)

  /** Drive `byte` on `line`, each bit held for `cyclesPerBit` cycles. On every
    * cycle `onCycle` is called after the clock step (for watching outputs).
    */
  def sendFrame(
      line: Bool,
      clock: Clock,
      byte: Int,
      cyclesPerBit: Int,
      stopBit: Boolean = true
  )(onCycle: => Unit = ()): Unit =
    for (bit <- frameBits(byte, stopBit); _ <- 0 until cyclesPerBit) {
      line.poke(bit.B)
      clock.step()
      onCycle
    }
}
