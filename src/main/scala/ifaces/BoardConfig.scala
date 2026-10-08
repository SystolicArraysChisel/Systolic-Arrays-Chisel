package ifaces

import chisel3._
import chisel3.util._

case class BoardConfig(
    clockHz: Int = 100_000_000, // Clock frequency in Hz
    baud: Int =
      115_200, // UART speed in bits per second. Must match the Python host.
    rxFifoDepth: Int =
      16, // Bytes RxFifo can hold while the decoder is busy; when full, the next byte is lost.
    frameTimeout: Int =
      1 << 20 // Cycles allowed between two bytes of a frame before the decoder drops it (about 10 ms).
) {
  // ---- derived values ----
  val cyclesPerBit: Int = math
    .round(clockHz.toDouble / baud)
    .toInt // Clock cycles per UART bit, rounded to the nearest integer.

  val byteCycles: Int =
    10 * cyclesPerBit // Clock cycles for one byte on the wire: start bit, 8 data bits, stop bit.

  val baudError: Double =
    math.abs(
      clockHz.toDouble / cyclesPerBit - baud
    ) / baud // Relative difference between the real bit rate and the requested one.

  val timeoutW: Int = log2Ceil(
    frameTimeout + 1
  ) // Bits of the decoder's timeout counter (counts 0 to frameTimeout).

  // ---- checks ----
  // base parameters checks :
  require(
    clockHz > 0 && baud > 0,
    s"clockHz and baud must be positive, got $clockHz and $baud"
  )
  require(rxFifoDepth >= 2, s"rxFifoDepth must be at least 2, got $rxFifoDepth")
  require(frameTimeout > 0, s"frameTimeout must be positive, got $frameTimeout")
  // derived parameters checks :
  require(
    cyclesPerBit >= 4,
    s"$cyclesPerBit cycles per bit: the receiver needs at least 4 to sample in the middle of a bit"
  )
  require(
    baudError < 0.02,
    f"baud error of ${baudError * 100}%.2f%%: a UART tolerates at most about 2%%"
  )
  require(
    frameTimeout >= 2 * byteCycles,
    s"frameTimeout ($frameTimeout) must be at least two byte times (${2 * byteCycles} cycles)"
  )
}

object BoardConfig {

  val nexys: BoardConfig =
    BoardConfig() // The real board: 100 MHz, 115 200 baud.
}
