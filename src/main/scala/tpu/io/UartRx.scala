package tpu.io
import chisel3._
import chisel3.util._

object UartRxState extends ChiselEnum {
  val sIdle, sStart, sData, sStop = Value
}

class UartRx(b: BoardConfig) extends Module {
  val io = IO(new Bundle {
    val rxd = Input(Bool())
    val out = Valid(UInt(8.W))
    val framingErr = Output(Bool())
  })

  /** The line, made safe to use: delayed by two cycles */
  val rxBit = RegNext(RegNext(io.rxd, true.B), true.B)

  /** rxBit one cycle earlier, to detect the falling edge of a start bit. */
  val prev = RegNext(rxBit, true.B)
  val falling = prev && !rxBit

  val state = RegInit(UartRxState.sIdle)

  /** Cycles since the last sample (or since the falling edge). */
  val cycleCnt = RegInit(0.U(log2Ceil(b.cyclesPerBit).W))

  /** Which data bit is being read: 0 to 7. */
  val bitCnt = RegInit(0.U(3.W))

  /** Byte being rebuilt; bits enter at the top and move down, so the first bit
    * ends in bit 0.
    */
  val data = RegInit(0.U(8.W))

  /** One-cycle output pulses. */
  val outValid = RegInit(false.B)
  val errPulse = RegInit(false.B)

  // Pulses are 0 by default; the stop state sets one of them for a single cycle.
  outValid := false.B
  errPulse := false.B

  io.out.valid := outValid
  io.out.bits := data
  io.framingErr := errPulse

  switch(state) {
    is(UartRxState.sIdle) {
      when(falling) {
        cycleCnt := 0.U
        state := UartRxState.sStart
      }
    }
    is(UartRxState.sStart) {
      cycleCnt := cycleCnt + 1.U
      when(cycleCnt === (b.cyclesPerBit / 2 - 1).U) { // middle of the start bit
        cycleCnt := 0.U
        when(rxBit) {
          state := UartRxState.sIdle // the line is back at 1: a glitch, not a start bit
        }.otherwise {
          bitCnt := 0.U
          state := UartRxState.sData
        }
      }
    }
    is(UartRxState.sData) {
      cycleCnt := cycleCnt + 1.U
      when(cycleCnt === (b.cyclesPerBit - 1).U) { // middle of a data bit
        cycleCnt := 0.U
        data := Cat(rxBit, data(7, 1))
        bitCnt := bitCnt + 1.U
        when(bitCnt === 7.U) {
          state := UartRxState.sStop
        }
      }
    }
    is(UartRxState.sStop) {
      cycleCnt := cycleCnt + 1.U
      when(cycleCnt === (b.cyclesPerBit - 1).U) { // middle of the stop bit
        cycleCnt := 0.U
        when(rxBit) {
          outValid := true.B // correct frame: output the byte next cycle
        }.otherwise {
          errPulse := true.B // stop bit is 0: framing error
        }
        state := UartRxState.sIdle // ready for the next start bit right away
      }
    }
  }
}
