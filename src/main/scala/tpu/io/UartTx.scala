package tpu.io

import chisel3._
import chisel3.util._

object UartTxState extends ChiselEnum {
  val sIdle, sBusy = Value
}

class UartTx(b: BoardConfig) extends Module {

  val io = IO(new Bundle {
    val in = Flipped(Decoupled(UInt(8.W)))
    val txd = Output(Bool())
  })

  val state = RegInit(UartTxState.sIdle)
  val shiftReg = RegInit(
    ((1 << 10) - 1).U(10.W)
  ) // initialize frame register at 1_11111111_1
  val cycleCnt = RegInit(0.U(log2Ceil(b.cyclesPerBit).W))
  val bitCnt = RegInit(0.U(4.W))

  val bitEnd =
    cycleCnt === (b.cyclesPerBit - 1).U // have we held the current bit accessible long enough?

  io.in.ready := state === UartTxState.sIdle
  io.txd := shiftReg(
    0
  ) // avoid glitching by using a clock synchronized register ensuring that the host does interpret noise as meaningful values

  switch(state) {
    is(UartTxState.sIdle) {
      when(io.in.fire) {
        shiftReg := Cat(
          1.U(1.W),
          io.in.bits,
          0.U(1.W)
        )
        cycleCnt := 0.U
        bitCnt := 0.U
        state := UartTxState.sBusy
      }
    }
    is(UartTxState.sBusy) {
      cycleCnt := cycleCnt + 1.U
      when(bitEnd) {
        cycleCnt := 0.U
        shiftReg := Cat(
          1.U(1.W),
          shiftReg(9, 1)
        )
        bitCnt := bitCnt + 1.U
        when(bitCnt === 9.U) {
          state := UartTxState.sIdle
        }
      }
    }
  }
}
