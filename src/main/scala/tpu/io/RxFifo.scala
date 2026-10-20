package tpu.io

import chisel3._
import chisel3.util._

class RxFifo(b: BoardConfig) extends Module {
  val io = IO(new Bundle {
    val enq = Flipped(Valid(UInt(8.W))) // bytes from UartRx.out
    val deq = Decoupled(UInt(8.W)) // bytes for the decoder, oldest first
    val overrun = Output(
      Bool()
    ) // pulse: a byte arrived while the FIFO was full, and was lost
    val count = Output(
      UInt(log2Ceil(b.rxFifoDepth + 1).W)
    ) // Number of stored bytes (for debugging).
  })
  // pipe = true means that when full, a byte may enter in the same cycle as one leaves.
  val queue = Module(new Queue(UInt(8.W), b.rxFifoDepth, pipe = true))

  queue.io.enq.valid := io.enq.valid
  queue.io.enq.bits := io.enq.bits
  io.deq <> queue.io.deq

  io.overrun := io.enq.valid && !queue.io.enq.ready // a byte offered, but no room for it
  io.count := queue.io.count
}
