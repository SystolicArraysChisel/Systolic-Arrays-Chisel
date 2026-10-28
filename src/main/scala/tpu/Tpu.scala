package tpu

import chisel3._
import chisel3.util._

import ifaces._
import tpu.board.{CoreIO, CoreModule}
import tpu.compute.array.SystolicArray
import tpu.compute.control.Sequencer
import tpu.compute.datapath.{Accumulators, DataMover}
import tpu.io.{BoardConfig, CommandDecoder, RxFifo, UartRx, UartTx}
import tpu.memory.UnifiedBuffer

/** everything but the UART. Kept apart from Tpu so that tests can talk to it one byte per cycle, synthesizes the same hardware as one merged module would.
  */
class TpuCore(c: SAConfig, b: BoardConfig) extends Module {
  val io = IO(new Bundle {
    val rx = Flipped(Decoupled(UInt(8.W))) // bytes from RxFifo
    val tx = Decoupled(UInt(8.W)) // bytes to UartTx
    val rxOverrun = Input(Bool()) // pulse from RxFifo: a byte was lost
    val busy = Output(Bool()) // a MATMUL is running
    val error = Output(Bool()) // an invalid or timed-out frame since reset
  })

  val dec = Module(new CommandDecoder(c, b))
  val buf = Module(new UnifiedBuffer(c))
  val seq = Module(new Sequencer(c))
  val mover = Module(new DataMover(c))
  val array = Module(new SystolicArray(c))
  val acc = Module(new Accumulators(c))

  // host side
  dec.io.rx <> io.rx
  io.tx <> dec.io.tx
  dec.io.rxOverrun := io.rxOverrun
  io.busy := seq.io.status.busy
  io.error := dec.io.error

  // decoder <-> buffer and sequencer
  buf.io.hostWr <> dec.io.bufWr
  buf.io.hostRdReq <> dec.io.bufRdReq
  dec.io.bufRdResp <> buf.io.hostRdResp
  seq.io.instr <> dec.io.instr
  dec.io.status := seq.io.status

  // sequencer -> data mover -> array -> accumulators -> buffer
  mover.io.cmd <> seq.io.tileCmd
  buf.io.mvRdReq <> mover.io.rdReq
  mover.io.rdResp <> buf.io.mvRdResp
  array.io.wIn := mover.io.wOut
  array.io.aIn := mover.io.aOut
  acc.io.in := array.io.out
  acc.io.ctrl <> seq.io.accCtrl
  seq.io.tileDone := acc.io.done
  buf.io.resWr <> acc.io.resWr
}

/** The whole accelerator behind the serial line, for BoardTop:
  * rxd -> UartRx -> RxFifo -> TpuCore -> UartTx -> txd.
  *
  * LEDs: busy while a MATMUL runs; error (until reset) after an invalid or
  * timed-out frame, or a UART framing error; overrun (until reset) if RxFifo
  * lost a byte.
  */
class Tpu(c: SAConfig, b: BoardConfig) extends CoreModule {
  val io = IO(new CoreIO)

  val rx = Module(new UartRx(b))
  val fifo = Module(new RxFifo(b))
  val tx = Module(new UartTx(b))
  val core = Module(new TpuCore(c, b))

  rx.io.rxd := io.rxd
  fifo.io.enq := rx.io.out
  core.io.rx <> fifo.io.deq
  core.io.rxOverrun := fifo.io.overrun
  tx.io.in <> core.io.tx
  io.txd := tx.io.txd

  val framingErr = RegInit(false.B)
  val overrun = RegInit(false.B)
  framingErr := framingErr || rx.io.framingErr
  overrun := overrun || fifo.io.overrun

  io.debug.busy := core.io.busy
  io.debug.error := core.io.error || framingErr
  io.debug.overrun := overrun
}
