package tpu.compute.datapath

import chisel3._
import chisel3.util._

import ifaces._

class Accumulators(c: SAConfig) extends Module {
  val io = IO(new Bundle {
    // SystolicArray <-> Accumulators interface
    val in = Flipped(Valid(Vec(c.cols, c.accT)))

    // Sequencer <-> Accumulators interface
    val ctrl = Flipped(Decoupled(new AccCtrl(c)))
    val done = Valid(Bool())

    // Accumulator <-> UnifiedBuffer interface
    val resWr = Decoupled(new BufResultWrite(c))
  })
}