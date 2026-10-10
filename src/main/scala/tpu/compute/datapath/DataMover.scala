package tpu.compute.datapath

import chisel3._
import chisel3.util._

import ifaces._

class DataMover(c: SAConfig) extends Module {
  val io = IO(new Bundle {
    // Sequencer <-> DataMover interface
    val cmd = Flipped(Decoupled(new TileCmd(c)))

    // UnifiedBuffer <-> DataMover interface
    val rdReq = Decoupled(new BufRowReq(c))
    val rdResp = Flipped(Decoupled(Vec(c.banks, c.inT)))

    // DataMover <-> SystolicArray interface
    val wOut = Valid(Vec(c.cols, c.inT))
    val aOut = Valid(Vec(c.rows, c.inT))
  })



  
}