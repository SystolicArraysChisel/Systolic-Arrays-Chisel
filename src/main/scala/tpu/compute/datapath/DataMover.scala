package tpu.compute.datapath

import chisel3._
import chisel3.util._

import ifaces._

/**
  * Streams weights and activations from the unified buffer to the systolic
  * array, overlapping consecutive tiles.
  *
  * Two independent streams, one per buffer read port:
  *  - W stream (port B): takes the next TileCmd and reads its rows weight
  *    rows, last row first, into the PEs' shadow registers. Then it sets
  *    shadowFull and hands the command to the A stream.
  *  - A stream (port A): when shadowFull is set and its previous tile is
  *    done, reads the m activation rows. The first one carries aFirst, which
  *    swaps the shadow weights in, and frees the W stream for the next tile.
  *
  * The PEs require every weight beat of tile k+1 to fall in
  * [t0 + arrayLatency, t1 - 1] (SystolicArrayIO.aFirst), where t0 and t1 are
  * the first activation rows of tiles k and k+1. Both ports have the same
  * read latency, so the rule holds at the array if it holds for requests:
  * sinceSwap delays the W stream, and shadowFull delays the A stream.
  */
class DataMover(c: SAConfig) extends Module {
  val io = IO(new Bundle {
    // Sequencer <-> DataMover interface
    val cmd = Flipped(Decoupled(new TileCmd(c)))
    val accFree = Input(Bool()) // pulse: the previous output tile is written back

    // UnifiedBuffer <-> DataMover interface: activations (port A), weights (port B)
    val rdReq = Decoupled(new BufRowReq(c))
    val rdResp = Flipped(Decoupled(Vec(c.banks, c.inT)))
    val wRdReq = Decoupled(new BufRowReq(c))
    val wRdResp = Flipped(Decoupled(Vec(c.banks, c.inT)))

    // DataMover <-> SystolicArray interface
    val wOut = Valid(Vec(c.cols, c.inT))
    val aOut = Valid(Vec(c.rows, c.inT))
    // Set with the first activation row of the tile: makes its weights active.
    val aFirst = Output(Bool())
  })

  // Cycles since the last swap request, saturating at arrayLatency; starts
  // saturated, so the first tile does not wait.
  val sinceSwap = RegInit(c.arrayLatency.U(log2Ceil(c.arrayLatency + 1).W))
  when(sinceSwap =/= c.arrayLatency.U) { sinceSwap := sinceSwap + 1.U }

  // Shadow registers hold a loaded tile that has not been swapped in yet.
  val shadowFull = RegInit(false.B)
  // Command of the tile in W, then waiting in the shadows for the A stream.
  val held = Reg(new TileCmd(c))

  // ---- W stream ----
  val wBusy = RegInit(false.B)
  val wAddr = Reg(UInt(c.addrW.W))
  val wCount = Reg(UInt(log2Ceil(c.rows + 1).W))

  // The first weight request goes out the cycle after cmd.fire, which must be
  // at least arrayLatency cycles after the swap request.
  io.cmd.ready := !wBusy && !shadowFull && sinceSwap >= (c.arrayLatency - 1).U

  when(io.cmd.fire) {
    held := io.cmd.bits
    wBusy := true.B
    wCount := 0.U
    // Weights are streamed last row first.
    wAddr := io.cmd.bits.bAddr + (c.rows - 1).U * io.cmd.bits.ldB
  }

  io.wRdReq.valid := wBusy
  io.wRdReq.bits.rowAddr := wAddr >> c.bankBits

  when(io.wRdReq.fire) {
    wAddr := wAddr - held.ldB
    wCount := wCount + 1.U
    when(wCount === (c.rows - 1).U) {
      wBusy := false.B
      shadowFull := true.B
    }
  }

  // ---- A stream ----
  val aBusy = RegInit(false.B)
  val aAddr = Reg(UInt(c.addrW.W))
  val ldA = Reg(UInt(c.dimW.W))
  val aLeft = Reg(UInt(c.dimW.W)) // rows still to request after the current one

  // accFree may come before the A stream reaches the tile that waits for it.
  val accFreeSeen = RegInit(false.B)
  when(io.accFree) { accFreeSeen := true.B }

  val canStart = shadowFull && !aBusy && (!held.waitAcc || accFreeSeen)
  val starting = canStart && held.m =/= 0.U

  io.rdReq.valid := aBusy || starting
  io.rdReq.bits.rowAddr := Mux(aBusy, aAddr, held.aAddr) >> c.bankBits

  // A tile with m = 0 has nothing to multiply: drop its weights unswapped.
  when(canStart && held.m === 0.U) {
    shadowFull := false.B
    when(held.waitAcc) { accFreeSeen := false.B }
  }

  val firstFire = starting && io.rdReq.fire
  when(firstFire) {
    shadowFull := false.B
    sinceSwap := 1.U
    when(held.waitAcc) { accFreeSeen := false.B }
    aAddr := held.aAddr + held.ldA
    ldA := held.ldA
    aLeft := held.m - 1.U
    aBusy := held.m =/= 1.U
  }
  when(aBusy && io.rdReq.fire) {
    aAddr := aAddr + ldA
    aLeft := aLeft - 1.U
    when(aLeft === 1.U) { aBusy := false.B }
  }

  // ---- outputs: responses arrive one cycle after their requests ----
  io.rdResp.ready := true.B
  io.wRdResp.ready := true.B

  // TODO: This is not correct if banks > cols or rows (to handle after MVP).
  io.wOut.valid := io.wRdResp.valid
  io.wOut.bits := VecInit(io.wRdResp.bits.take(c.cols))
  io.aOut.valid := io.rdResp.valid
  io.aOut.bits := VecInit(io.rdResp.bits.take(c.rows))
  io.aFirst := RegNext(firstFire, false.B)
}
