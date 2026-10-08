package ifaces

import chisel3._
import chisel3.util._

/** Ports of the SystolicArray (Andreea), driven by the DataMover and read by
  * the Accumulators (Giovanni). Directions are from the array's point of view.
  * All three use protocol V (valid-only): the array never stalls and its
  * outputs must always be accepted. Written for vecWidth = 1 (MVP).
  */
class SystolicArrayIO(c: SAConfig) extends Bundle {

  /** Weights in, at the top of the array. Each cycle with valid = 1 carries one
    * row of the B tile (cols numbers, one per PE column) and pushes the weights
    * already loaded down by one PE row. The data mover sends the rows last row
    * first, so that after `rows` valid cycles PE row r holds row r of the tile.
    * While valid = 0, the weights stay where they are.
    */
  val wIn = Flipped(Valid(Vec(c.cols, c.inT)))

  /** Activations in, on the left of the array. Each cycle with valid = 1
    * carries one row of A for the current tile (rows numbers: element r goes to
    * PE row r). The row is sent as is; the array staggers it internally (skew).
    * Cycles with valid = 0 are allowed between rows: they are ignored. The
    * first row may arrive weightToActGap + 1 cycles after the last weight row.
    */
  val aIn = Flipped(Valid(Vec(c.rows, c.inT)))

  /** Results out, at the bottom of the array. Each cycle with valid = 1 carries
    * one row of partial sums (cols numbers, one per PE column), exactly
    * arrayLatency cycles after the aIn row it comes from, in the same order and
    * with the same gaps. These are sums over the current K-tile only; the
    * accumulators add them across K-tiles.
    */
  val out = Valid(Vec(c.cols, c.accT))
}
