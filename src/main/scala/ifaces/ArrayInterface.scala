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
    * row of the B tile (cols numbers, one per PE column) and pushes the shadow
    * weights already loaded down by one PE row. The data mover sends the rows
    * last row first, so that after `rows` valid cycles PE row r holds row r of
    * the tile in its shadow register. While valid = 0, they stay where they are.
    * The PEs compute with a separate active weight, so the next tile may load
    * while the current one computes; see aFirst for the allowed window.
    */
  val wIn = Flipped(Valid(Vec(c.cols, c.inT)))

  /** Activations in, on the left of the array. Each cycle with valid = 1
    * carries one row of A for the current tile (rows numbers: element r goes to
    * PE row r). The row is sent as is; the array staggers it internally (skew).
    * Cycles with valid = 0 are allowed between rows: they are ignored. The
    * first row may arrive weightToActGap + 1 cycles after the last weight row.
    */
  val aIn = Flipped(Valid(Vec(c.rows, c.inT)))

  /** Set together with the first aIn row of every tile (ignored while aIn.valid
    * = 0). It makes the shadow weights active, starting with that row. With t0
    * the cycle of that row, the next tile's wIn beats must all fall within
    * [t0 + arrayLatency, t1 - 1], where t1 is the next tile's first aIn row:
    * the swap must have left the array before the shadows shift again, and
    * the shadows must be full before the next swap.
    */
  val aFirst = Input(Bool())

  /** Results out, at the bottom of the array. Each cycle with valid = 1 carries
    * one row of partial sums (cols numbers, one per PE column), exactly
    * arrayLatency cycles after the aIn row it comes from, in the same order and
    * with the same gaps. These are sums over the current K-tile only; the
    * accumulators add them across K-tiles.
    */
  val out = Valid(Vec(c.cols, c.accT))
}
