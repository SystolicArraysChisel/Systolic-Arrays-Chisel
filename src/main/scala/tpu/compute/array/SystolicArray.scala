package tpu.compute.array

import chisel3._
import chisel3.util._

import ifaces._

/**
  * Weight-stationary systolic array behind SystolicArrayIO.
  *
  * Skew: aIn element i is delayed i cycles before entering PE row i.
  * De-skew: column j leaves the grid at rows + j, so it is delayed
  * cols - 1 - j more cycles. Every out row therefore appears exactly
  * c.arrayLatency = rows + cols - 1 cycles after its aIn row.
  */
class SystolicArray(c: SAConfig) extends Module {
  val io = IO(new SystolicArrayIO(c))

  val grid = Module(new PEGrid(c))

  grid.io.wIn := io.wIn

  for (i <- 0 until c.rows)
    grid.io.aIn(i) := ShiftRegister(io.aIn.bits(i), i)

  for (j <- 0 until c.cols)
    io.out.bits(j) := ShiftRegister(grid.io.psumOut(j), c.cols - 1 - j)

  // Gap cycles flow through the PEs as junk; only valid marks real rows.
  io.out.valid := ShiftRegister(io.aIn.valid, c.arrayLatency, false.B, true.B)
}
