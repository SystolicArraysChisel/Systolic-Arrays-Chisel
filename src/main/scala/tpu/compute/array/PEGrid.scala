package tpu.compute.array

import chisel3._
import chisel3.util._

import ifaces._

/**
  * rows x cols ProcessingElements, without skew or de-skew.
  *
  * Each valid wIn beat pushes the weights down one PE row (the rows of the
  * tile are sent last row first). aIn(i) feeds PE row i and must already
  * be skewed: A[m][i] arrives at cycle m + i. psumOut(j) is the bottom PE of
  * column j, so column j of A row m shows up at cycle m + rows + j.
  */
class PEGrid(c: SAConfig) extends Module {
  val io = IO(new Bundle {
    val wIn = Flipped(Valid(Vec(c.cols, c.inT)))
    val aIn = Input(Vec(c.rows, c.inT))
    val psumOut = Output(Vec(c.cols, c.accT))
  })

  val pes = Seq.fill(c.rows, c.cols)(Module(new ProcessingElement(c)))

  for (i <- 0 until c.rows; j <- 0 until c.cols) {
    val pe = pes(i)(j).io
    pe.wEn := io.wIn.valid
    pe.wIn := (if (i == 0) io.wIn.bits(j) else pes(i - 1)(j).io.wOut)
    pe.aIn := (if (j == 0) io.aIn(i) else pes(i)(j - 1).io.aOut)
    pe.psumIn := (if (i == 0) 0.S else pes(i - 1)(j).io.psumOut)
  }

  for (j <- 0 until c.cols)
    io.psumOut(j) := pes(c.rows - 1)(j).io.psumOut
}
