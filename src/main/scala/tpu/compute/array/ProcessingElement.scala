package tpu.compute.array

import chisel3._
import chisel3.util._
import chisel3.experimental.{annotate, ChiselAnnotation}

import ifaces._

/**
  * One cell of the weight-stationary systolic array.
  *
  * Double-buffered weight: the shadow weight moves down one PE row on every
  * cycle with wEn = 1 (the shift chain), while the active weight w feeds the
  * multiplier. swapIn copies shadow into w; it travels to the right next to
  * aIn, so it reaches every PE together with the first activation row of a
  * tile, and that row already uses the new weight. The next tile can
  * therefore shift in while the current one computes.
  *
  * Every cycle it computes psumOut = psumIn + aIn * w and forwards aIn and
  * swapIn to the right. All outputs are registered: one register stage per
  * hop (see SAConfig.arrayLatency).
  */
class ProcessingElement(c: SAConfig) extends Module {
  val io = IO(new Bundle {
    // Weight shift chain, top to bottom.
    val wIn = Input(c.inT)
    val wEn = Input(Bool())
    val wOut = Output(c.inT)

    // Activations, left to right.
    val aIn = Input(c.inT)
    val aOut = Output(c.inT)

    // Weight swap, left to right next to aIn.
    val swapIn = Input(Bool())
    val swapOut = Output(Bool())

    // Partial sums, top to bottom.
    val psumIn = Input(c.accT)
    val psumOut = Output(c.accT)
  })

  val shadow = RegEnable(io.wIn, io.wEn)
  io.wOut := shadow

  val w = RegEnable(shadow, io.swapIn)
  // The swap cycle already computes with the new weight: no bubble between tiles.
  val wEff = Mux(io.swapIn, shadow, w)

  io.aOut := RegNext(io.aIn)
  io.swapOut := RegNext(io.swapIn, false.B)

  // + keeps accWidth bits, so an overflow wraps like int32 arithmetic.
  val psum = RegNext(io.psumIn + io.aIn * wEff)
  io.psumOut := psum

  // Vivado builds the 8x8 products from LUTs on its own: force a DSP slice.
  annotate(new ChiselAnnotation {
    def toFirrtl = firrtl.AttributeAnnotation(psum.toTarget, "use_dsp = \"yes\"")
  })
}
