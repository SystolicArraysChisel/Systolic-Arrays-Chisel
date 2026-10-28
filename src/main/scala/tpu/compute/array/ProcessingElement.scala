package tpu.compute.array

import chisel3._
import chisel3.util._
import chisel3.experimental.{annotate, ChiselAnnotation}
import firrtl.AttributeAnnotation

import ifaces._

/**
  * One cell of the weight-stationary systolic array.
  *
  * Holds one weight, which moves down one PE row on every cycle with
  * wEn = 1 and stays put otherwise. Every cycle it computes
  * psumOut = psumIn + aIn * w and forwards aIn to the right. Both outputs
  * are registered: one register stage per hop (see SAConfig.arrayLatency).
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

    // Partial sums, top to bottom.
    val psumIn = Input(c.accT)
    val psumOut = Output(c.accT)
  })

  val w = RegEnable(io.wIn, io.wEn)
  io.wOut := w

  io.aOut := RegNext(io.aIn)

  // + keeps accWidth bits, so an overflow wraps like int32 arithmetic.
  val psum = RegNext(io.psumIn + io.aIn * w)
  io.psumOut := psum

  // Vivado builds the 8x8 products from LUTs on its own: force a DSP slice.
  annotate(new ChiselAnnotation {
    def toFirrtl = AttributeAnnotation(psum.toTarget, "use_dsp = \"yes\"")
  })
}
