package tpu.board

import chisel3._
import chisel3.experimental.DoubleParam

/** Clock generation for BoardTop: makes the core clock (50 MHz) from the
  * 100 MHz board clock.
  *
  * MMCME2_BASE and BUFG are hardware blocks built into the FPGA, not logic
  * we describe. Chisel declares them as BlackBoxes: it only writes an
  * instance of a module with that name (and those parameters) into the
  * Verilog, and Vivado recognizes the name as its own primitive.
  */

/** MMCM (Mixed-Mode Clock Manager): the FPGA's clock generator.
  *
  * It multiplies the input clock up to a fast internal clock, the VCO, then
  * divides the VCO down to the output clock:
  *
  *   VCO    = input * mult      100 MHz * 10 = 1 GHz
  *   output = VCO / divide      1 GHz / 20   = 50 MHz
  *
  * The constructor arguments become the Verilog parameters of the primitive.
  * Only the ports ClockGen uses are declared; see Xilinx UG953 for the rest.
  */
class MMCME2_BASE(mult: Double, divide: Double, inPeriodNs: Double)
    extends BlackBox(
      Map(
        "CLKIN1_PERIOD" -> DoubleParam(inPeriodNs), // period of the input clock in ns (10.0 for 100 MHz)
        "CLKFBOUT_MULT_F" -> DoubleParam(mult), // VCO = input * mult
        "CLKOUT0_DIVIDE_F" -> DoubleParam(divide) // CLKOUT0 = VCO / divide
      )
    ) {
  val io = IO(new Bundle {
    val CLKIN1 = Input(Clock()) // input clock
    val CLKOUT0 = Output(Clock()) // output clock
    // The MMCM keeps its output in phase with its input through a feedback
    // loop that leaves the block: CLKFBOUT must be wired back to CLKFBIN.
    val CLKFBOUT = Output(Clock())
    val CLKFBIN = Input(Clock())
    val LOCKED = Output(Bool()) // high once CLKOUT0 is stable
    val RST = Input(Bool()) // restarts the MMCM (unused: tied to 0)
    val PWRDWN = Input(Bool()) // turns the MMCM off (unused: tied to 0)
  })
}

/** Global clock buffer: puts a clock on the FPGA's dedicated clock network,
  * which reaches every flip-flop at nearly the same time. A clock made by an
  * MMCM goes through one before it drives any logic.
  */
class BUFG extends BlackBox {
  val io = IO(new Bundle {
    val I = Input(Clock())
    val O = Output(Clock())
  })
}

/** Makes a clock of outHz from a clock of inHz (the 100 MHz board clock).
  *
  * The VCO always runs at 1 GHz, inside the 600-1200 MHz range the MMCM
  * supports on our FPGA (-1 speed grade), so:
  *
  *   mult   = 1 GHz / inHz      must be a whole number from 2 to 64
  *   divide = 1 GHz / outHz     must be a multiple of 1/8 from 1 to 128
  *
  * For example 50 MHz (/20), 40 MHz (/25) or 80 MHz (/12.5) work, but not
  * 30 MHz (/33.33...). Elaboration stops with an error for a frequency the
  * MMCM cannot make.
  *
  * A RawModule: it has no implicit clock or reset, since it makes a clock
  * instead of running on one.
  *
  * No timing constraint is needed for the output: Vivado derives it from the
  * create_clock on CLK100MHZ in the .xdc and the MMCM parameters.
  */
class ClockGen(outHz: Int, inHz: Int = 100_000_000) extends RawModule {
  val vcoHz = 1_000_000_000.0
  val mult = vcoHz / inHz
  val divide = vcoHz / outHz
  require(mult == mult.round && mult >= 2 && mult <= 64, s"no integer multiplier for $inHz Hz in")
  require((divide * 8) == (divide * 8).round && divide >= 1 && divide <= 128,
    s"$outHz Hz is not 1 GHz divided by a multiple of 1/8 between 1 and 128")

  val clkIn = IO(Input(Clock())) // the board clock
  val clkOut = IO(Output(Clock())) // the generated clock
  val locked = IO(Output(Bool())) // clkOut is stable; BoardTop holds the core in reset until then

  val mmcm = Module(new MMCME2_BASE(mult, divide, 1e9 / inHz))
  mmcm.io.CLKIN1 := clkIn
  mmcm.io.CLKFBIN := mmcm.io.CLKFBOUT
  mmcm.io.RST := false.B
  mmcm.io.PWRDWN := false.B

  val buf = Module(new BUFG)
  buf.io.I := mmcm.io.CLKOUT0
  clkOut := buf.io.O
  locked := mmcm.io.LOCKED
}
