package tpu.board

import chisel3.stage.ChiselStage

import ifaces.SAConfig
import tpu.Tpu
import tpu.io.BoardConfig

/** Writes the Verilog of BoardTop with the Tpu core for Vivado into
  * generated/:
  *
  * sbt "runMain tpu.board.EmitBoard"
  *
  * Then: vivado -mode batch -source fpga/build.tcl     (bitstream and reports in build/)
  *       vivado -mode batch -source fpga/program.tcl   (load it on the board)
  */
object EmitBoard extends App {
  val b = BoardConfig.nexysTpu

  emit(() => new Tpu(SAConfig(), b), "generated", b.clockHz)

  /** chisel3.stage.ChiselStage is deprecated in Chisel 3.6, but unlike
    * circt.stage.ChiselStage it does not need the firtool binary.
    */
  @annotation.nowarn("cat=deprecation")
  def emit(core: () => CoreModule, targetDir: String, coreHz: Int = 100_000_000): String =
    (new ChiselStage).emitVerilog(new BoardTop(core(), coreHz), Array("--target-dir", targetDir))
}
