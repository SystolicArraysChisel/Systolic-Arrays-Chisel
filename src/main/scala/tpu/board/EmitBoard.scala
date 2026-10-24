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
  * Add the generated BoardTop.v and fpga/nexys4ddr.xdc to the Vivado project.
  */
object EmitBoard extends App {
  val b = BoardConfig.nexys

  emit(() => new Tpu(SAConfig(), b), "generated")

  /** chisel3.stage.ChiselStage is deprecated in Chisel 3.6, but unlike
    * circt.stage.ChiselStage it does not need the firtool binary.
    */
  @annotation.nowarn("cat=deprecation")
  def emit(core: () => CoreModule, targetDir: String): String =
    (new ChiselStage).emitVerilog(new BoardTop(core()), Array("--target-dir", targetDir))
}
