package tpu.board

import java.nio.file.{Files, Paths}

import chisel3.stage.ChiselStage

import ifaces.SAConfig
import tpu.Tpu
import tpu.io.BoardConfig

/** Writes the Verilog of BoardTop with the Tpu core for Vivado into
  * generated/<config>/, with config.json, the parameters the host needs:
  *
  * sbt "runMain tpu.board.EmitBoard"         -- 4x4 array (default)
  * sbt "runMain tpu.board.EmitBoard 8x8"     -- one of EmitBoard.configs
  *
  * Then: vivado -mode batch -source fpga/build.tcl -tclargs 8x8     (build/8x8/)
  *       vivado -mode batch -source fpga/program.tcl -tclargs 8x8   (load it on the board)
  */
object EmitBoard {
  val b = BoardConfig.nexysTpu

  /** The array sizes that can be built for the board. */
  val configs: Map[String, SAConfig] = Map(
    "2x2" -> SAConfig(rows = 2, cols = 2),
    "4x4" -> SAConfig(),
    "8x8" -> SAConfig(rows = 8, cols = 8)
  )

  def main(args: Array[String]): Unit = {
    val name = args.headOption.getOrElse("4x4")
    val c = configs.getOrElse(name,
      sys.error(s"unknown config '$name' (available: ${configs.keys.toSeq.sorted.mkString(", ")})"))
    val dir = s"generated/$name"
    emit(() => new Tpu(c, b), dir, b.clockHz)
    Files.write(Paths.get(dir, "config.json"), configJson(name, c, b).getBytes("UTF-8"))
  }

  /** The parameters of a build the host needs (tpu_host.config.HwConfig). */
  def configJson(name: String, c: SAConfig, b: BoardConfig): String =
    Seq(
      "name" -> s"\"$name\"", "rows" -> c.rows, "cols" -> c.cols, "in_depth" -> c.inDepth,
      "out_depth" -> c.outDepth, "max_m" -> c.maxM, "acc_width" -> c.accWidth, "baud" -> b.baud,
      "rx_fifo_depth" -> b.rxFifoDepth, "clock_hz" -> b.clockHz
    ).map { case (k, v) => s"  \"$k\": $v" }.mkString("{\n", ",\n", "\n}\n")

  /** chisel3.stage.ChiselStage is deprecated in Chisel 3.6, but unlike
    * circt.stage.ChiselStage it does not need the firtool binary.
    */
  @annotation.nowarn("cat=deprecation")
  def emit(core: () => CoreModule, targetDir: String, coreHz: Int = 100_000_000): String =
    (new ChiselStage).emitVerilog(new BoardTop(core(), coreHz), Array("--target-dir", targetDir))
}
