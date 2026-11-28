package tpu.memory

import chisel3._
import chisel3.experimental.{IntParam, StringParam}
import chisel3.util._

/** Xilinx xpm_memory_sdpram: a simple dual-port block RAM with one clock,
  * port A writes and port B reads.
  *
  * Port B reads with a latency of 1 cycle, like a SyncReadMem. If port A
  * writes an address in the same cycle as port B reads it, port B returns
  * the old data (read_first).
  *
  * Only for Vivado: Chisel simulation cannot run a BlackBox, so the tests
  * keep the SyncReadMem version (SAConfig.xpmMemories = false).
  */
class XpmSdpRam(depth: Int, width: Int)
    extends BlackBox(
      Map(
        "MEMORY_SIZE"        -> IntParam(depth * width), // in bits
        "MEMORY_PRIMITIVE"   -> StringParam("block"),
        "CLOCKING_MODE"      -> StringParam("common_clock"),
        "MEMORY_INIT_FILE"   -> StringParam("none"),
        "USE_MEM_INIT"       -> IntParam(0),
        "WRITE_DATA_WIDTH_A" -> IntParam(width),
        "BYTE_WRITE_WIDTH_A" -> IntParam(width),
        "ADDR_WIDTH_A"       -> IntParam(log2Ceil(depth)),
        "READ_DATA_WIDTH_B"  -> IntParam(width),
        "ADDR_WIDTH_B"       -> IntParam(log2Ceil(depth)),
        "READ_LATENCY_B"     -> IntParam(1),
        "WRITE_MODE_B"       -> StringParam("read_first")
      )
    ) {
  override def desiredName = "xpm_memory_sdpram"

  val io = IO(new Bundle {
    val sleep = Input(Bool())

    val clka           = Input(Clock())
    val ena            = Input(Bool())
    val wea            = Input(UInt(1.W))
    val addra          = Input(UInt(log2Ceil(depth).W))
    val dina           = Input(UInt(width.W))
    val injectsbiterra = Input(Bool())
    val injectdbiterra = Input(Bool())

    val clkb     = Input(Clock())
    val rstb     = Input(Bool())
    val enb      = Input(Bool())
    val regceb   = Input(Bool())
    val addrb    = Input(UInt(log2Ceil(depth).W))
    val doutb    = Output(UInt(width.W))
    val sbiterrb = Output(Bool())
    val dbiterrb = Output(Bool())
  })

  /** Connects the clock and ties the ports that a plain RAM without ECC,
    * reset or sleep does not use.
    */
  def init(clock: Clock): Unit = {
    io.clka := clock
    io.clkb := clock
    io.sleep := false.B
    io.rstb := false.B
    io.regceb := true.B
    io.injectsbiterra := false.B
    io.injectdbiterra := false.B
  }
}
