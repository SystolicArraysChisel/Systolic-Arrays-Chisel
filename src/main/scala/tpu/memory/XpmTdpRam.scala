package tpu.memory

import chisel3._
import chisel3.experimental.{IntParam, StringParam}
import chisel3.util._

/** Xilinx xpm_memory_tdpram: a true dual-port block RAM with one clock.
  *
  * Vivado maps it to block RAM as declared, instead of inferring a memory
  * from the Verilog. Both ports read with a latency of 1 cycle, like a
  * SyncReadMem. If one port writes an address in the same cycle as the
  * other port reads it, the read returns the old data: with one clock and
  * both ports in read_first, XPM treats the collision as safe
  * (IS_COLLISION_SAFE in xpm_memory.sv). In UnifiedBuffer only port A
  * writes, and the protocol keeps it off what port B reads: C must not
  * overlap A or B with toInput, and the host waits for a MATMUL to finish
  * before it writes again.
  *
  * Only for Vivado: Chisel simulation cannot run a BlackBox, so the tests
  * keep the SyncReadMem version (SAConfig.xpmMemories = false).
  */
class XpmTdpRam(depth: Int, width: Int)
    extends BlackBox(
      Map(
        "MEMORY_SIZE"        -> IntParam(depth * width), // in bits
        "MEMORY_PRIMITIVE"   -> StringParam("block"),
        "CLOCKING_MODE"      -> StringParam("common_clock"),
        "MEMORY_INIT_FILE"   -> StringParam("none"),
        "USE_MEM_INIT"       -> IntParam(0),
        "WRITE_DATA_WIDTH_A" -> IntParam(width),
        "READ_DATA_WIDTH_A"  -> IntParam(width),
        "BYTE_WRITE_WIDTH_A" -> IntParam(width),
        "ADDR_WIDTH_A"       -> IntParam(log2Ceil(depth)),
        "READ_LATENCY_A"     -> IntParam(1),
        "WRITE_MODE_A"       -> StringParam("read_first"),
        "WRITE_DATA_WIDTH_B" -> IntParam(width),
        "READ_DATA_WIDTH_B"  -> IntParam(width),
        "BYTE_WRITE_WIDTH_B" -> IntParam(width),
        "ADDR_WIDTH_B"       -> IntParam(log2Ceil(depth)),
        "READ_LATENCY_B"     -> IntParam(1),
        "WRITE_MODE_B"       -> StringParam("read_first")
      )
    ) {
  override def desiredName = "xpm_memory_tdpram"

  val io = IO(new Bundle {
    val sleep = Input(Bool())

    val clka           = Input(Clock())
    val rsta           = Input(Bool())
    val ena            = Input(Bool())
    val regcea         = Input(Bool())
    val wea            = Input(UInt(1.W))
    val addra          = Input(UInt(log2Ceil(depth).W))
    val dina           = Input(UInt(width.W))
    val injectsbiterra = Input(Bool())
    val injectdbiterra = Input(Bool())
    val douta          = Output(UInt(width.W))
    val sbiterra       = Output(Bool())
    val dbiterra       = Output(Bool())

    val clkb           = Input(Clock())
    val rstb           = Input(Bool())
    val enb            = Input(Bool())
    val regceb         = Input(Bool())
    val web            = Input(UInt(1.W))
    val addrb          = Input(UInt(log2Ceil(depth).W))
    val dinb           = Input(UInt(width.W))
    val injectsbiterrb = Input(Bool())
    val injectdbiterrb = Input(Bool())
    val doutb          = Output(UInt(width.W))
    val sbiterrb       = Output(Bool())
    val dbiterrb       = Output(Bool())
  })

  /** Connects the clock and ties the ports that a plain RAM without ECC,
    * reset or sleep does not use.
    */
  def init(clock: Clock): Unit = {
    io.clka := clock
    io.clkb := clock
    io.sleep := false.B
    io.rsta := false.B
    io.rstb := false.B
    io.regcea := true.B
    io.regceb := true.B
    io.injectsbiterra := false.B
    io.injectdbiterra := false.B
    io.injectsbiterrb := false.B
    io.injectdbiterrb := false.B
  }
}
