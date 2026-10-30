package tpu.compute.control

import chisel3._
import chisel3.util._

import ifaces._

/**
  * Sequences tiled matrix-multiplication commands across the data mover
  * and accumulator.
  *
  * The sequencer accepts one decoded instruction, issues the corresponding
  * tile commands, and waits for the accumulator to finish each tile. It
  * advances through the K dimension before moving to the next N tile.
  */
class Sequencer(c: SAConfig) extends Module {
  val io = IO(new Bundle {
    // Command Decoder <-> Sequencer interface
    val instr = Flipped(Decoupled(new Instruction(c)))
    val status = Output(new SeqStatus())

    // Data Mover <-> Sequencer interface
    val tileCmd = Decoupled(new TileCmd(c))

    // Accumulator <-> Sequencer interface
    val tileDone = Flipped(Valid(Bool()))
    val accCtrl = Decoupled(new AccCtrl(c))
  })

  // The sequencer accepts instructions only while it is idle.
  val sIdle :: sIssue :: sWait :: Nil = Enum(3)
  val state = RegInit(sIdle)

  // Latched matrix addresses, dimensions, strides, 
  // and post-processing options.
  val aAddr = RegInit(0.U(c.addrW.W))
  val bAddr = RegInit(0.U(c.addrW.W))
  val cAddr = RegInit(0.U(c.outAddrW.W))
  val m = RegInit(0.U(c.dimW.W))
  val k = RegInit(0.U(c.dimW.W))
  val n = RegInit(0.U(c.dimW.W))
  val ldA = RegInit(0.U(c.dimW.W))
  val ldB = RegInit(0.U(c.dimW.W))
  val ldC = RegInit(0.U(c.dimW.W))
  val requant = RegInit(false.B)
  val relu = RegInit(false.B)
  val shift = RegInit(0.U(c.shiftW.W))

  // Offsets identify the current tile within the K and N dimensions.
  val nOff = RegInit(0.U(c.dimW.W))
  val kOff = RegInit(0.U(c.dimW.W))

  // Running addresses for matrix B tiles. 
  // bColBase tracks the starting address
  // of the current N-column block; bTileAddr 
  // steps through K tiles by row increments.
  val bColBase = RegInit(0.U(c.addrW.W))
  val bTileAddr = RegInit(0.U(c.addrW.W))
  val bRowStep = (ldB << log2Ceil(c.rows))(c.addrW - 1, 0)

  // Track whether the current tile's two commands have been accepted.
  val accSent = RegInit(false.B)
  val cmdSent = RegInit(false.B)

  // Cycle counters reported through the status interface.
  val busyPeCycles = RegInit(0.U(SeqStatus.counterW.W))
  val cycles = RegInit(0.U(SeqStatus.counterW.W))

  // Tile boundary flags used to control accumulation and address updates.
  val firstK = kOff === 0.U
  val lastK = (kOff + c.rows.U) === k
  val lastN = (nOff + c.cols.U) === n

  // An instruction is accepted only in the idle state.
  io.instr.ready := state === sIdle
  io.status.busy := state =/= sIdle
  io.status.cycles := cycles
  io.status.busyPeCycles := busyPeCycles

  // The accumulator and data mover receive one command per tile.
  io.accCtrl.valid := (state === sIssue) && !accSent
  io.accCtrl.bits.first := firstK
  io.accCtrl.bits.last := lastK
  io.accCtrl.bits.cAddr := cAddr + nOff
  io.accCtrl.bits.ldC := ldC
  io.accCtrl.bits.m := m
  io.accCtrl.bits.relu := relu
  io.accCtrl.bits.requant := requant
  io.accCtrl.bits.shift := shift

  io.tileCmd.valid := (state === sIssue) && !cmdSent
  // Advance the input addresses by the current K and N tile offsets.
  io.tileCmd.bits.aAddr := aAddr + kOff
  io.tileCmd.bits.bAddr := bTileAddr
  io.tileCmd.bits.ldA := ldA
  io.tileCmd.bits.ldB := ldB
  io.tileCmd.bits.m := m

  when(state =/= sIdle) {
    cycles := cycles + 1.U
  }

  switch(state) {
    is(sIdle) {
      // Latch a new matrix-multiplication instruction 
      // and start with its first tile.
      when(io.instr.fire) {
        assert(io.instr.bits.op === Opcode.Matmul)
        aAddr := io.instr.bits.aAddr
        bAddr := io.instr.bits.bAddr
        cAddr := io.instr.bits.cAddr
        m := io.instr.bits.m
        k := io.instr.bits.k
        n := io.instr.bits.n
        ldA := io.instr.bits.ldA
        ldB := io.instr.bits.ldB
        ldC := io.instr.bits.ldC
        requant := io.instr.bits.requant
        relu := io.instr.bits.relu
        shift := io.instr.bits.shift

        kOff := 0.U
        nOff := 0.U
        bColBase := io.instr.bits.bAddr
        bTileAddr := io.instr.bits.bAddr

        accSent := false.B
        cmdSent := false.B

        busyPeCycles := 0.U
        cycles := 0.U

        state := sIssue
      }
    }

    is(sIssue) {
      // Wait until both downstream modules have accepted their tile commands.
      when(io.accCtrl.fire) { 
        accSent := true.B 
      }
      when(io.tileCmd.fire) { 
        cmdSent := true.B 
      }
      when((accSent || io.accCtrl.fire) && (cmdSent || io.tileCmd.fire)) { 
        state := sWait 
      }
    }

    is(sWait) {
      // Wait for the accumulator before issuing the next tile.
      when(io.tileDone.valid) {
        busyPeCycles := busyPeCycles + 
          (m << log2Ceil(c.rows * c.cols))

        accSent := false.B
        cmdSent := false.B

        // Complete the next K tile for the current output tile first.
        when(!lastK) { 
          kOff := kOff + c.rows.U
          bTileAddr := bTileAddr + bRowStep
          state := sIssue 
        }
        .elsewhen(!lastN) {
          // Once K is complete, reset K and advance to the next N tile.
          kOff := 0.U
          nOff := nOff + c.cols.U
          val nextColBase = (bColBase + c.cols.U)(c.addrW - 1, 0)
          bColBase := nextColBase
          bTileAddr := nextColBase
          state := sIssue 
        }
        .otherwise { 
          // The complete matrix multiplication has finished.
          state := sIdle
        }
      }
    }
  }
}

/**
  * TODO: Improve performance by not waiting for the 
  * accumulator to finish before issuing the next tile command.
  */