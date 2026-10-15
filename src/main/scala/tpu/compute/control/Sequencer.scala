package tpu.compute.control

import chisel3._
import chisel3.util._

import ifaces._

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

  

  val sIdle :: sIssue :: sWait :: Nil = Enum(3)
  val state = RegInit(sIdle)

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

  val nOff = RegInit(0.U(c.dimW.W))
  val kOff = RegInit(0.U(c.dimW.W))

  val accSent = RegInit(false.B)
  val cmdSent = RegInit(false.B)

  val busyPeCycles = RegInit(0.U(SeqStatus.counterW.W))
  val cycles = RegInit(0.U(SeqStatus.counterW.W))

  val firstK = kOff === 0.U
  val lastK = (kOff + c.rows.U) === k
  val lastN = (nOff + c.cols.U) === n

  io.instr.ready := state === sIdle
  io.status.busy := state =/= sIdle
  io.status.cycles := cycles
  io.status.busyPeCycles := busyPeCycles

  io.accCtrl.valid := (state === sIssue) && !accSent
  io.accCtrl.bits.first := firstK
  io.accCtrl.bits.last  := lastK
  io.accCtrl.bits.cAddr := cAddr + nOff
  io.accCtrl.bits.ldC := ldC
  io.accCtrl.bits.m := m
  io.accCtrl.bits.relu := relu
  io.accCtrl.bits.requant := requant
  io.accCtrl.bits.shift := shift

  io.tileCmd.valid := (state === sIssue) && !cmdSent
  io.tileCmd.bits.aAddr := aAddr + kOff
  io.tileCmd.bits.bAddr := bAddr + kOff * ldB + nOff
  io.tileCmd.bits.ldA := ldA
  io.tileCmd.bits.ldB := ldB
  io.tileCmd.bits.m := m



  switch(state) {
    is(sIdle) {
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

        accSent := false.B
        cmdSent := false.B

        busyPeCycles := 0.U
        cycles := 0.U

        state := sIssue
      }
    }

    is(sIssue) {
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
      when(io.tileDone.valid) {
        busyPeCycles := busyPeCycles + (m << log2Ceil(c.rows * c.cols))

        accSent := false.B
        cmdSent := false.B

        when(!lastK)      { 
          kOff := kOff + c.rows.U; state := sIssue 
        }
        .elsewhen(!lastN) {
          kOff := 0.U
          nOff := nOff + c.cols.U
          
          state := sIssue 
        }
        .otherwise { 
          state := sIdle }
      }
    }
  }
}
