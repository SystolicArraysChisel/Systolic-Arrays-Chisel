package tpu.compute.datapath

import chisel3._
import chisel3.util._
import chisel3.experimental.{annotate, ChiselAnnotation}

import ifaces._

/**
  * Accumulates partial results produced by the systolic array and writes
  * completed output tiles to the unified buffer.
  *
  * Uses on-chip Block RAM (SyncReadMem) with a pipelined read-modify-write
  * datapath for accumulation across K-tiles, and a 2-stage pipelined
  * post-processing write datapath (Stage 1: rounding sum; Stage 2: 
  * variable shift, saturation, ReLU) to achieve high clock frequency into 
  * the unified buffer.
  */
class Accumulators(c: SAConfig) extends Module {
  val io = IO(new Bundle {
    // SystolicArray <-> Accumulators interface
    val in = Flipped(Valid(Vec(c.cols, c.accT)))

    // Sequencer <-> Accumulators interface
    val ctrl = Flipped(Decoupled(new AccCtrl(c)))
    val done = Valid(Bool())

    // Accumulator <-> UnifiedBuffer interface
    val resWr = Decoupled(new BufResultWrite(c))
  })

  // Control fields latched when a new tile command is accepted.
  val first = RegInit(false.B)
  val last = RegInit(false.B)
  val cAddr = RegInit(0.U(c.outAddrW.W))
  val ldC = RegInit(0.U(c.dimW.W))
  val m = RegInit(0.U(c.dimW.W))
  val relu = RegInit(false.B)
  val requant = RegInit(false.B)
  val shift = RegInit(0.U(c.shiftW.W))

  // Control, accumulation, and write phases of the module.
  val sCtrl :: sAccum :: sWrite :: Nil = Enum(3)
  val state = RegInit(sCtrl)

  // Accumulator storage: one SyncReadMem per column, mapped to BRAM
  val accMem = Seq.fill(c.cols) {
    val mem = SyncReadMem(c.maxM, c.accT)
    annotate(new ChiselAnnotation {
      def toFirrtl = firrtl.AttributeAnnotation(mem.toTarget, "ram_style = \"block\"")
    })
    mem
  }

  // Row counter for incoming rows from the systolic array.
  val rowCounter = RegInit(0.U(c.dimW.W))

  // Accumulation pipeline (Read-Modify-Write).
  // Stage 0: sample incoming array row, issue 
  // read to accMem (if not first tile).
  // Stage 1: read data arrives from accMem, 
  // add to incoming data, write to accMem.
  val accumPipeValid = RegInit(false.B)
  val accumPipeRow = Reg(UInt(c.dimW.W))
  val accumPipeData = Reg(Vec(c.cols, c.accT))
  val accumPipeFirst = RegInit(false.B)
  val accumPipeLastRow = RegInit(false.B)

  // Stage 0 registration.
  when(state === sAccum && io.in.valid) {
    accumPipeValid := true.B
    accumPipeRow := rowCounter
    accumPipeData := io.in.bits
    accumPipeFirst := first
    accumPipeLastRow := (rowCounter === m - 1.U)
  } .otherwise {
    accumPipeValid := false.B
    accumPipeLastRow := false.B
  }

  // Write-Back pipeline.
  // Stage R: generate read address for accMem.
  // Stage Rnd: BRAM data arrives, compute 33-bit 
  // widening + rounding sum, and register into pipeRnd.
  // Stage P: register rounded data, compute variable barrel shift,
  // symmetric int8 clamp, ReLU, and register into pipeP.
  // Stage O: drive io.resWr decoupled interface.
  val wrReadCounter = RegInit(0.U(c.dimW.W))
  val wrRowAddr = RegInit(0.U(c.outAddrW.W))
  val wrReadsDone = RegInit(false.B)

  // R -> Rnd pipeline registers.
  val pipeR_valid = RegInit(false.B)
  val pipeR_rdAddr = RegInit(0.U(c.dimW.W))
  val pipeR_rowAddr = Reg(UInt(c.outAddrW.W))
  val pipeR_last = RegInit(false.B)

  // Rnd -> P pipeline registers.
  val pipeRnd_valid = RegInit(false.B)
  val pipeRnd_rowAddr = Reg(UInt(c.outAddrW.W))
  val pipeRnd_rounded = Reg(Vec(c.cols, SInt((c.accWidth + 1).W)))
  val pipeRnd_raw = Reg(Vec(c.cols, c.accT))
  val pipeRnd_last = RegInit(false.B)

  // P -> O pipeline registers.
  val pipeP_valid = RegInit(false.B)
  val pipeP_rowAddr = Reg(UInt(c.outAddrW.W))
  val pipeP_data = Reg(Vec(c.cols, c.accT))
  val pipeP_last = RegInit(false.B)

  val stageP_ready = !pipeP_valid || io.resWr.ready
  val stageRnd_ready = !pipeRnd_valid || stageP_ready

  // Unified BRAM Read Port (shared between sAccum and sWrite).
  val accumRdEn = (state === sAccum) && io.in.valid && !first
  val wrHoldRd = (state === sWrite) && pipeR_valid && !stageRnd_ready
  val wrRdEn = (state === sWrite) && !wrReadsDone && stageRnd_ready

  val rdAddr = Mux(state === sWrite, Mux(wrHoldRd, pipeR_rdAddr, wrReadCounter), rowCounter)
  val rdEn = Mux(state === sWrite, wrRdEn || wrHoldRd, accumRdEn)

  val memRdData = Wire(Vec(c.cols, c.accT))
  for (col <- 0 until c.cols) {
    memRdData(col) := accMem(col).read(rdAddr, rdEn)
  }

  // Stage 1 write datapath.
  when(accumPipeValid) {
    for (col <- 0 until c.cols) {
      val writeData = Mux(
        accumPipeFirst,
        accumPipeData(col),
        memRdData(col) + accumPipeData(col)
      )
      accMem(col).write(accumPipeRow, writeData)
    }
  }

  // Combinational stage 1.
  val roundedData = Wire(Vec(c.cols, SInt((c.accWidth + 1).W)))
  for (col <- 0 until c.cols) {
    val value = memRdData(col)
    val widenedValue = Cat(value.head(1), value).asSInt
    val rounding = Mux(
      shift === 0.U,
      0.S((c.accWidth + 1).W),
      (1.S((c.accWidth + 1).W) << (shift - 1.U)).asSInt
    )
    roundedData(col) := widenedValue + rounding
  }

  // Combinational stage 2.
  val postProcessed = Wire(Vec(c.cols, c.accT))
  for (col <- 0 until c.cols) {
    val rounded = pipeRnd_rounded(col)
    val rawValue = pipeRnd_raw(col)
    val widenedValue = Cat(rawValue.head(1), rawValue).asSInt

    val requantized = (rounded >> shift).asSInt
    val maxValue = 127.S((c.accWidth + 1).W)
    val minValue = (-127).S((c.accWidth + 1).W)
    val saturated = Mux(
      requantized > maxValue,
      maxValue,
      Mux(requantized < minValue, minValue, requantized)
    )
    val processed = Mux(requant, saturated, widenedValue)
    postProcessed(col) := Mux(
      relu,
      Mux(processed < 0.S, 0.S, processed),
      processed
    )
  }

  // Default interface assignments.
  io.ctrl.ready := (state === sCtrl)
  io.done.valid := false.B
  io.done.bits := true.B

  io.resWr.valid := pipeP_valid
  io.resWr.bits.rowAddr := pipeP_rowAddr
  io.resWr.bits.data := pipeP_data

  assert(!(io.in.valid && state =/= sAccum), "array output while accumulators not ready")

  switch(state) {
    is(sCtrl) {
      pipeR_valid := false.B
      pipeRnd_valid := false.B
      pipeP_valid := false.B

      when(io.ctrl.fire) {
        first := io.ctrl.bits.first
        last := io.ctrl.bits.last
        cAddr := io.ctrl.bits.cAddr
        ldC := io.ctrl.bits.ldC
        m := io.ctrl.bits.m
        relu := io.ctrl.bits.relu
        requant := io.ctrl.bits.requant
        shift := io.ctrl.bits.shift

        rowCounter := 0.U
        state := sAccum
      }
    }

    is(sAccum) {
      when(io.in.valid) {
        when(rowCounter === m - 1.U) {
          rowCounter := 0.U
          when(!last) {
            io.done.valid := true.B
            state := sCtrl
          }
        } .otherwise {
          rowCounter := rowCounter + 1.U
        }
      }

      // Transition to sWrite when the last row 
      // has been committed by stage 1.
      when(accumPipeValid && accumPipeLastRow && last) {
        state := sWrite
        wrReadCounter := 0.U
        wrRowAddr := cAddr >> log2Ceil(c.cols)
        wrReadsDone := false.B
        pipeR_valid := false.B
        pipeRnd_valid := false.B
        pipeP_valid := false.B
      }
    }

    is(sWrite) {
      // Stage R: generate read address for accMem.
      when(stageRnd_ready) {
        when(!wrReadsDone) {
          pipeR_valid := true.B
          pipeR_rdAddr := wrReadCounter
          pipeR_rowAddr := wrRowAddr
          pipeR_last := (wrReadCounter === m - 1.U)

          wrRowAddr := wrRowAddr + (ldC >> log2Ceil(c.cols))
          when(wrReadCounter === m - 1.U) {
            wrReadsDone := true.B
          } .otherwise {
            wrReadCounter := wrReadCounter + 1.U
          }
        } .otherwise {
          pipeR_valid := false.B
        }
      }

      // Stage Rnd: register rounding sum into pipeRnd.
      when(stageRnd_ready) {
        pipeRnd_valid := pipeR_valid
        when(pipeR_valid) {
          pipeRnd_rounded := roundedData
          pipeRnd_raw := memRdData
          pipeRnd_rowAddr := pipeR_rowAddr
          pipeRnd_last := pipeR_last
        }
      }

      // Stage P: register post-processed data into pipeP.
      when(stageP_ready) {
        pipeP_valid := pipeRnd_valid
        when(pipeRnd_valid) {
          pipeP_data := postProcessed
          pipeP_rowAddr := pipeRnd_rowAddr
          pipeP_last := pipeRnd_last
        }
      }

      // Stage O: handshake with UnifiedBuffer.
      when(io.resWr.fire && pipeP_last) {
        io.done.valid := true.B
        state := sCtrl
      }
    }
  }
}
