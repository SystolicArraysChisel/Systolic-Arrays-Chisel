package tpu.compute.datapath

import chisel3._
import chisel3.util._

import ifaces._

/**
  * Accumulates partial results produced by the systolic array and writes
  * completed output tiles to the unified buffer.
  *
  * The module accepts one control command for each accumulation phase. It
  * stores the partial sums by row, optionally combines them with previously
  * accumulated values, and applies requantization, saturation, and ReLU
  * before writing the final results.
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


  // Control fields are latched when a new accumulation phase is accepted.
  val first = RegInit(false.B)
  val last = RegInit(false.B)
  val cAddr = RegInit(0.U(c.outAddrW.W))
  val ldC = RegInit(0.U(c.dimW.W))

  // Keep the current output row address to avoid multiplying by writeCounter.
  val rowAddr = RegInit(0.U(c.outAddrW.W))
  
  // Post-processing options for values written to the unified buffer.
  val m = RegInit(0.U(c.dimW.W))
  val relu = RegInit(false.B)
  val requant = RegInit(false.B)
  val shift = RegInit(0.U(c.shiftW.W))

  // Control, accumulation, and write phases of the module.
  val sCtrl :: sAccum :: sWrite :: Nil = Enum(3)
  val state = RegInit(sCtrl)

  // Temporary storage for the partial sums of the current output tile.
  val accReg = RegInit(
    VecInit(Seq.fill(c.maxM)(
      VecInit(Seq.fill(c.cols)(0.S(c.accWidth.W)))
    ))
  )
  val rowCounter = RegInit(0.U(c.dimW.W))
  val writeCounter = RegInit(0.U(c.dimW.W))

  io.ctrl.ready := (state === sCtrl)
  io.resWr.valid := (state === sWrite)
  io.done.valid := false.B
  io.resWr.bits := DontCare
  io.done.bits := false.B

  // The systolic array may only provide data during the accumulation phase.
  assert(
    !(io.in.valid && state =/= sAccum), 
    "array output while accumulators not ready"
  )



  switch(state) {
    is(sCtrl) {
      // Load the command for the next accumulation phase.
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
        // The first tile initializes the accumulator; later tiles are added.
        when(first) {
          accReg(rowCounter) := io.in.bits
        } .otherwise {
          accReg(rowCounter) := (accReg(rowCounter) zip io.in.bits).map { 
            case (a, b) => a + b 
          }
        }

        when(rowCounter === m - 1.U) {
          io.done.valid := true.B
          when(last) {
            // The final tile is followed by a write phase.
            writeCounter := 0.U
            // cAddr and ldC are multiples of cols, so division becomes a shift.
            rowAddr := cAddr >> log2Ceil(c.cols)
            state := sWrite
          } .otherwise {
            state := sCtrl
          }
        } .otherwise(
          rowCounter := rowCounter + 1.U
        )
      }
    }

    is(sWrite) {
      // Emit one accumulated row per handshake with the unified buffer.
      io.resWr.bits.rowAddr := rowAddr

      for (col <- 0 until c.cols) {
        val value = accReg(writeCounter)(col)

        // Widen the value to (c.accWidth + 1) bits to avoid 
        // overflow during rounding and saturation.
        val widenedValue = Cat(value.head(1), value).asSInt

        // Round the value before shifting to avoid losing precision.
        val rounding = Mux(
          shift === 0.U,
          0.S((c.accWidth + 1).W),
          (1.S((c.accWidth + 1).W) << (shift - 1.U)).asSInt
        )
        val rounded = widenedValue + rounding

        // Shift the rounded value to get the final result.
        val requantized = (rounded >> shift).asSInt

        // Saturate the requantized value to the range [-127, 127].
        val maxValue = 127.S((c.accWidth + 1).W)
        val minValue = (-127).S((c.accWidth + 1).W)
        val saturated = Mux(
          requantized > maxValue,
          maxValue,
          Mux(requantized < minValue, 
            minValue, 
            requantized
          )
        )

        val processed = Mux(requant, saturated, widenedValue)

        // Apply ReLU if enabled, otherwise use the processed value directly.
        io.resWr.bits.data(col) := Mux(
          relu,
          Mux(processed < 0.S, 0.S, processed),
          processed
        )
      }

      when(io.resWr.fire) {
        // Incrementing replaces (cAddr + writeCounter * ldC) / c.cols.U.
        rowAddr := rowAddr + (ldC >> log2Ceil(c.cols))

        when(writeCounter === m - 1.U) {
          state := sCtrl
        } .otherwise {
          writeCounter := writeCounter + 1.U
        }
      }
    }
  }
}

/**
  * TODO: Accumulators in too big registers. 
  * maxM × cols × 32 flip-flops, and with RegInit (the spec says no reset 
  * is needed). With maxM = 256 and cols = 8, that's 65,536 FFs, about half 
  * of the 126,800 of the A7-100T, plus a 256:1 mux on 256 bits. For the MVP 
  * it's fine with small maxM (32 or 64) and Reg(Vec(...)) without reset; 
  * then switch to a SyncReadMem (BRAM) with pipelined read-modify-write.
  */