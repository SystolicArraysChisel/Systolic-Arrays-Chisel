package tpu.compute.datapath

import chisel3._
import chisel3.util._

import ifaces._

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



  val first = RegInit(false.B)
  val last = RegInit(false.B)
  val cAddr = RegInit(0.U(c.outAddrW.W))
  val ldC = RegInit(0.U(c.dimW.W))
  val m = RegInit(0.U(c.dimW.W))
  val relu = RegInit(false.B)
  val requant = RegInit(false.B)
  val shift = RegInit(0.U(c.shiftW.W))

  val sCtrl :: sAccum :: sWrite :: Nil = Enum(3)
  val state = RegInit(sCtrl)

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

  assert(
    !(io.in.valid && state =/= sAccum), 
    "array output while accumulators not ready"
  )



  switch(state) {
    is(sCtrl) {
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
            writeCounter := 0.U
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
      io.resWr.bits.rowAddr := (cAddr + writeCounter * ldC) / c.cols.U

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
        when(writeCounter === m - 1.U) {
          state := sCtrl
        } .otherwise {
          writeCounter := writeCounter + 1.U
        }
      }
    }
  }
}