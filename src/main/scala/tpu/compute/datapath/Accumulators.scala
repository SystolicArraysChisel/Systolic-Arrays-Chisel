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

  val ctrl :: accum :: write :: Nil = Enum(3)
  val state = RegInit(ctrl)

  val accReg = RegInit(
    VecInit(Seq.fill(c.maxM)(
      VecInit(Seq.fill(c.cols)(0.S(c.accWidth.W)))
    ))
  )
  val rowCounter = RegInit(0.U(c.dimW.W))
  val writeCounter = RegInit(0.U(c.dimW.W))

  io.ctrl.ready := (state === ctrl)
  io.resWr.valid := (state === write)
  io.done.valid := false.B



  switch(state) {
    is(ctrl) {
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

        state := accum
      }
    }

    is(accum) {
      when(io.in.valid) {
        when(first) {
          accReg(rowCounter) := io.in.bits
        } .otherwise {
          accReg(rowCounter) := (accReg(rowCounter) zip io.in.bits).map { 
            case (a, b) => a + b 
          }
        }

        when(rowCounter === m - 1.U) {
          when(last) {
            writeCounter := 0.U
            state := write
          } .otherwise {
            io.done.valid := true.B
          }
        } .otherwise(
          rowCounter := rowCounter + 1.U
        )
      }
    }

    is(write) {
      io.resWr.bits.rowAddr := (cAddr + writeCounter * ldC) / c.cols.U

      for (col <- 0 until c.cols) {
        val value = accReg(writeCounter)(col)

        // Widen the value to (c.accWidth + 1) bits to avoid 
        // overflow during rounding and saturation.
        val widenedValue = Cat(value.head(1), value).asSInt

        val rounding = Mux(
          shift === 0.U,
          0.S((c.accWidth + 1).W),
          (1.S((c.accWidth + 1).W) << (shift - 1.U)).asSInt
        )
        val rounded = widenedValue + rounding

        val requantized = (rounded >> shift).asSInt

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

        io.resWr.bits.data(col) := Mux(
          relu,
          Mux(processed < 0.S, 0.S, processed),
          processed
        )
      }

      when(io.resWr.fire) {
        when(writeCounter === m - 1.U) {
          io.done.valid := true.B
          state := ctrl
        } .otherwise {
          writeCounter := writeCounter + 1.U
        }
      }
    }
  }
}