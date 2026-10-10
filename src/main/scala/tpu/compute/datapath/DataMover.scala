package tpu.compute.datapath

import chisel3._
import chisel3.util._

import ifaces._

/**
  * Streams matrix weights and activations from the unified buffer
  * to the systolic array for one matrix multiplication.
  *
  * Weights are read first, followed by activations. The addresses
  * are updated by the leading dimensions supplied with the command.
  */
class DataMover(c: SAConfig) extends Module {
  val io = IO(new Bundle {
    // Sequencer <-> DataMover interface
    val cmd = Flipped(Decoupled(new TileCmd(c)))

    // UnifiedBuffer <-> DataMover interface
    val rdReq = Decoupled(new BufRowReq(c))
    val rdResp = Flipped(Decoupled(Vec(c.banks, c.inT)))

    // DataMover <-> SystolicArray interface
    val wOut = Valid(Vec(c.cols, c.inT))
    val aOut = Valid(Vec(c.rows, c.inT))
  })


  // The data mover requests weights, optionally waits, and then requests activations.
  val idle :: weightReq :: actGap :: actReq :: Nil = Enum(4)
	val state = RegInit(idle)

  // Current addresses and row strides in the input memory.
	val aAddr = Reg(UInt(c.addrW.W))
	val bAddr = Reg(UInt(c.addrW.W))
	val ldA = Reg(UInt(c.dimW.W))
	val ldB = Reg(UInt(c.dimW.W))
	val m = Reg(UInt(c.dimW.W))

  // Counters track issued rows and the total number of response beats.
	val weightCount = RegInit(0.U(c.dimW.W))
	val actCount = RegInit(0.U(c.dimW.W))
  val gapCount = RegInit(0.U(c.dimW.W))

  // busy prevents a new command from being accepted while a transfer is active.
	val busy = RegInit(false.B)
	val respCount = RegInit(0.U((c.dimW + 1).W)) 
	val total = Reg(UInt((c.dimW + 1).W))


  // Only one command can be processed at a time.
	io.cmd.ready := !busy

  // A request is generated only while the FSM is issuing the corresponding rows.
	io.rdReq.valid := state === weightReq || state === actReq
	io.rdReq.bits.rowAddr := Mux(
		state === weightReq,
		bAddr,
		aAddr
	) >> c.bankBits

	io.rdResp.ready := true.B

	// The first response beats contain weights; the remaining beats contain activations.
	val isW = respCount < c.rows.U
	io.wOut.valid := io.rdResp.valid && isW
	io.aOut.valid := io.rdResp.valid && !isW
	io.wOut.bits := io.rdResp.bits
	io.aOut.bits := io.rdResp.bits

  // Count accepted responses and release the command interface after the final beat.
	when(io.rdResp.fire) {
		when(respCount === total - 1.U) {
			respCount := 0.U
			busy      := false.B
		}.otherwise {
			respCount := respCount + 1.U
		}
	}


  /**
    * Accepts a command and initializes the transfer.
    *
    * The weight address is positioned at the last row because weights
    * are streamed in reverse row order, while activations start at A[0][0].
    */
	switch(state) {
		is(idle) {
			when(io.cmd.fire) {
				aAddr := io.cmd.bits.aAddr
				bAddr := (io.cmd.bits.bAddr + (c.rows - 1).U * io.cmd.bits.ldB)(c.addrW - 1, 0)
				ldA := io.cmd.bits.ldA
				ldB := io.cmd.bits.ldB
				m := io.cmd.bits.m
				total       := c.rows.U +& io.cmd.bits.m
				weightCount := 0.U
				actCount    := 0.U
				busy        := true.B
				state       := weightReq
			}
		}

		is(weightReq) {
      // Issue one request per weight row and move towards the first row.
			when(io.rdReq.fire) {
				bAddr := bAddr -% ldB
				weightCount := weightCount + 1.U
				when(weightCount === (c.rows - 1).U) {
					when(m === 0.U) { 
						state := idle 
					}
					.otherwise {
            // Insert the configured gap before the first activation request.
						if (c.weightToActGap == 0) { 
							state := actReq 
						}
						else { 
							gapCount := 0.U
							state := actGap 
						}
					}
				}
			}
		}

		is(actGap) {
      // Wait the requested number of empty cycles between weights and activations.
			when(gapCount + 1.U === c.weightToActGap.U) {
				state := actReq
			}.otherwise {
				gapCount := gapCount + 1.U
			}
		}

		is(actReq) {
      // Issue one request per activation row and advance by the A row stride.
			when(io.rdReq.fire) {
				aAddr := aAddr +% ldA
				actCount := actCount + 1.U
				when(actCount === m - 1.U) { 
					state := idle 
				}
			}
		}
	}
}