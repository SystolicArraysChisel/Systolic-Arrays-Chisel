package tpu.compute.datapath

import chisel3._
import chisel3.util._

import ifaces._

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



  val idle :: weightReq :: actGap :: actReq :: Nil = Enum(4)
	val state = RegInit(idle)

	val aAddr = Reg(UInt(c.addrW.W))
	val bAddr = Reg(UInt(c.addrW.W))
	val ldA = Reg(UInt(c.dimW.W))
	val ldB = Reg(UInt(c.dimW.W))
	val m = Reg(UInt(c.dimW.W))

	val weightCount = RegInit(0.U(c.dimW.W))
	val actCount = RegInit(0.U(c.dimW.W))
  val gapCount = RegInit(0.U(c.dimW.W))

	val busy = RegInit(false.B)
	val respCount = RegInit(0.U((c.dimW + 1).W)) 
	val total = Reg(UInt((c.dimW + 1).W))



	io.cmd.ready := !busy

	io.rdReq.valid := state === weightReq || state === actReq
	io.rdReq.bits.rowAddr := Mux(
		state === weightReq,
		bAddr,
		aAddr
	) >> c.bankBits

	io.rdResp.ready := true.B

	val isW = respCount < c.rows.U
	io.wOut.valid := io.rdResp.valid && isW
	io.aOut.valid := io.rdResp.valid && !isW
	io.wOut.bits := io.rdResp.bits
	io.aOut.bits := io.rdResp.bits

	when(io.rdResp.fire) {
		when(respCount === total - 1.U) {
			respCount := 0.U
			busy      := false.B
		}.otherwise {
			respCount := respCount + 1.U
		}
	}



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
			when(io.rdReq.fire) {
				bAddr := bAddr -% ldB
				weightCount := weightCount + 1.U
				when(weightCount === (c.rows - 1).U) {
					when(m === 0.U) { 
						state := idle 
					}
					.otherwise {
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
			when(gapCount + 1.U === c.weightToActGap.U) {
				state := actReq
			}.otherwise {
				gapCount := gapCount + 1.U
			}
		}

		is(actReq) {
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