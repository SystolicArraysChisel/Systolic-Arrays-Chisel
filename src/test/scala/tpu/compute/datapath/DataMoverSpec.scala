package tpu.compute.datapath

import chiseltest._
import chisel3._
import ifaces._
import org.scalatest.flatspec.AnyFlatSpec

class DataMoverSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "DataMover"



  private val baseConfig = SAConfig(
    rows = 4,
    cols = 4,
    inDepth = 64,
    outDepth = 64,
    maxM = 16
  )

  private def clearInputs(dut: DataMover): Unit = {
    dut.io.cmd.valid.poke(false.B)
    dut.io.rdReq.ready.poke(true.B)
    dut.io.rdResp.valid.poke(false.B)
  }

  private def sendCommand(
		dut: DataMover,
		aAddr: Int,
		bAddr: Int,
		ldA: Int,
		ldB: Int,
		m: Int
  ): Unit = {
    dut.io.cmd.bits.aAddr.poke(aAddr.U)
    dut.io.cmd.bits.bAddr.poke(bAddr.U)
    dut.io.cmd.bits.ldA.poke(ldA.U)
    dut.io.cmd.bits.ldB.poke(ldB.U)
    dut.io.cmd.bits.m.poke(m.U)
    dut.io.cmd.valid.poke(true.B)

    dut.io.cmd.ready.expect(true.B)

    dut.clock.step()
    dut.io.cmd.valid.poke(false.B)
  }

  private def driveResponse(
		dut: DataMover, 
		values: Seq[Int]): Unit = {
			// Drive the response with the given values.
			values.zipWithIndex.foreach { case (value, bank) =>
				dut.io.rdResp.bits(bank).poke(value.S)
			}
			dut.io.rdResp.valid.poke(true.B)
  }



  it should "issue weight and activation requests in order" in {
    val c = baseConfig

    test(new DataMover(c)) { dut =>
      clearInputs(dut)
      sendCommand(dut, aAddr = 4, bAddr = 8, ldA = 4, ldB = 4, m = 2)

      val expectedRequests = Seq(5, 4, 3, 2, 1, 2)
      expectedRequests.zipWithIndex.foreach { case (expectedRow, beat) =>
        dut.io.rdReq.valid.expect(true.B)
        dut.io.rdReq.bits.rowAddr.expect(expectedRow.U)

        val response = (0 until c.banks).map(_ + beat * 10)
        driveResponse(dut, response)
        if (beat < c.rows) {
          dut.io.wOut.valid.expect(true.B)
          dut.io.aOut.valid.expect(false.B)
          response.zipWithIndex.foreach { case (value, bank) =>
            dut.io.wOut.bits(bank).expect(value.S)
          }
        } else {
          dut.io.wOut.valid.expect(false.B)
          dut.io.aOut.valid.expect(true.B)
          response.zipWithIndex.foreach { case (value, bank) =>
            dut.io.aOut.bits(bank).expect(value.S)
          }
        }
        dut.clock.step()
      }

      dut.io.rdResp.valid.poke(false.B)
      dut.io.rdReq.valid.expect(false.B)
      dut.io.cmd.ready.expect(true.B)
    }
  }
}