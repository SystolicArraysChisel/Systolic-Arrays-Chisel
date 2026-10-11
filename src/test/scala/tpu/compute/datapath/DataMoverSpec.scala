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

			// Weight requests: 20 >> 2 = 5, 16 >> 2 = 4, 12 >> 2 = 3, 8 >> 2 = 2, 
			// Activation requests: 4 >> 2 = 1, 8 >> 2 = 2
      val expectedRequests = Seq(5, 4, 3, 2, 1, 2)

      expectedRequests.zipWithIndex.foreach { case (expectedRow, beat) =>
				// Assert that the request is valid and has the expected row address.
        dut.io.rdReq.valid.expect(true.B)
        dut.io.rdReq.bits.rowAddr.expect(expectedRow.U)

				// Simulate the response from the memory for this beat.
				// E.g., for beat 1: [10, 11, 12, 13], etc.
        val response = (0 until c.banks).map(_ + beat * 10) 
        driveResponse(dut, response)

				// If the beat is less than the number of weight rows, 
				// expect wOut to be valid and aOut to be invalid.
        if (beat < c.rows) {
          dut.io.wOut.valid.expect(true.B)
          dut.io.aOut.valid.expect(false.B)

          response.zipWithIndex.foreach { case (value, bank) =>
            dut.io.wOut.bits(bank).expect(value.S)
          }
        } else { // Otherwise, expect aOut to be valid and wOut to be invalid.
          dut.io.wOut.valid.expect(false.B)
          dut.io.aOut.valid.expect(true.B)

          response.zipWithIndex.foreach { case (value, bank) =>
            dut.io.aOut.bits(bank).expect(value.S)
          }
        }
				// Step the clock to move to the next beat.
        dut.clock.step()
      }

      dut.io.rdResp.valid.poke(false.B)

			// After all beats, expect rdReq to be invalid and cmd to be ready for a new command.
      dut.io.rdReq.valid.expect(false.B)
      dut.io.cmd.ready.expect(true.B)
    }
  }

  it should "finish after weights when there are no activation rows" in {
    val c = baseConfig

    test(new DataMover(c)) { dut =>
      clearInputs(dut)
			// Send a command with m = 0, meaning no activation rows.
      sendCommand(dut, aAddr = 0, bAddr = 0, ldA = 1, ldB = 4, m = 0)

      (0 until c.rows).foreach { beat =>
        dut.io.rdReq.valid.expect(true.B)
				// Weights are streamed in reverse order.
        dut.io.rdReq.bits.rowAddr.expect((c.rows - 1 - beat).U)

        driveResponse(dut, Seq.fill(c.banks)(beat + 1))

        dut.io.wOut.valid.expect(true.B)
        dut.io.aOut.valid.expect(false.B)

        dut.clock.step()
      }

      dut.io.rdResp.valid.poke(false.B)

      dut.io.rdReq.valid.expect(false.B)
      dut.io.cmd.ready.expect(true.B)
    }
  }
}

/**
	* TODO: Add tests, including:
	* - weightToActGap > 0.
	* - One activation row (m = 1).
	* - No response from memory for some cycles.
	*/