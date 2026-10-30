package tpu.memory

import chiseltest._
import chisel3._
import ifaces._
import org.scalatest.flatspec.AnyFlatSpec

class UnifiedBufferSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "UnifiedBuffer"



  private val configs = Seq(1, 2, 4, 8).map { size =>
    SAConfig(
      rows = size,
      cols = size,
      inDepth = 64,
      outDepth = 64,
      maxM = 16
    )
  }

  private def clearInputs(dut: UnifiedBuffer): Unit = {
    dut.io.hostWr.valid.poke(false.B)
    dut.io.hostRdReq.valid.poke(false.B)
    dut.io.hostRdResp.ready.poke(true.B)

    dut.io.mvRdReq.valid.poke(false.B)
    dut.io.mvRdResp.ready.poke(true.B) // DataMover always accepts a row read.

    dut.io.resWr.valid.poke(false.B)
  }

  private def writeInput(
      dut: UnifiedBuffer,
      c: SAConfig,
      addr: Int,
      data: Int
  ): Unit = {
    dut.io.hostWr.bits.addr.poke(addr.U)
    dut.io.hostWr.bits.data.poke(data.S)
    dut.io.hostWr.valid.poke(true.B)

    dut.io.hostWr.ready.expect(true.B)

    dut.clock.step()
    dut.io.hostWr.valid.poke(false.B)
  }

  private def readInputRow(
      dut: UnifiedBuffer,
      c: SAConfig,
      rowAddr: Int
  ): Seq[Int] = {
    dut.io.mvRdReq.bits.rowAddr.poke(rowAddr.U)
    dut.io.mvRdReq.valid.poke(true.B)

    dut.io.mvRdReq.ready.expect(true.B)

    dut.clock.step()
    dut.io.mvRdReq.valid.poke(false.B)

    dut.io.mvRdResp.valid.expect(true.B)
    val result = (0 until c.banks).map { bank =>
			// Conversion from SInt to Int is done using peek().litValue.toInt.
      dut.io.mvRdResp.bits(bank).peek().litValue.toInt
    }
		
    dut.clock.step()
    result
  }

  private def writeResultRow(
      dut: UnifiedBuffer,
      c: SAConfig,
      rowAddr: Int,
      data: Seq[Int]
  ): Unit = {
    dut.io.resWr.bits.rowAddr.poke(rowAddr.U)
    data.zipWithIndex.foreach { case (value, col) =>
      dut.io.resWr.bits.data(col).poke(value.S)
    }
    dut.io.resWr.valid.poke(true.B)

    dut.io.resWr.ready.expect(true.B)

    dut.clock.step()
    dut.io.resWr.valid.poke(false.B)
  }

  private def requestHostRead(
      dut: UnifiedBuffer,
      c: SAConfig,
      addr: Int
  ): Unit = {
    dut.io.hostRdReq.bits.addr.poke(addr.U)
    dut.io.hostRdReq.valid.poke(true.B)
    dut.io.hostRdReq.ready.expect(true.B)

    dut.clock.step()
    dut.io.hostRdReq.valid.poke(false.B)

    // One cycle for the synchronous memory read and one for response capture.
    dut.clock.step()
  }



  it should "write and read input rows for every supported array size" in {
    configs.foreach { c =>
      test(new UnifiedBuffer(c)) { dut =>
        clearInputs(dut)

        val row = 3

        (0 until c.banks).foreach { bank =>
					// Write one element to each bank in the selected row.
          writeInput(dut, c, bank + (row << c.bankBits), 20 + bank)
        }

        val actual = readInputRow(dut, c, row)

        assert(actual == (0 until c.banks).map(bank => 20 + bank))
      }
    }
  }

  it should "write and read result rows for every supported array size" in {
    configs.foreach { c =>
      test(new UnifiedBuffer(c)) { dut =>
        clearInputs(dut)

        val row = 2
        val expected = (0 until c.cols).map(col => 100 + col)

        writeResultRow(dut, c, row, expected)

        (0 until c.cols).foreach { col =>
          requestHostRead(dut, c, col + (row << c.colBits))

          dut.io.hostRdResp.valid.expect(true.B)
          dut.io.hostRdResp.bits.expect(expected(col).S)
          dut.io.hostRdResp.ready.poke(true.B)

          dut.clock.step()
        }
      }
    }
  }

  it should "hold a host response stable while the response is backpressured" in {
    configs.foreach { c =>
      test(new UnifiedBuffer(c)) { dut =>
        clearInputs(dut)

        val row = 1
        val col = c.cols - 1
        val expected = 73

        writeResultRow(dut, c, row, Seq.fill(c.cols)(expected))

        requestHostRead(dut, c, col + (row << c.colBits))

				// Backpressure the response.
        dut.io.hostRdResp.ready.poke(false.B) 

				// The request is still pending because the response is not accepted.
        dut.io.hostRdReq.ready.expect(false.B) 
				// The response is valid and holds the expected value.
        dut.io.hostRdResp.valid.expect(true.B)
        val heldBits = dut.io.hostRdResp.bits.peek().litValue

        for (_ <- 0 until 10) { // Hold the response for 10 cycles.
          dut.io.hostRdResp.valid.expect(true.B)
          dut.io.hostRdResp.bits.expect(heldBits.S)
          dut.io.hostRdReq.ready.expect(false.B)

          dut.clock.step()
        }

        dut.io.hostRdResp.ready.poke(true.B) // Accept the response.

        dut.io.hostRdResp.valid.expect(true.B)
        dut.io.hostRdResp.bits.expect(expected.S)

        dut.clock.step()
				// The response is no longer valid after being accepted.
        dut.io.hostRdResp.valid.expect(false.B)
				// The request is now ready to accept a new read request.
				dut.io.hostRdReq.ready.expect(true.B) 
      }
    }
  }

  it should "keep result columns aligned across consecutive reads" in {
    configs.foreach { c =>
      test(new UnifiedBuffer(c)) { dut =>
        clearInputs(dut)

        val row = 0
        val values = (0 until c.cols).map(col => 200 + col)

        writeResultRow(dut, c, row, values)

        (0 until c.cols).foreach { col =>
          requestHostRead(dut, c, col)

          dut.io.hostRdResp.valid.expect(true.B)
          dut.io.hostRdResp.bits.expect(values(col).S)

					// Backpressure the response every other column to test alignment.
          dut.io.hostRdResp.ready.poke(col % 2 == 0)

          dut.clock.step()

          if (col % 2 == 0) {
						// The response was accepted, so there isn't a pending response.
            dut.io.hostRdResp.valid.expect(false.B)
          } else {
            dut.io.hostRdResp.valid.expect(true.B)
            dut.io.hostRdResp.bits.expect(values(col).S)

            dut.io.hostRdResp.ready.poke(true.B)

            dut.clock.step()
						dut.io.hostRdResp.valid.expect(false.B)
          }
        }
      }
    }
  }

  it should "clear a pending host read response on reset" in {
    configs.foreach { c =>
      test(new UnifiedBuffer(c)) { dut =>
        clearInputs(dut)

        writeResultRow(dut, c, rowAddr = 0, Seq.fill(c.cols)(55))

        requestHostRead(dut, c, addr = 0)

        dut.io.hostRdResp.valid.expect(true.B)

        dut.reset.poke(true.B)
        dut.clock.step()
        dut.reset.poke(false.B)

        dut.io.hostRdResp.valid.expect(false.B)
        dut.io.hostRdReq.ready.expect(true.B)
      }
    }
  }

  it should "burst read input rows with consecutive requests" in {
    configs.foreach { c =>
      test(new UnifiedBuffer(c)) { dut =>
        clearInputs(dut)

        val numRows = 4
        val rowData = (0 until numRows).map { r =>
          (0 until c.banks).map(b => r * 10 + b + 1)
        }

        for (r <- 0 until numRows) {
          for (b <- 0 until c.banks) {
            writeInput(dut, c, b + (r << c.bankBits), rowData(r)(b))
          }
        }

        // Request row 0.
        dut.io.mvRdReq.bits.rowAddr.poke(0.U)
        dut.io.mvRdReq.valid.poke(true.B)
        dut.io.mvRdReq.ready.expect(true.B)
        dut.clock.step()

        // Request next row while reading previous response.
        for (r <- 1 until numRows) {
          dut.io.mvRdReq.bits.rowAddr.poke(r.U)
          dut.io.mvRdReq.valid.poke(true.B)
          dut.io.mvRdResp.valid.expect(true.B)
          for (b <- 0 until c.banks) {
            dut.io.mvRdResp.bits(b).expect(rowData(r - 1)(b).S)
          }
          dut.clock.step()
        }

        // Deassert request, read the last row response.
        dut.io.mvRdReq.valid.poke(false.B)
        dut.io.mvRdResp.valid.expect(true.B)
        for (b <- 0 until c.banks) {
          dut.io.mvRdResp.bits(b).expect(rowData(numRows - 1)(b).S)
        }
        dut.clock.step()
        dut.io.mvRdResp.valid.expect(false.B)
      }
    }
  }

  it should "reject new host read requests while previous response is held by backpressure" in {
    configs.foreach { c =>
      test(new UnifiedBuffer(c)) { dut =>
        clearInputs(dut)

        val valA = 42
        val valB = 99
        writeResultRow(dut, c, rowAddr = 0, Seq.fill(c.cols)(valA))
        writeResultRow(dut, c, rowAddr = 1, Seq.fill(c.cols)(valB))

        // Request address 0.
        dut.io.hostRdReq.bits.addr.poke(0.U)
        dut.io.hostRdReq.valid.poke(true.B)
        dut.io.hostRdReq.ready.expect(true.B)
        dut.clock.step()
        dut.io.hostRdReq.valid.poke(false.B)

        // Wait until response is valid.
        dut.clock.step()
        dut.io.hostRdResp.valid.expect(true.B)
        dut.io.hostRdResp.bits.expect(valA.S)

        // Backpressure response.
        dut.io.hostRdResp.ready.poke(false.B)

        // Now attempt a new read request to address row 1.
        dut.io.hostRdReq.bits.addr.poke((1 << c.colBits).U)
        dut.io.hostRdReq.valid.poke(true.B)

        // It must not be accepted while response is pending.
        dut.io.hostRdReq.ready.expect(false.B)
        for (_ <- 0 until 5) {
          dut.clock.step()
          dut.io.hostRdReq.ready.expect(false.B)
          dut.io.hostRdResp.valid.expect(true.B)
          dut.io.hostRdResp.bits.expect(valA.S)
        }

        // Deassert request valid, then accept the held response.
        dut.io.hostRdReq.valid.poke(false.B)
        dut.io.hostRdResp.ready.poke(true.B)
        dut.clock.step()

        // Response should be cleared and ready should become true.
        dut.io.hostRdResp.valid.expect(false.B)
        dut.io.hostRdReq.ready.expect(true.B)

        // Now address 1 can be read.
        requestHostRead(dut, c, 1 << c.colBits)
        dut.io.hostRdResp.valid.expect(true.B)
        dut.io.hostRdResp.bits.expect(valB.S)
      }
    }
  }

  it should "abort an in-flight host read on reset before response is valid" in {
    configs.foreach { c =>
      test(new UnifiedBuffer(c)) { dut =>
        clearInputs(dut)

        writeResultRow(dut, c, rowAddr = 0, Seq.fill(c.cols)(88))

        // Cycle 0: request.
        dut.io.hostRdReq.bits.addr.poke(0.U)
        dut.io.hostRdReq.valid.poke(true.B)
        dut.io.hostRdReq.ready.expect(true.B)
        dut.clock.step()
        dut.io.hostRdReq.valid.poke(false.B)

        // Cycle 1: request is in-flight in memory, assert reset.
        dut.reset.poke(true.B)
        dut.clock.step()
        dut.reset.poke(false.B)

        // Cycle 2: no response should appear and buffer is ready.
        dut.io.hostRdResp.valid.expect(false.B)
        dut.io.hostRdReq.ready.expect(true.B)

        dut.clock.step()
        dut.io.hostRdResp.valid.expect(false.B)
        dut.io.hostRdReq.ready.expect(true.B)
      }
    }
  }

  it should "write and read at maximum boundary addresses of input and result memory" in {
    configs.foreach { c =>
      test(new UnifiedBuffer(c)) { dut =>
        clearInputs(dut)

        val maxInRow = (c.inDepth / c.banks) - 1
        val inExpected = (0 until c.banks).map(b => b + 10)
        (0 until c.banks).foreach { b =>
          writeInput(dut, c, b + (maxInRow << c.bankBits), inExpected(b))
        }
        val inActual = readInputRow(dut, c, maxInRow)
        assert(inActual == inExpected)

        val maxOutRow = (c.outDepth / c.cols) - 1
        val outExpected = (0 until c.cols).map(col => col + 50)
        writeResultRow(dut, c, maxOutRow, outExpected)

        (0 until c.cols).foreach { col =>
          requestHostRead(dut, c, col + (maxOutRow << c.colBits))
          dut.io.hostRdResp.valid.expect(true.B)
          dut.io.hostRdResp.bits.expect(outExpected(col).S)
          dut.io.hostRdResp.ready.poke(true.B)
          dut.clock.step()
        }
      }
    }
  }
}
