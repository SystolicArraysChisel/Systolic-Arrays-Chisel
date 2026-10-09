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
    dut.io.mvRdResp.ready.poke(true.B)
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

        dut.io.hostRdResp.ready.poke(false.B)
        dut.io.hostRdReq.ready.expect(false.B)
        dut.io.hostRdResp.valid.expect(true.B)
        val heldBits = dut.io.hostRdResp.bits.peek().litValue

        for (_ <- 0 until 10) {
          dut.io.hostRdResp.valid.expect(true.B)
          dut.io.hostRdResp.bits.expect(heldBits.S)
          dut.io.hostRdReq.ready.expect(false.B)
          dut.clock.step()
        }

        dut.io.hostRdResp.ready.poke(true.B)
        dut.io.hostRdResp.valid.expect(true.B)
        dut.io.hostRdResp.bits.expect(expected.S)
        dut.clock.step()
        dut.io.hostRdResp.valid.expect(false.B)
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
          dut.io.hostRdResp.ready.poke(col % 2 == 0)
          dut.clock.step()

          if (col % 2 == 0) {
            dut.io.hostRdResp.valid.expect(false.B)
          } else {
            dut.io.hostRdResp.valid.expect(true.B)
            dut.io.hostRdResp.bits.expect(values(col).S)
            dut.io.hostRdResp.ready.poke(true.B)
            dut.clock.step()
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
}
