package tpu.io

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class RxFifoSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "RxFifo"

  val b = BoardConfig() // rxFifoDepth = 16; the FIFO does not depend on the baud rate
  val depth = b.rxFifoDepth

  def init(dut: RxFifo): Unit = {
    dut.io.enq.valid.poke(false.B)
    dut.io.deq.ready.poke(false.B)
  }

  /** Offer one byte for one cycle, as UartRx does (it ignores ready). */
  def push(dut: RxFifo, byte: Int): Unit = {
    dut.io.enq.valid.poke(true.B)
    dut.io.enq.bits.poke(byte.U)
    dut.clock.step()
    dut.io.enq.valid.poke(false.B)
  }

  /** Take the oldest byte. */
  def pop(dut: RxFifo): Int = {
    dut.io.deq.valid.expect(true.B)
    val byte = dut.io.deq.bits.peekInt().toInt
    dut.io.deq.ready.poke(true.B)
    dut.clock.step()
    dut.io.deq.ready.poke(false.B)
    byte
  }

  it should "be empty after reset" in {
    test(new RxFifo(b)) { dut =>
      init(dut)
      dut.io.deq.valid.expect(false.B)
      dut.io.count.expect(0.U)
      dut.io.overrun.expect(false.B)
    }
  }

  it should "return bytes in arrival order and count them" in {
    test(new RxFifo(b)) { dut =>
      init(dut)
      val bytes = Seq(0x11, 0x22, 0x33, 0x44, 0x55)
      for ((byte, i) <- bytes.zipWithIndex) {
        dut.io.count.expect(i.U)
        push(dut, byte)
      }
      dut.io.count.expect(bytes.length.U)
      assert(bytes.map(_ => pop(dut)) == bytes)
      dut.io.count.expect(0.U)
      dut.io.deq.valid.expect(false.B)
    }
  }

  it should "hold the oldest byte on deq while the decoder waits" in {
    test(new RxFifo(b)) { dut =>
      init(dut)
      push(dut, 0xab)
      push(dut, 0xcd)
      for (_ <- 0 until 5) {
        dut.io.deq.valid.expect(true.B)
        dut.io.deq.bits.expect(0xab.U)
        dut.clock.step()
      }
      assert(pop(dut) == 0xab)
      assert(pop(dut) == 0xcd)
    }
  }

  it should "pulse overrun and drop the byte when full" in {
    test(new RxFifo(b)) { dut =>
      init(dut)
      for (i <- 0 until depth) {
        dut.io.overrun.expect(false.B)
        push(dut, i)
      }
      dut.io.count.expect(depth.U)

      dut.io.enq.valid.poke(true.B)
      dut.io.enq.bits.poke(0xee.U)
      dut.io.overrun.expect(true.B) // combinational: in the same cycle as the lost byte
      dut.clock.step()
      dut.io.enq.valid.poke(false.B)
      dut.io.overrun.expect(false.B) // a pulse, not a level

      dut.io.count.expect(depth.U)
      assert((0 until depth).map(_ => pop(dut)) == (0 until depth)) // 0xee is not there
      dut.io.deq.valid.expect(false.B)
    }
  }

  it should "accept a byte when full if one leaves in the same cycle" in {
    test(new RxFifo(b)) { dut =>
      init(dut)
      for (i <- 0 until depth) push(dut, i)
      dut.io.enq.valid.poke(true.B)
      dut.io.enq.bits.poke(0xee.U)
      dut.io.deq.ready.poke(true.B)
      dut.io.overrun.expect(false.B)
      dut.io.deq.bits.expect(0.U)
      dut.clock.step()
      init(dut)
      dut.io.count.expect(depth.U)
      assert((0 until depth).map(_ => pop(dut)) == (1 until depth) :+ 0xee)
    }
  }

  it should "keep its size when a byte enters and one leaves every cycle" in {
    test(new RxFifo(b)) { dut =>
      init(dut)
      push(dut, 0)
      push(dut, 1)
      dut.io.enq.valid.poke(true.B)
      dut.io.deq.ready.poke(true.B)
      for (i <- 2 until 40) {
        dut.io.enq.bits.poke(i.U)
        dut.io.deq.bits.expect((i - 2).U)
        dut.io.overrun.expect(false.B)
        dut.clock.step()
        dut.io.count.expect(2.U)
      }
    }
  }

  it should "be emptied by reset" in {
    test(new RxFifo(b)) { dut =>
      init(dut)
      for (i <- 0 until 5) push(dut, i)
      dut.reset.poke(true.B)
      dut.clock.step()
      dut.reset.poke(false.B)
      dut.io.count.expect(0.U)
      dut.io.deq.valid.expect(false.B)
      push(dut, 0x42)
      assert(pop(dut) == 0x42)
    }
  }
}
