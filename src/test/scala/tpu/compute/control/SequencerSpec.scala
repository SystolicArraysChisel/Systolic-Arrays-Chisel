package tpu.compute.control

import chisel3._
import chiseltest._
import ifaces._
import org.scalatest.flatspec.AnyFlatSpec

class SequencerSpec extends AnyFlatSpec with ChiselScalatestTester {

  behavior of "Sequencer"

  private val config = SAConfig(
    rows = 2,
    cols = 2,
    inDepth = 64,
    outDepth = 64,
    maxM = 4
  )

  private def initialize(dut: Sequencer): Unit = {
    dut.io.instr.valid.poke(false.B)
    dut.io.accCtrl.ready.poke(false.B)
    dut.io.tileCmd.ready.poke(false.B)
    dut.io.tileDone.valid.poke(false.B)

    dut.reset.poke(true.B)
    dut.clock.step()
    dut.reset.poke(false.B)

    dut.io.instr.ready.expect(true.B)
    dut.io.status.busy.expect(false.B)
  }

  private def pokeInstruction(
      dut: Sequencer,
      aAddr: Int,
      bAddr: Int,
      cAddr: Int,
      m: Int,
      k: Int,
      n: Int,
      ldA: Int,
      ldB: Int,
      ldC: Int,
      requant: Boolean = false,
      relu: Boolean = false,
      shift: Int = 0
  ): Unit = {
    dut.io.instr.bits.op.poke(Opcode.Matmul)
    dut.io.instr.bits.aAddr.poke(aAddr.U)
    dut.io.instr.bits.bAddr.poke(bAddr.U)
    dut.io.instr.bits.cAddr.poke(cAddr.U)
    dut.io.instr.bits.m.poke(m.U)
    dut.io.instr.bits.k.poke(k.U)
    dut.io.instr.bits.n.poke(n.U)
    dut.io.instr.bits.ldA.poke(ldA.U)
    dut.io.instr.bits.ldB.poke(ldB.U)
    dut.io.instr.bits.ldC.poke(ldC.U)
    dut.io.instr.bits.requant.poke(requant.B)
    dut.io.instr.bits.relu.poke(relu.B)
    dut.io.instr.bits.shift.poke(shift.U)
  }

  private def acceptInstruction(dut: Sequencer): Unit = {
    dut.io.instr.valid.poke(true.B)
    dut.io.instr.ready.expect(true.B)
    dut.clock.step()
    dut.io.instr.valid.poke(false.B)
    dut.io.status.busy.expect(true.B)
  }

  private def expectTile(
      dut: Sequencer,
      aAddr: Int,
      bAddr: Int,
      m: Int,
      cAddr: Int,
      ldC: Int,
      first: Boolean,
      last: Boolean,
      acceptAccumulator: Boolean = true,
      acceptMover: Boolean = true
  ): Unit = {
    dut.io.accCtrl.valid.expect(true.B)
    dut.io.tileCmd.valid.expect(true.B)

    dut.io.accCtrl.bits.first.expect(first.B)
    dut.io.accCtrl.bits.last.expect(last.B)
    dut.io.accCtrl.bits.cAddr.expect(cAddr.U)
    dut.io.accCtrl.bits.ldC.expect(ldC.U)
    dut.io.accCtrl.bits.m.expect(m.U)

    dut.io.tileCmd.bits.aAddr.expect(aAddr.U)
    dut.io.tileCmd.bits.bAddr.expect(bAddr.U)
    dut.io.tileCmd.bits.m.expect(m.U)

    dut.io.accCtrl.ready.poke(acceptAccumulator.B)
    dut.io.tileCmd.ready.poke(acceptMover.B)
    dut.clock.step()
    dut.io.accCtrl.ready.poke(false.B)
    dut.io.tileCmd.ready.poke(false.B)
  }

  private def completeTile(dut: Sequencer): Unit = {
    dut.io.tileDone.valid.poke(true.B)
    dut.io.tileDone.bits.poke(true.B)
    dut.clock.step()
    dut.io.tileDone.valid.poke(false.B)
  }

  it should "remain idle until a matrix multiplication is accepted" in {
    test(new Sequencer(config)) { dut =>
      initialize(dut)

      dut.io.accCtrl.valid.expect(false.B)
      dut.io.tileCmd.valid.expect(false.B)
      dut.io.instr.ready.expect(true.B)
    }
  }

  it should "hold each command until its independent downstream handshake" in {
    test(new Sequencer(config)) { dut =>
      initialize(dut)
      pokeInstruction(
        dut,
        aAddr = 4,
        bAddr = 8,
        cAddr = 12,
        m = 2,
        k = 2,
        n = 2,
        ldA = 2,
        ldB = 2,
        ldC = 2
      )
      acceptInstruction(dut)

      // Only the accumulator accepts the first command.
      expectTile(
        dut,
        aAddr = 4,
        bAddr = 8,
        m = 2,
        cAddr = 12,
        ldC = 2,
        first = true,
        last = true,
        acceptMover = false
      )
      dut.io.accCtrl.valid.expect(false.B)
      dut.io.tileCmd.valid.expect(true.B)
      dut.io.tileCmd.bits.aAddr.expect(4.U)
      dut.io.tileCmd.bits.bAddr.expect(8.U)

      // The data-mover command is accepted later, without being lost.
      dut.io.tileCmd.ready.poke(true.B)
      dut.clock.step()
      dut.io.tileCmd.ready.poke(false.B)
      dut.io.accCtrl.valid.expect(false.B)
      dut.io.tileCmd.valid.expect(false.B)

      completeTile(dut)
      dut.io.status.busy.expect(false.B)
    }
  }

  it should "issue K tiles before advancing to the next N tile" in {
    test(new Sequencer(config)) { dut =>
      initialize(dut)
      pokeInstruction(
        dut,
        aAddr = 16,
        bAddr = 32,
        cAddr = 8,
        m = 3,
        k = 4,
        n = 4,
        ldA = 4,
        ldB = 6,
        ldC = 8,
        requant = true,
        relu = true,
        shift = 3
      )
      acceptInstruction(dut)

      // First K tile of the first N tile.
      expectTile(
        dut,
        aAddr = 16,
        bAddr = 32,
        m = 3,
        cAddr = 8,
        ldC = 8,
        first = true,
        last = false
      )
      completeTile(dut)
      dut.io.status.busyPeCycles.expect(12.U)

      // Second K tile accumulates into the same output tile.
      expectTile(
        dut,
        aAddr = 18,
        bAddr = 44,
        m = 3,
        cAddr = 8,
        ldC = 8,
        first = false,
        last = true
      )
      completeTile(dut)
      dut.io.status.busyPeCycles.expect(24.U)

      // K resets before the sequencer advances to the next N tile.
      expectTile(
        dut,
        aAddr = 16,
        bAddr = 34,
        m = 3,
        cAddr = 10,
        ldC = 8,
        first = true,
        last = false
      )
      completeTile(dut)
      dut.io.status.busyPeCycles.expect(36.U)

      // Final K tile of the final N tile completes the instruction.
      expectTile(
        dut,
        aAddr = 18,
        bAddr = 46,
        m = 3,
        cAddr = 10,
        ldC = 8,
        first = false,
        last = true
      )
      completeTile(dut)

      dut.io.status.busy.expect(false.B)
      dut.io.instr.ready.expect(true.B)
      dut.io.status.busyPeCycles.expect(48.U)
    }
  }
}
