package ifaces

import chisel3._

/** Shared bundles between the host side (CommandDecoder) and Giovanni's blocks
  * (UnifiedBuffer, Sequencer), plus the debug flags sent to the board top.
  */

object Opcode extends ChiselEnum {
  val Matmul = Value
}

/** One element written by the decoder into the INPUT memory (when executing the
  * WRITE command).
  *
  * Used as Decoupled(new BufWritePort(c)) from the decoder to the buffer.
  */
class BufWritePort(c: SAConfig) extends Bundle {

  val addr = UInt(c.addrW.W) // Element address in the input memory

  val data = c.inT // The element itself (int8).
}

/** Request for one element of the RESULT memory (when executing READ command).
  *
  * Used as Decoupled(new BufReadReq(c));
  *
  * the answer comes back as Decoupled(c.accT).
  */
class BufReadReq(c: SAConfig) extends Bundle {
  val addr = UInt(c.outAddrW.W) // Element address in the result memory
}

/** One MATMUL instruction, from the decoder to the sequencer: C = A · B with A
  * m×k, B k×n.
  *
  * Used as Decoupled(new Instruction(c))
  *
  * its acceptance starts the operation.
  */
class Instruction(c: SAConfig) extends Bundle {

  val op = Opcode() // Operation to execute

  val aAddr = UInt(c.addrW.W) // Input-memory address of A[0][0]
  val bAddr = UInt(c.addrW.W) // Input-memory address of B[0][0]
  val cAddr = UInt(
    c.outAddrW.W
  ) // Result-memory address where C[0][0] is written

  val m = UInt(c.dimW.W) // Rows of A and C (at most maxM)
  val k = UInt(
    c.dimW.W
  ) // Columns of A = rows of B (padded to a multiple of banks).
  val n = UInt(c.dimW.W) // Columns of B and C (padded to a multiple of banks).

  val ldA = UInt(c.dimW.W) // Distance between two rows of A in the input memory
  val ldB = UInt(c.dimW.W) // Distance between two rows of B in the input memory
  val ldC = UInt(
    c.dimW.W
  ) // Distance between two rows of C in the result memory

  val requant = Bool() // Requantize results to int8 before writing them

  val relu = Bool() // Apply ReLU after requantization.

  val shift = UInt(
    c.shiftW.W
  ) // Right shift used by requantization; ignored if requant is false.
}

/** State and performance counters of the sequencer, read by the decoder on
  * STATUS. Plain wires (no handshake): always meaningful.
  */
class SeqStatus extends Bundle {
  val busy = Bool() // a MATMUL is running
  val cycles = UInt(
    SeqStatus.counterW.W
  ) // Cycles of the last MATMUL, from its acceptance until busy dropped; the count so far while busy.
  val busyPeCycles = UInt(
    SeqStatus.counterW.W
  ) // PE-cycles spent on useful products in the last MATMUL
}

object SeqStatus {
  val counterW = 32 // Width of the performance counters
}

/** Flags shown on the board LEDs. Unlike the STATUS flags, overrun and error
  * stay set until reset.
  */
class DebugInfo extends Bundle {

  val busy = Bool() // A MATMUL is running.
  val overrun = Bool() // RxFifo has lost at least one byte since reset
  val error =
    Bool() // An unknown opcode or a timed-out frame has been received since reset.
}
