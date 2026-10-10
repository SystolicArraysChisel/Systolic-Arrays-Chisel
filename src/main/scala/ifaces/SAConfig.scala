package ifaces

import chisel3._
import chisel3.util._

sealed trait Dataflow
case object WeightStationary extends Dataflow

sealed trait LoadMode
case object Space extends LoadMode

case class SAConfig(
    rows: Int = 4, // Number of PE rows
    cols: Int = 4, // Number of PE columns
    vecWidth: Int = 1, // Products each PE computes per cycle (V )
    inWidth: Int = 8, // Bit width of an input element of A and B.
    signed: Boolean = true,
    accWidth: Int = 32, // Bit width of an accumulator and of a raw result
    dataflow: Dataflow =
      WeightStationary, // What stays inside the PEs while the other data flows through.
    loadMode: LoadMode = Space, // How weights are loaded:
    // Space shifts one row in per cycle (few wires),
    // Speed loads all rows at once (many wires)
    inDepth: Int =
      4096, // Number of inT elements the input memory holds (A and B together)
    outDepth: Int =
      4096, // Number of accT words the result memory holds (C). Separate from the input memory.
    maxM: Int =
      256 /* Depth of the accumulator memory (the Accumulators block below the array), in rows of C.
       * Partial sums of every row of C are stored there between K-tiles, so one MATMUL
       * can have at most maxM rows (m <= maxM). The host splits a larger A by rows into
       * several MATMULs. Unrelated to the PEs, which keep no sum. */
) {
  // ---- derived values ----
  def inT: SInt = SInt(inWidth.W); // Chisel type of an input element
  def accT: SInt = SInt(accWidth.W); // Chisel type of a partial sum or result.

  val banks: Int =
    (rows max cols) * vecWidth; // Number of input-memory banks, i.e elements the data mover reads per cycle.

  val bankBits: Int = log2Ceil(
    banks
  ) // Bits of an element address that select the bank
  val addrW: Int =
    log2Ceil(inDepth); // Bits of an element address in the input memory
  val rowAddrW: Int = addrW - bankBits; // Bits of a row address inside a bank.

  val colBits: Int = log2Ceil(
    cols
  ) // Bits of an element address that selects the column of a PE
  val outAddrW: Int = log2Ceil(
    outDepth
  ) // Bits of an element address in the result memory.
  val outRowAddrW: Int = outAddrW - colBits; // Bits of a row address inside the result memory.

  val shiftW: Int = log2Ceil(accWidth) // Bits of the requantization shift
  val dimW: Int =
    16; // width, in bits, of the hardware registers and wires that hold matrix sizes
  val arrayLatency: Int =
    rows + cols - 1; /* Cycles between row m of aIn entering the array and row m of out leaving it, after skew and de-skew */
  /** Minimum number of empty cycles between the last weight beat (wIn) and the
    * first activation (aIn): the first aIn may be valid weightToActGap + 1
    * cycles after the last wIn beat. Depends on Andreea's PE design: TO
    * CONFIRM. 0 for now.
    */
  val weightToActGap: Int = 0
  // WARNING: Comes from Andreea’s schedule function may be subject to change
  val dspCount: Int = rows * cols * vecWidth; /* Number of multipliers, one DSP
  each. Used by the checks and the resource reports */

  // ---- checks ----
  // base parameters checks :
  require(
    rows >= 1 && cols >= 1,
    s"array size must be at least 1x1, got ${rows}x${cols}"
  )
  require(
    inWidth >= 1 && accWidth >= 1 && inDepth >= 1 && outDepth >= 1 && maxM >= 1,
    s"widths and depths must be positive (inWidth=$inWidth, accWidth=$accWidth, " +
      s"inDepth=$inDepth, maxM=$maxM)"
  )
  require(
    isPow2(inDepth) && isPow2(outDepth),
    s"inDepth ($inDepth) and outDepth ($outDepth) must be powers of two"
  )
  require(
    dataflow == WeightStationary,
    s"MVP: only WeightStationary is supported, got $dataflow"
  )
  require(
    loadMode == Space,
    s"MVP: only Space (shift-in) weight loading is supported, got $loadMode"
  )
  require(vecWidth == 1, s"MVP: vecWidth must be 1, got $vecWidth")
  require(
    signed,
    "MVP: only signed inputs are supported (inT and accT are SInt)"
  )
  require(rows == cols, s"MVP: the array must be square, got ${rows}x${cols}")
  require(isPow2(rows), s"rows (= banks) must be a power of two, got $rows")

  // derived parameters checks :
  require(
    inDepth % banks == 0,
    s"inDepth ($inDepth) must be a multiple of banks ($banks)"
  )
  require(
    addrW <= 16,
    s"inDepth = $inDepth needs $addrW address bits, but addresses are 2 bytes in the host protocol"
  )
  require(
    maxM < (1 << dimW),
    s"maxM = $maxM does not fit in the $dimW-bit m field"
  )
  require(
    accWidth >= 2 * inWidth + dimW,
    s"accWidth = $accWidth may overflow: a sum of up to 2^$dimW products of $inWidth-bit " +
      s"values needs ${2 * inWidth + dimW} bits"
  )
  require(
    outDepth % cols == 0,
    s"outDepth ($outDepth) must be a multiple of cols ($cols)"
  )
  require(
    outAddrW <= 16,
    s"outDepth = $outDepth needs $outAddrW address bits, but addresses are 2 bytes in the host protocol"
  )
  require(
    shiftW <= 8,
    s"shiftW = $shiftW does not fit in the 1-byte shift field of MATMUL"
  )
  require(
    dspCount <= 240,
    s"dspCount must be at most 240 for the MVP, got $dspCount"
  )
}
