package tpu.memory

import chisel3._
import chisel3.util._

import chisel3.experimental.{annotate, ChiselAnnotation}
import ifaces._

/**
  * Banked on-chip memory for matrices A and B, plus the result matrix C.
  */
class UnifiedBuffer(c: SAConfig) extends Module {
  val io = IO(new Bundle {
    // CommandDecoder <-> UnifiedBuffer interface
    val hostWr = Flipped(Decoupled(new BufWritePort(c)))
    val hostRdReq = Flipped(Decoupled(new BufReadReq(c)))
    val hostRdResp = Decoupled(c.accT)

    // DataMover <-> UnifiedBuffer interface: activation rows (port A,
    // stalled by writes) and weight rows (port B, never stalled)
    val mvRdReq = Flipped(Decoupled(new BufRowReq(c)))
    val mvRdResp = Decoupled(Vec(c.banks, c.inT))
    val mvWRdReq = Flipped(Decoupled(new BufRowReq(c)))
    val mvWRdResp = Decoupled(Vec(c.banks, c.inT))

    // Accumulator <-> UnifiedBuffer interface
    val resWr = Flipped(Decoupled(new BufResultWrite(c)))
  })



  /**
    * Splits an element address into a bank or column index and
    * a row address within that bank or column.
    *
    * The least-significant bits select the bank or column because
    * consecutive logical elements are stored in separate memories.
    */
  def splitAddr(addr: UInt, bits: Int): (UInt, UInt) = {
    if (bits > 0) {
      (addr(bits - 1, 0), addr(addr.getWidth - 1, bits))
    } else {
      (0.U, addr)
    }
  }



  /**
    * Input memory for A and B.
    *
    * The host writes one element at a time through hostWr. The data
    * mover reads up to two complete rows per cycle (one activation row,
    * one weight row), with one element from each bank.
    */
  val inputMem = Seq.fill(c.banks) {
    SyncReadMem(c.inDepth / c.banks, c.inT)
  }

  /**
    * Input-memory ports: each bank has exactly two, so Vivado maps it to
    * one true dual-port BRAM.
    *
    * Port A (read/write), by priority:
    *  1. a whole row of C written back by the accumulators (toInput): one
    *     element per bank at the same row address (cols == banks, and
    *     cAddr is a multiple of banks);
    *  2. one host-provided element;
    *  3. an activation row for the data mover.
    * Port B (read only): a weight row for the data mover.
    *
    * A write stalls activation reads for that cycle (mvRdReq.ready drops);
    * the write-back also holds the host for that cycle (hostWr.ready drops).
    */
  require(c.cols == c.banks, "write-back to the input memory needs one bank per column")

  val resToInput = io.resWr.valid && io.resWr.bits.toInput
  val resInRow = io.resWr.bits.rowAddr(c.rowAddrW - 1, 0)

  io.hostWr.ready := !resToInput
  io.mvRdReq.ready := !resToInput && !io.hostWr.valid
  io.mvWRdReq.ready := true.B

  val (hostWrBank, hostWrRow) = splitAddr(io.hostWr.bits.addr, c.bankBits)

  val mvRdRespVec = Wire(Vec(c.banks, c.inT))
  val mvWRdRespVec = Wire(Vec(c.banks, c.inT))
  for (b <- 0 until c.banks) {
    val hostHere = io.hostWr.fire && hostWrBank === b.U
    val isWrite = resToInput || hostHere
    val addrA = Mux(resToInput, resInRow, Mux(hostHere, hostWrRow, io.mvRdReq.bits.rowAddr))
    // Requantized values fit in inWidth bits: keep only those.
    val dataA = Mux(resToInput, io.resWr.bits.data(b)(c.inWidth - 1, 0).asSInt, io.hostWr.bits.data)
    mvRdRespVec(b) := inputMem(b).readWrite(addrA, dataA, isWrite || io.mvRdReq.fire, isWrite)
    mvWRdRespVec(b) := inputMem(b).read(io.mvWRdReq.bits.rowAddr, io.mvWRdReq.valid)
  }

  // Both responses are valid one cycle after their request.
  io.mvRdResp.bits := mvRdRespVec
  io.mvRdResp.valid := RegNext(io.mvRdReq.fire, false.B)
  io.mvWRdResp.bits := mvWRdRespVec
  io.mvWRdResp.valid := RegNext(io.mvWRdReq.valid, false.B)

  // No backpressure on DataMover: it must always be ready to accept the responses.
  assert(!io.mvRdResp.valid || io.mvRdResp.ready,
    "UnifiedBuffer: mvRdResp not accepted, DataMover must always be ready")
  assert(!io.mvWRdResp.valid || io.mvWRdResp.ready,
    "UnifiedBuffer: mvWRdResp not accepted, DataMover must always be ready")



  /**
    * Result memory for C.
    *
    * Each column has a separate memory. The accumulators write one
    * complete row per cycle, while the host reads one element at a time.
    */
  val resultMem = Seq.fill(c.cols) {
    val m = SyncReadMem(c.outDepth / c.cols, c.accT)
    annotate(new ChiselAnnotation {
      def toFirrtl = firrtl.AttributeAnnotation(m.toTarget, "ram_style = \"block\"")
    })
    m
  }

  /**
    * Writes one complete result row, unless it goes to the input memory.
    *
    * Each element of resWr.bits.data is written to the memory
    * corresponding to its column. resWr is always ready: both memories
    * take a full row in any cycle.
    */
  io.resWr.ready := true.B

  when (io.resWr.fire && !io.resWr.bits.toInput) {
    for (col <- 0 until c.cols) {
      resultMem(col).write(
        io.resWr.bits.rowAddr(c.outRowAddrW - 1, 0),
        io.resWr.bits.data(col))
    }
  }

  /**
    * Reads one result element for the host.
    *
    * The request address is split into a column and a row. The selected
    * column is read from its SyncReadMem, and the response is buffered
    * until the host accepts it.
    */
  val (hostRdCol, hostRdRow) = splitAddr(io.hostRdReq.bits.addr, c.colBits)

  // Delay the request and column to align them with the synchronous read.
  val rdFireD = RegNext(io.hostRdReq.fire, false.B)
  val rdColD = RegEnable(hostRdCol, io.hostRdReq.fire)
  val respValid = RegInit(false.B)

  val hostRdRespVec = Wire(Vec(c.cols, c.accT))

  for (col <- 0 until c.cols) {
    hostRdRespVec(col) := resultMem(col).read(
      hostRdRow, 
      io.hostRdReq.fire)
  }

  // Register each column's memory output individually before multiplexing.
  // This allows Vivado to infer a dedicated Block RAM (RAMB36) with its output
  // register for each column, avoiding distributed LUTRAM.
  val respDataVec = Reg(Vec(c.cols, c.accT))

  when (rdFireD) {
    respDataVec := hostRdRespVec
    respValid := true.B
  }
  when (io.hostRdResp.fire) {
    respValid := false.B
  }

  // Select the response data from the registered column outputs.
  val respData = if (c.cols == 1) {
    respDataVec(0)
  } else {
    respDataVec(rdColD)
  }

  // Allow at most one outstanding host read.
  io.hostRdReq.ready := !rdFireD && !respValid
  io.hostRdResp.bits := respData
  io.hostRdResp.valid := respValid
}
