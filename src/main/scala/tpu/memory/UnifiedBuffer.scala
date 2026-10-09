package tpu.memory

import chisel3._
import chisel3.util._

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

    // DataMover <-> UnifiedBuffer interface
    val mvRdReq = Flipped(Decoupled(new BufRowReq(c)))
    val mvRdResp = Decoupled(Vec(c.banks, c.inT))

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
    * mover reads one complete row per cycle, with one element from each
    * bank.
    */
  val inputMem = Seq.fill(c.banks) {
    SyncReadMem(c.inDepth / c.banks, c.inT)
  }

  /**
    * Writes one host-provided element to the selected input-memory bank.
    *
    * hostWr.ready is permanently asserted because the input memory
    * accepts one write request in every cycle.
    */
  io.hostWr.ready := true.B

  val (hostWrBank, hostWrRow) = splitAddr(io.hostWr.bits.addr, c.bankBits)

  when (io.hostWr.fire) {
    for (b <- 0 until c.banks) {
      when (hostWrBank === b.U) {
        inputMem(b).write(
          hostWrRow, 
          io.hostWr.bits.data)
      }
    }
  }

  /**
    * Reads one complete input-memory row for the data mover.
    *
    * All banks use the same row address, so the result contains one
    * element from each bank. SyncReadMem has one cycle of read latency.
    */
  io.mvRdReq.ready := true.B

  val mvRdRespVec = Wire(Vec(c.banks, c.inT))
  for (b <- 0 until c.banks) {
    mvRdRespVec(b) := inputMem(b).read(
      io.mvRdReq.bits.rowAddr, 
      io.mvRdReq.valid)
  }
  io.mvRdResp.bits := mvRdRespVec
  // The response is valid one cycle after the request.
  io.mvRdResp.valid := RegNext(io.mvRdReq.valid, false.B)

  assert(!io.mvRdResp.valid || io.mvRdResp.ready, 
    "UnifiedBuffer: mvRdResp not accepted, DataMover must always be ready")



  /**
    * Result memory for C.
    *
    * Each column has a separate memory. The accumulators write one
    * complete row per cycle, while the host reads one element at a time.
    */
  val resultMem = Seq.fill(c.cols) {
    SyncReadMem(c.outDepth / c.cols, c.accT)
  }

  /**
    * Writes one complete result row.
    *
    * Each element of resWr.bits.data is written to the memory
    * corresponding to its column.
    */
  io.resWr.ready := true.B

  when (io.resWr.fire) {
    for (col <- 0 until c.cols) {
      resultMem(col).write(
        io.resWr.bits.rowAddr, 
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
  val respData = Reg(c.accT)

  val hostRdRespVec = Wire(Vec(c.cols, c.accT))

  for (col <- 0 until c.cols) {
    hostRdRespVec(col) := resultMem(col).read(
      hostRdRow, 
      io.hostRdReq.fire && (hostRdCol === col.U))
  }

  // Select the memory output corresponding to the delayed column.
  val rdRaw = if (c.cols == 1) {
    hostRdRespVec(0)
  } else {
    hostRdRespVec(rdColD)
  }

  // Capture the memory output and hold it until the response is accepted.
  when (rdFireD) {
    respData := rdRaw
    respValid := true.B
  }
  when (io.hostRdResp.fire) {
    respValid := false.B
  }

  // Allow at most one outstanding host read.
  io.hostRdReq.ready := !rdFireD && !respValid
  io.hostRdResp.bits := respData
  io.hostRdResp.valid := respValid
}
