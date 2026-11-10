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
    * Writes into the input memory: a whole row of C written back by the
    * accumulators (toInput), or one host-provided element.
    *
    * A written-back row is one element per bank at the same row address
    * (cols == banks, and cAddr is a multiple of banks), so it uses every
    * bank's write port in that cycle. It wins: hostWr.ready drops for that
    * cycle and the decoder holds its element until the next one.
    */
  require(c.cols == c.banks, "write-back to the input memory needs one bank per column")

  val resToInput = io.resWr.valid && io.resWr.bits.toInput
  val resInRow = io.resWr.bits.rowAddr(c.rowAddrW - 1, 0)

  io.hostWr.ready := !resToInput

  val (hostWrBank, hostWrRow) = splitAddr(io.hostWr.bits.addr, c.bankBits)

  when (resToInput) {
    for (b <- 0 until c.banks) {
      // Requantized values fit in inWidth bits: keep only those.
      inputMem(b).write(
        resInRow,
        io.resWr.bits.data(b)(c.inWidth - 1, 0).asSInt)
    }
  } .elsewhen (io.hostWr.fire) {
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

  // No backpressure on DataMover: it must always be ready to accept the response.
  assert(!io.mvRdResp.valid || io.mvRdResp.ready, 
    "UnifiedBuffer: mvRdResp not accepted, DataMover must always be ready")



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
