package tpu.io

import chisel3._
import chisel3.util._

import ifaces._
import tpu.io.HostProtocol._

object DecoderState extends ChiselEnum {
  val sOpcode, sHeader, sExec, sWriteData, sReadReq, sReadResp, sSend, sInstr =
    Value
}

/** Parses the byte stream from the host into commands (protocol in
  * HostProtocol). WRITE and READ are executed here, against the buffer; MATMUL
  * is handed to the sequencer; STATUS is answered from the sequencer's counters
  * and the sticky error/overrun flags.
  */
class CommandDecoder(c: SAConfig, b: BoardConfig) extends Module {
  val io = IO(new Bundle {
    // Host side
    val rx = Flipped(Decoupled(UInt(8.W))) // bytes from RxFifo
    val tx = Decoupled(UInt(8.W)) // bytes to UartTx
    val rxOverrun = Input(Bool()) // pulse from RxFifo: a byte was lost
    val error =
      Output(Bool()) // level: an error happened since reset (for the LEDs)

    // Unified Buffer side
    val bufWr =
      Decoupled(new BufWritePort(c)) // WRITE: one element per data byte
    val bufRdReq = Decoupled(new BufReadReq(c)) // READ: one request per element
    val bufRdResp = Flipped(Decoupled(c.accT)) // READ: the requested element

    // Sequencer side
    val instr = Decoupled(new Instruction(c)) // MATMUL
    val status = Input(new SeqStatus) // busy and counters, for STATUS
  })

  val maxHeader = Matmul.payloadBytes // longest header: the MATMUL payload
  val replyBits = 8 * Status.replyBytes // longest reply held at once: STATUS

  val state = RegInit(DecoderState.sOpcode)

  /** Opcode of the frame being received. */
  val op = RegInit(0.U(8.W))

  /** Bytes after the opcode (the header), stored as they arrive. MATMUL keeps
    * them until the sequencer accepts the instruction.
    */
  val hdr = Reg(Vec(maxHeader, UInt(8.W)))
  val hdrCnt = RegInit(
    0.U(log2Ceil(maxHeader + 1).W)
  ) // header bytes received so far
  val hdrLen = RegInit(0.U(log2Ceil(maxHeader + 1).W)) // header bytes expected

  /** Little-endian 2-byte field at payload offset `off` (frame offset - 1). */
  def field16(off: Int): UInt = Cat(hdr(off + 1), hdr(off))
  def field8(off: Int): UInt = hdr(off)

  /** WRITE / READ progress: next element address and elements left. */
  val addr = RegInit(0.U(16.W))
  val remaining = RegInit(0.U(16.W))
  val readSize = RegInit(
    0.U(3.W)
  ) // bytes per element of the current READ (1 or 4)

  /** Reply being sent, low byte first, and the number of bytes left in it. */
  val txShift = RegInit(0.U(replyBits.W))
  val txLeft = RegInit(0.U(log2Ceil(Status.replyBytes + 1).W))

  /** Cycles since the last byte of the current frame. */
  val timer = RegInit(0.U(b.timeoutW.W))

  // defaults
  io.rx.ready := false.B
  io.tx.valid := false.B
  io.tx.bits := txShift(7, 0)
  io.bufWr.valid := false.B
  io.bufWr.bits.addr := addr(c.addrW - 1, 0)
  io.bufWr.bits.data := io.rx.bits.asSInt
  io.bufRdReq.valid := false.B
  io.bufRdReq.bits.addr := addr(c.outAddrW - 1, 0)
  io.bufRdResp.ready := false.B
  io.instr.valid := false.B

  // error and overrun flags
  val errEvent = WireDefault(
    false.B
  ) // pulse: unknown opcode, invalid frame or timeout
  val capture = WireDefault(
    false.B
  ) // pulse: a STATUS reply is captured this cycle

  // Sticky between STATUS replies; a new event wins over the clear.
  val errFlag = RegInit(false.B)
  val ovrFlag = RegInit(false.B)
  errFlag := (errFlag && !capture) || errEvent
  ovrFlag := (ovrFlag && !capture) || io.rxOverrun

  // LED copy stays set until reset.
  val errSeen = RegInit(false.B)
  errSeen := errSeen || errEvent
  io.error := errSeen

  // frame timeout only while waiting for a byte of a started frame
  val inFrame =
    state === DecoderState.sHeader || state === DecoderState.sWriteData
  val timedOut = inFrame && !io.rx.valid && timer === b.frameTimeout.U
  when(!inFrame || io.rx.fire) {
    timer := 0.U
  }.elsewhen(!timedOut) {
    timer := timer + 1.U
  }
  when(timedOut) {
    errEvent := true.B // the partial frame is dropped
    state := DecoderState.sOpcode
  }

  // MATMUL fields, read straight from the header
  val mmFlags = field8(Matmul.FlagsOffset - 1)
  val mmShift = field8(Matmul.ShiftOffset - 1)
  io.instr.bits.op := Opcode.Matmul
  io.instr.bits.aAddr := field16(Matmul.AAddrOffset - 1)(c.addrW - 1, 0)
  io.instr.bits.bAddr := field16(Matmul.BAddrOffset - 1)(c.addrW - 1, 0)
  io.instr.bits.cAddr := field16(Matmul.CAddrOffset - 1)(c.outAddrW - 1, 0)
  io.instr.bits.m := field16(Matmul.MOffset - 1)
  io.instr.bits.k := field16(Matmul.KOffset - 1)
  io.instr.bits.n := field16(Matmul.NOffset - 1)
  io.instr.bits.ldA := field16(Matmul.LdAOffset - 1)
  io.instr.bits.ldB := field16(Matmul.LdBOffset - 1)
  io.instr.bits.ldC := field16(Matmul.LdCOffset - 1)
  io.instr.bits.requant := mmFlags(Matmul.RequantBit)
  io.instr.bits.relu := mmFlags(Matmul.ReluBit)
  io.instr.bits.shift := mmShift(c.shiftW - 1, 0)

  switch(state) {
    is(DecoderState.sOpcode) {
      io.rx.ready := true.B
      when(io.rx.fire) {
        op := io.rx.bits
        hdrCnt := 0.U
        switch(io.rx.bits) {
          is(Op.Write.U) {
            hdrLen := Write.headerBytes.U
            state := DecoderState.sHeader
          }
          is(Op.Read.U) {
            hdrLen := Read.headerBytes.U
            state := DecoderState.sHeader
          }
          is(Op.Matmul.U) {
            hdrLen := Matmul.payloadBytes.U
            state := DecoderState.sHeader
          }
          is(Op.Status.U) {
            // Snapshot of the 9 reply bytes, flags in the low byte.
            capture := true.B
            val flags = (io.status.busy.asUInt << Status.BusyBit) |
              (errFlag.asUInt << Status.ErrorBit) |
              (ovrFlag.asUInt << Status.OverrunBit)
            txShift := Cat(
              io.status.busyPeCycles,
              io.status.cycles,
              flags.pad(8)
            )
            txLeft := Status.replyBytes.U
            state := DecoderState.sSend
          }
        }
        when(!Op.all.map(io.rx.bits === _.U).reduce(_ || _)) {
          errEvent := true.B // unknown opcode are ignored
        }
      }
    }

    is(DecoderState.sHeader) {
      io.rx.ready := true.B
      when(io.rx.fire) {
        hdr(hdrCnt) := io.rx.bits
        hdrCnt := hdrCnt + 1.U
        when(hdrCnt === hdrLen - 1.U) {
          state := DecoderState.sExec
        }
      }
    }

    is(DecoderState.sExec) { // header complete: check it and start the command
      val count = field16(2) // WRITE and READ: count follows addr
      addr := field16(0)
      remaining := count
      state := DecoderState.sOpcode
      switch(op) {
        is(Op.Write.U) {
          when(count =/= 0.U) { state := DecoderState.sWriteData }
        }
        is(Op.Read.U) {
          val size = field8(4)
          readSize := size
          when(size =/= 1.U && size =/= 4.U) {
            errEvent := true.B
          }.elsewhen(count =/= 0.U) {
            state := DecoderState.sReadReq
          }
        }
        is(Op.Matmul.U) {
          when(mmShift >= c.accWidth.U || mmFlags(7, 2) =/= 0.U) {
            errEvent := true.B
          }.otherwise {
            state := DecoderState.sInstr
          }
        }
      }
    }

    is(DecoderState.sWriteData) { // each data byte goes straight to the buffer
      io.bufWr.valid := io.rx.valid
      io.rx.ready := io.bufWr.ready
      when(io.rx.fire) {
        addr := addr + 1.U
        remaining := remaining - 1.U
        when(remaining === 1.U) { state := DecoderState.sOpcode }
      }
    }

    is(DecoderState.sReadReq) {
      io.bufRdReq.valid := true.B
      when(io.bufRdReq.fire) { state := DecoderState.sReadResp }
    }

    is(DecoderState.sReadResp) {
      io.bufRdResp.ready := true.B
      when(io.bufRdResp.fire) {
        txShift := io.bufRdResp.bits.asUInt(31, 0)
        txLeft := readSize
        addr := addr + 1.U
        remaining := remaining - 1.U
        state := DecoderState.sSend
      }
    }

    is(DecoderState.sSend) { // STATUS reply, or one element of a READ
      io.tx.valid := true.B
      when(io.tx.fire) {
        txShift := txShift >> 8
        txLeft := txLeft - 1.U
        when(txLeft === 1.U) {
          state := Mux(
            op === Op.Read.U && remaining =/= 0.U,
            DecoderState.sReadReq,
            DecoderState.sOpcode
          )
        }
      }
    }

    is(
      DecoderState.sInstr
    ) { // no new byte is read until the sequencer takes the MATMUL
      io.instr.valid := true.B
      when(io.instr.fire) { state := DecoderState.sOpcode }
    }
  }
}
