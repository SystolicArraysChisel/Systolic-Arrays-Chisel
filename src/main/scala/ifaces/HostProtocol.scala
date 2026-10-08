package ifaces

/** Byte-level protocol between the Python host and the CommandDecoder.
  * Multi-byte fields are little-endian.
  */
object HostProtocol {

  /** Command codes: the first byte of every frame. */
  object Op {
    val Write = 0x01
    val Read = 0x02
    val Matmul = 0x03
    val Status = 0x04
    val all: Seq[Int] = Seq(Write, Read, Matmul, Status)
  }

  val FieldBytes = 2 // Bytes of an address, a count or a dimension field.

  /** WRITE: addr, count, then count data bytes (one per element). No reply. */
  object Write {
    val headerBytes: Int = 2 * FieldBytes // addr + count
  }

  /** READ: addr, count, size; reply: count * size bytes. */
  object Read {
    val headerBytes: Int = 2 * FieldBytes + 1 // addr + count + size
    val sizes: Seq[Int] =
      Seq(1, 4) // 1 = low byte (after requant), 4 = full int32
  }

  /** MATMUL (0x03): start C = A · B, with A of size m×k and B of size k×n.
    *
    * Frame sent by the host: 1 opcode byte + 20 payload bytes = 21 bytes.
    * 2-byte fields are little-endian (low byte first).
    *
    * {{{
    * byte    field    meaning
    * 0       opcode   0x03
    * 1–2     aAddr    input-memory address of A[0][0]
    * 3–4     bAddr    input-memory address of B[0][0]
    * 5–6     cAddr    result-memory address where C[0][0] is written
    * 7–8     m        rows of A and of C; must be <= maxM
    * 9–10    k        columns of A = rows of B; padded to a multiple of banks
    * 11–12   n        columns of B and of C; padded to a multiple of banks
    * 13–14   ldA      distance between two rows of A in the input memory (= k if A is dense)
    * 15–16   ldB      distance between two rows of B in the input memory (= n if B is dense)
    * 17–18   ldC      distance between two rows of C in the result memory (= n if C is dense)
    * 19      flags    bit 0 = requant, bit 1 = relu, bits 2–7 = 0
    * 20      shift    right shift for requantization (0 to accWidth - 1); ignored if requant = 0
    * }}}
    *
    * Matrices are row-major: element (i, j) is at base + i * ld + j. The host
    * guarantees: ldA >= k, ldB >= n, ldC >= n; aAddr, bAddr, ldA, ldB are
    * multiples of banks; cAddr and ldC are multiples of cols. Violations are
    * undefined behaviour.
    *
    * There is no reply. The decoder hands the instruction to the sequencer as
    * soon as it is idle. The host then polls STATUS until busy = 0, and fetches
    * C with READ.
    */
  object Matmul {
    val payloadBytes: Int =
      9 * FieldBytes + 1 + 1 // 9 two-byte fields + flags + shift = 20
    // byte offsets inside the frame (opcode at offset 0)
    val AAddrOffset = 1
    val BAddrOffset = 3
    val CAddrOffset = 5
    val MOffset = 7
    val KOffset = 9
    val NOffset = 11
    val LdAOffset = 13
    val LdBOffset = 15
    val LdCOffset = 17
    val FlagsOffset = 19
    val ShiftOffset = 20
    // bit positions inside the flags byte
    val RequantBit = 0
    val ReluBit = 1
  }

  /** STATUS: ask whether the accelerator is busy and how long the last MATMUL
    * took.
    *
    * Frame sent by the host: only the opcode byte, 0x04. Reply from the FPGA: 9
    * bytes. 4-byte counters are little-endian, unsigned.
    *
    * {{{
    *   byte    field          meaning
    *   0       flags          bit 0 = busy:    a MATMUL is running (results not ready yet)
    *                          bit 1 = error:   an unknown opcode or a timed-out frame was received
    *                          bit 2 = overrun: RxFifo was full and lost at least one byte
    *                          bits 3–7 = 0
    *   1–4     cycles         clock cycles of the last MATMUL, from its acceptance until busy
    *                          dropped (UART time excluded); while busy, the count so far
    *   5–8     busyPeCycles   PE-cycles spent on useful products in the last MATMUL (= m·k·n);
    *                          utilization = busyPeCycles / (cycles · rows · cols)
    * }}}
    * error and overrun mean earlier frames may have been misread: the host
    * should reset and resend its matrices. error and overrun are sticky between
    * reads: each is set by its event and cleared when a STATUS reply is
    * captured, so a reply reports what happened since the previous STATUS. A
    * new event in the same cycle as the capture wins over the clear. busy is a
    * live level, never cleared. Reset clears both flags.
    */
  object Status {
    val replyBytes: Int = 1 + 4 + 4
    val BusyBit = 0 // bit positions in the flags byte
    val ErrorBit = 1
    val OverrunBit = 2
  }
}
