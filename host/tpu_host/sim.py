"""A software stand-in for the FPGA, for testing the host without hardware.

FakeTpu is a link (write / read, like pyserial's Serial) that answers the
host protocol the way the CommandDecoder does: same frames, same error
rules, sticky error and overrun flags cleared by STATUS. MATMUL is computed
with the reference model. Its cycle count is only a rough estimate.
"""

import numpy as np

from . import protocol as p
from .config import HwConfig
from .reference import post_process


class FakeTpu:
    def __init__(self, cfg: HwConfig = HwConfig(), busy_polls: int = 1):
        self.cfg = cfg
        self.busy_polls = busy_polls  # STATUS replies with busy = 1 after each MATMUL
        self.inp = np.zeros(cfg.in_depth, dtype=np.int8)
        self.res = np.zeros(cfg.out_depth, dtype=np.int64)
        self.error = False
        self.overrun = False
        self.cycles = 0
        self.busy_pe_cycles = 0
        self.ops = []  # opcode of every frame received, for tests
        self._busy_left = 0
        self._rx = bytearray()
        self._tx = bytearray()

    # ---- link ----

    def write(self, data: bytes) -> int:
        self._rx += data
        self._process()
        return len(data)

    def read(self, n: int) -> bytes:
        out = bytes(self._tx[:n])
        del self._tx[:n]
        return out

    def close(self) -> None:
        pass

    # ---- decoder ----

    def _frame_len(self):
        """Length of the frame at the head of _rx, or None if not known yet."""
        op = self._rx[0]
        if op == p.OP_WRITE:
            return 5 + int.from_bytes(self._rx[3:5], "little") if len(self._rx) >= 5 else None
        return {p.OP_READ: 6, p.OP_MATMUL: p.MATMUL_FRAME_BYTES, p.OP_STATUS: 1}[op]

    def _process(self) -> None:
        while self._rx:
            op = self._rx[0]
            if op not in (p.OP_WRITE, p.OP_READ, p.OP_MATMUL, p.OP_STATUS):
                self.error = True  # unknown opcode: ignored
                del self._rx[0]
                continue
            n = self._frame_len()
            if n is None or len(self._rx) < n:
                return
            frame = bytes(self._rx[:n])
            del self._rx[:n]
            self.ops.append(op)
            {p.OP_WRITE: self._write, p.OP_READ: self._read,
             p.OP_MATMUL: self._matmul, p.OP_STATUS: self._status}[op](frame)

    @staticmethod
    def _f16(frame: bytes, off: int) -> int:
        return int.from_bytes(frame[off:off + 2], "little")

    def _write(self, f: bytes) -> None:
        addr = self._f16(f, 1)
        for i, byte in enumerate(f[5:]):
            self.inp[(addr + i) % self.cfg.in_depth] = np.int8(byte - 256 if byte > 127 else byte)

    def _read(self, f: bytes) -> None:
        addr, count, size = self._f16(f, 1), self._f16(f, 3), f[5]
        if size not in p.READ_SIZES:
            self.error = True
            return
        for i in range(count):
            v = int(self.res[(addr + i) % self.cfg.out_depth]) & ((1 << (8 * size)) - 1)
            self._tx += v.to_bytes(size, "little")

    def _matmul(self, f: bytes) -> None:
        a_addr, b_addr, c_addr, m, k, n, ld_a, ld_b, ld_c = (self._f16(f, 1 + 2 * i) for i in range(9))
        flags, shift = f[19], f[20]
        if shift >= self.cfg.acc_width or flags & ~0b11:
            self.error = True
            return
        ia = (a_addr + np.arange(m)[:, None] * ld_a + np.arange(k)) % self.cfg.in_depth
        ib = (b_addr + np.arange(k)[:, None] * ld_b + np.arange(n)) % self.cfg.in_depth
        acc = self.inp[ia].astype(np.int64) @ self.inp[ib].astype(np.int64)
        c = post_process(acc, bool(flags >> p.REQUANT_BIT & 1), bool(flags >> p.RELU_BIT & 1), shift)
        ic = (c_addr + np.arange(m)[:, None] * ld_c + np.arange(n)) % self.cfg.out_depth
        self.res[ic] = c
        rows, cols = self.cfg.rows, self.cfg.cols
        tiles = (k // rows) * (n // cols)
        self.cycles = tiles * (rows + m + rows + cols - 1)  # load + stream + drain, per tile
        self.busy_pe_cycles = m * k * n
        self._busy_left = self.busy_polls

    def _status(self, _f: bytes) -> None:
        busy = self._busy_left > 0
        self._busy_left = max(0, self._busy_left - 1)
        flags = (busy << p.BUSY_BIT) | (self.error << p.ERROR_BIT) | (self.overrun << p.OVERRUN_BIT)
        self._tx += bytes([flags]) + self.cycles.to_bytes(4, "little") + self.busy_pe_cycles.to_bytes(4, "little")
        self.error = self.overrun = False  # cleared when the reply is captured
