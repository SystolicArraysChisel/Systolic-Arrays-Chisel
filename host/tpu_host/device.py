"""Talking to the accelerator over the serial link.

`Tpu` works with any link that has write(bytes) and read(n) -> bytes, like
pyserial's Serial; tests use tpu_host.sim.FakeTpu instead.

Host habits from the spec, followed here:
  * STATUS after each WRITE and before each READ: a set error or overrun flag
    then always points to the last transfer, and raises TpuError.
  * After a MATMUL, poll STATUS until busy = 0 before sending anything else:
    the decoder reads no byte while a MATMUL waits for the sequencer, so a
    long command sent then would overflow the 16-byte RxFifo.
"""

import time
from dataclasses import dataclass

import numpy as np

from . import protocol as p
from .config import HwConfig
from .quant import choose_scale, choose_shift, dequantize, quantize, requant_scale
from .reference import matmul_acc


class TpuError(RuntimeError):
    """The accelerator reported a problem, or did not answer."""


@dataclass(frozen=True)
class Perf:
    cycles: int = 0  # accelerator cycles, UART time excluded
    busy_pe_cycles: int = 0  # useful PE-cycles (m * k * n, padded sizes)

    def __add__(self, other: "Perf") -> "Perf":
        return Perf(self.cycles + other.cycles, self.busy_pe_cycles + other.busy_pe_cycles)

    def utilization(self, cfg: HwConfig) -> float:
        return self.busy_pe_cycles / (self.cycles * cfg.rows * cfg.cols) if self.cycles else 0.0


@dataclass(frozen=True)
class FloatResult:
    c: np.ndarray  # dequantized result
    scale: float  # scale of the integer results
    shift: int  # requantization shift used (0 without requant)
    perf: Perf


def _round_up(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


class Tpu:
    def __init__(self, link, cfg: HwConfig = HwConfig(), timeout: float = 5.0, poll_interval: float = 0.0):
        self.link = link
        self.cfg = cfg
        self.timeout = timeout  # seconds to wait for a reply or for busy to drop
        self.poll_interval = poll_interval  # seconds between two STATUS polls

    @classmethod
    def open(cls, port: str, cfg: HwConfig = HwConfig(), **kwargs) -> "Tpu":
        import serial  # only needed with real hardware

        return cls(serial.Serial(port, cfg.baud, timeout=0.5), cfg, **kwargs)

    def close(self) -> None:
        self.link.close()

    # ---- raw link ----

    def _send(self, frame: bytes) -> None:
        self.link.write(frame)

    def _recv(self, n: int) -> bytes:
        data = b""
        deadline = time.monotonic() + self.timeout
        while len(data) < n:
            chunk = self.link.read(n - len(data))
            if chunk:
                data += chunk
            elif time.monotonic() > deadline:
                raise TpuError(f"timeout: got {len(data)} of {n} reply bytes")
        return data

    # ---- commands ----

    def status(self) -> p.Status:
        self._send(p.status_frame())
        return p.parse_status(self._recv(p.STATUS_REPLY_BYTES))

    def check(self) -> p.Status:
        """STATUS, raising TpuError if error or overrun is set."""
        s = self.status()
        if s.error or s.overrun:
            raise TpuError(f"accelerator reported error={s.error} overrun={s.overrun}: "
                           "reset it and resend the matrices")
        return s

    def wait_idle(self) -> p.Status:
        """Poll STATUS until busy = 0."""
        deadline = time.monotonic() + self.timeout
        while True:
            s = self.check()
            if not s.busy:
                return s
            if time.monotonic() > deadline:
                raise TpuError("timeout: still busy")
            time.sleep(self.poll_interval)

    def write(self, addr: int, data) -> None:
        """Store int8 elements in the input memory from addr on."""
        data = np.asarray(data, dtype=np.int8).ravel().tobytes()
        if addr < 0 or addr + len(data) > self.cfg.in_depth:
            raise p.ProtocolError(f"WRITE of {len(data)} elements at {addr} leaves the input memory")
        for off in range(0, len(data), p.MAX_FIELD):
            chunk = data[off:off + p.MAX_FIELD]
            self._send(p.write_frame(addr + off, chunk))
            self.check()

    def read(self, addr: int, count: int, size: int = 4) -> np.ndarray:
        """Fetch count results from the result memory, as signed integers."""
        if addr < 0 or addr + count > self.cfg.out_depth:
            raise p.ProtocolError(f"READ of {count} elements at {addr} leaves the result memory")
        dtype = {4: "<i4", 1: "i1"}[size]
        out = []
        for off in range(0, count, p.MAX_FIELD):
            n = min(p.MAX_FIELD, count - off)
            self.check()
            self._send(p.read_frame(addr + off, n, size))
            out.append(np.frombuffer(self._recv(n * size), dtype=dtype))
        return np.concatenate(out).astype(np.int64) if out else np.zeros(0, dtype=np.int64)

    def run(self, mm: p.Matmul) -> Perf:
        """Send one MATMUL and wait until it has finished."""
        mm.check(self.cfg)
        self._send(mm.frame())
        s = self.wait_idle()
        return Perf(s.cycles, s.busy_pe_cycles)

    # ---- whole products ----

    def matmul(self, a_q, b_q, requant: bool = False, relu: bool = False, shift: int = 0):
        """C = A . B on int8 matrices of any shape that fits in the memories.

        K and N are zero-padded to multiples of banks, M is split into
        MATMULs of at most max_m rows. Returns (C, Perf); C is int32 values
        (or int8 values with requant), as int64.
        """
        a = np.asarray(a_q, dtype=np.int8)
        b = np.asarray(b_q, dtype=np.int8)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError(f"cannot multiply {a.shape} by {b.shape}")
        m, k = a.shape
        n = b.shape[1]
        cfg = self.cfg
        kp, np_ = _round_up(k, cfg.banks), _round_up(n, cfg.banks)
        a_addr, b_addr, c_addr = 0, m * kp, 0
        if b_addr + kp * np_ > cfg.in_depth or m * np_ > cfg.out_depth:
            raise p.ProtocolError(f"{m}x{k} . {k}x{n} does not fit in the memories")

        a_pad = np.zeros((m, kp), dtype=np.int8)
        a_pad[:, :k] = a
        b_pad = np.zeros((kp, np_), dtype=np.int8)
        b_pad[:k, :n] = b
        self.write(a_addr, a_pad)
        self.write(b_addr, b_pad)

        perf = Perf()
        for r0 in range(0, m, cfg.max_m):
            rows = min(cfg.max_m, m - r0)
            perf += self.run(p.Matmul(
                a_addr=a_addr + r0 * kp, b_addr=b_addr, c_addr=c_addr + r0 * np_,
                m=rows, k=kp, n=np_, ld_a=kp, ld_b=np_, ld_c=np_,
                requant=requant, relu=relu, shift=shift))

        c = self.read(c_addr, m * np_, size=1 if requant else 4).reshape(m, np_)
        return c[:, :n], perf

    def matmul_float(self, a, b, requant: bool = False, relu: bool = False) -> FloatResult:
        """C ~= A . B on real matrices: quantize, compute on the FPGA, dequantize."""
        sa, sb = choose_scale(a), choose_scale(b)
        qa, qb = quantize(a, sa), quantize(b, sb)
        shift = choose_shift(matmul_acc(qa, qb, self.cfg.acc_width), self.cfg.acc_width) if requant else 0
        c_q, perf = self.matmul(qa, qb, requant=requant, relu=relu, shift=shift)
        scale = requant_scale(sa, sb, shift)
        return FloatResult(dequantize(c_q, scale), scale, shift, perf)
