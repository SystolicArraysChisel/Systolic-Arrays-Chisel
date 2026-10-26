"""Byte-level host protocol, mirroring HostProtocol.scala.

Multi-byte fields are little-endian. Every frame starts with an opcode byte.
"""

from dataclasses import dataclass

from .config import HwConfig

OP_WRITE = 0x01
OP_READ = 0x02
OP_MATMUL = 0x03
OP_STATUS = 0x04

FIELD_BYTES = 2
MAX_FIELD = 0xFFFF  # largest value of a 2-byte field (addr, count, dimension)
READ_SIZES = (1, 4)  # bytes per element in a READ reply

MATMUL_FRAME_BYTES = 21  # opcode + 20 payload bytes
REQUANT_BIT = 0  # bit positions in the MATMUL flags byte
RELU_BIT = 1

STATUS_REPLY_BYTES = 9
BUSY_BIT = 0  # bit positions in the STATUS flags byte
ERROR_BIT = 1
OVERRUN_BIT = 2


class ProtocolError(ValueError):
    """A command breaks a guarantee the host must give the hardware."""


def le16(value: int) -> bytes:
    if not 0 <= value <= MAX_FIELD:
        raise ProtocolError(f"{value} does not fit in a 2-byte field")
    return value.to_bytes(2, "little")


def write_frame(addr: int, data: bytes) -> bytes:
    """WRITE: addr, count, then one byte per element. No reply."""
    return bytes([OP_WRITE]) + le16(addr) + le16(len(data)) + bytes(data)


def read_frame(addr: int, count: int, size: int) -> bytes:
    """READ: addr, count, size. Reply: count * size bytes."""
    if size not in READ_SIZES:
        raise ProtocolError(f"READ size must be 1 or 4, got {size}")
    return bytes([OP_READ]) + le16(addr) + le16(count) + bytes([size])


def status_frame() -> bytes:
    return bytes([OP_STATUS])


@dataclass(frozen=True)
class Matmul:
    """One MATMUL: C = A . B, with A m x k and B k x n (row-major)."""

    a_addr: int
    b_addr: int
    c_addr: int
    m: int
    k: int
    n: int
    ld_a: int
    ld_b: int
    ld_c: int
    requant: bool = False
    relu: bool = False
    shift: int = 0

    def frame(self) -> bytes:
        fields = (self.a_addr, self.b_addr, self.c_addr, self.m, self.k, self.n,
                  self.ld_a, self.ld_b, self.ld_c)
        flags = (int(self.requant) << REQUANT_BIT) | (int(self.relu) << RELU_BIT)
        return bytes([OP_MATMUL]) + b"".join(le16(f) for f in fields) + bytes([flags, self.shift])

    def check(self, cfg: HwConfig) -> None:
        """Raise ProtocolError if the hardware's preconditions do not hold."""
        b, cols = cfg.banks, cfg.cols
        rules = [
            (1 <= self.m <= cfg.max_m, f"m = {self.m} must be in 1..{cfg.max_m}"),
            (self.k > 0 and self.k % b == 0, f"k = {self.k} must be a positive multiple of banks ({b})"),
            (self.n > 0 and self.n % b == 0, f"n = {self.n} must be a positive multiple of banks ({b})"),
            (self.ld_a >= self.k, f"ldA = {self.ld_a} < k = {self.k}"),
            (self.ld_b >= self.n, f"ldB = {self.ld_b} < n = {self.n}"),
            (self.ld_c >= self.n, f"ldC = {self.ld_c} < n = {self.n}"),
            (self.a_addr % b == 0 and self.b_addr % b == 0, f"aAddr and bAddr must be multiples of banks ({b})"),
            (self.ld_a % b == 0 and self.ld_b % b == 0, f"ldA and ldB must be multiples of banks ({b})"),
            (self.c_addr % cols == 0 and self.ld_c % cols == 0, f"cAddr and ldC must be multiples of cols ({cols})"),
            (self.a_addr + (self.m - 1) * self.ld_a + self.k <= cfg.in_depth, "A does not fit in the input memory"),
            (self.b_addr + (self.k - 1) * self.ld_b + self.n <= cfg.in_depth, "B does not fit in the input memory"),
            (self.c_addr + (self.m - 1) * self.ld_c + self.n <= cfg.out_depth, "C does not fit in the result memory"),
            (0 <= self.shift < cfg.acc_width, f"shift = {self.shift} must be in 0..{cfg.acc_width - 1}"),
        ]
        broken = [msg for ok, msg in rules if not ok]
        if broken:
            raise ProtocolError("; ".join(broken))


@dataclass(frozen=True)
class Status:
    busy: bool
    error: bool
    overrun: bool
    cycles: int  # cycles of the last MATMUL (so far, while busy)
    busy_pe_cycles: int  # m * k * n of the last MATMUL


def parse_status(reply: bytes) -> Status:
    if len(reply) != STATUS_REPLY_BYTES:
        raise ProtocolError(f"STATUS reply has {len(reply)} bytes, expected {STATUS_REPLY_BYTES}")
    flags = reply[0]
    return Status(
        busy=bool(flags >> BUSY_BIT & 1),
        error=bool(flags >> ERROR_BIT & 1),
        overrun=bool(flags >> OVERRUN_BIT & 1),
        cycles=int.from_bytes(reply[1:5], "little"),
        busy_pe_cycles=int.from_bytes(reply[5:9], "little"),
    )
