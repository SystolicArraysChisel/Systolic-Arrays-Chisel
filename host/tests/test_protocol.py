import pytest

from tpu_host import protocol as p
from tpu_host.config import HwConfig

CFG = HwConfig()


def test_write_frame():
    assert p.write_frame(0x1234, bytes([1, 0xff])) == bytes([0x01, 0x34, 0x12, 0x02, 0x00, 0x01, 0xff])


def test_read_frame_and_size_check():
    assert p.read_frame(0x0102, 3, 4) == bytes([0x02, 0x02, 0x01, 0x03, 0x00, 0x04])
    with pytest.raises(p.ProtocolError):
        p.read_frame(0, 1, 2)


def test_matmul_frame_follows_the_byte_table():
    mm = p.Matmul(a_addr=0x0102, b_addr=0x0304, c_addr=0x0506, m=0x0708, k=0x090a, n=0x0b0c,
                  ld_a=0x0d0e, ld_b=0x0f10, ld_c=0x1112, requant=True, relu=True, shift=9)
    f = mm.frame()
    assert len(f) == p.MATMUL_FRAME_BYTES
    assert f[0] == p.OP_MATMUL
    # each 2-byte field at its offset, low byte first
    for off, value in [(1, 0x0102), (3, 0x0304), (5, 0x0506), (7, 0x0708), (9, 0x090a), (11, 0x0b0c),
                       (13, 0x0d0e), (15, 0x0f10), (17, 0x1112)]:
        assert int.from_bytes(f[off:off + 2], "little") == value
    assert f[19] == 0b11 and f[20] == 9


def dense(m=4, k=8, n=4, **kw):
    return p.Matmul(a_addr=0, b_addr=64, c_addr=0, m=m, k=k, n=n, ld_a=k, ld_b=n, ld_c=n, **kw)


def test_matmul_check_accepts_a_dense_product():
    dense().check(CFG)
    dense(m=CFG.max_m, shift=31, requant=True).check(CFG)


@pytest.mark.parametrize("mm, reason", [
    (dense(m=0), "m"), (dense(m=CFG.max_m + 1), "m"), (dense(k=6), "k"), (dense(n=2), "n"),
    (dense(shift=32), "shift"),
    (p.Matmul(0, 64, 0, 4, 8, 4, ld_a=4, ld_b=4, ld_c=4), "ldA"),
    (p.Matmul(2, 64, 0, 4, 8, 4, 8, 4, 4), "aAddr"),
    (p.Matmul(0, 64, 2, 4, 8, 4, 8, 4, 4), "cAddr"),
    (p.Matmul(0, 4090, 0, 4, 8, 4, 8, 4, 4), "B does not fit"),
])
def test_matmul_check_rejects_broken_guarantees(mm, reason):
    with pytest.raises(p.ProtocolError, match=reason):
        mm.check(CFG)


def test_parse_status():
    s = p.parse_status(bytes([0b111]) + (0xdeadbeef).to_bytes(4, "little") + (64).to_bytes(4, "little"))
    assert s == p.Status(busy=True, error=True, overrun=True, cycles=0xdeadbeef, busy_pe_cycles=64)
    with pytest.raises(p.ProtocolError):
        p.parse_status(bytes(8))
