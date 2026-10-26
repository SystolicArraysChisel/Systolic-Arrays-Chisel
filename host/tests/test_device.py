"""Host driver against FakeTpu, the software model of the FPGA."""

import numpy as np
import pytest

from tpu_host import protocol as p
from tpu_host.config import HwConfig
from tpu_host.device import Tpu, TpuError
from tpu_host.quant import choose_scale
from tpu_host.reference import matmul_ref
from tpu_host.sim import FakeTpu


@pytest.fixture
def fake():
    return FakeTpu(busy_polls=3)


@pytest.fixture
def tpu(fake):
    return Tpu(fake, timeout=0.5)


def test_write_then_check_status(tpu, fake):
    tpu.write(8, [1, -1, 127, -128])
    assert list(fake.inp[8:12]) == [1, -1, 127, -128]
    assert fake.ops == [p.OP_WRITE, p.OP_STATUS]  # STATUS after each WRITE


def test_read_signed_results_in_both_sizes(tpu, fake):
    fake.res[0:4] = [-1, 2**31 - 1, -(2**31), 100]
    assert list(tpu.read(0, 4)) == [-1, 2**31 - 1, -(2**31), 100]
    fake.res[0:3] = [-127, 127, 0]
    assert list(tpu.read(0, 3, size=1)) == [-127, 127, 0]
    assert fake.ops == [p.OP_STATUS, p.OP_READ] * 2  # STATUS before each READ


def test_run_polls_status_until_not_busy(tpu, fake):
    perf = tpu.run(p.Matmul(0, 16, 0, m=4, k=4, n=4, ld_a=4, ld_b=4, ld_c=4))
    assert fake.ops == [p.OP_MATMUL] + [p.OP_STATUS] * 4  # 3 busy replies, then idle
    assert perf.busy_pe_cycles == 64


def test_errors_raise(tpu, fake):
    fake.write(bytes([0x99]))  # unknown opcode sets error on the FPGA
    with pytest.raises(TpuError, match="error=True"):
        tpu.write(0, [1])
    fake.overrun = True  # as if RxFifo had lost a byte
    with pytest.raises(TpuError, match="overrun=True"):
        tpu.check()
    tpu.check()  # both flags were cleared by the STATUS that reported them


def test_timeout_when_nothing_answers():
    class Silent:
        def write(self, data): return len(data)
        def read(self, n): return b""
    with pytest.raises(TpuError, match="timeout"):
        Tpu(Silent(), timeout=0.05).status()


def test_writes_are_refused_outside_the_memory(tpu):
    with pytest.raises(p.ProtocolError):
        tpu.write(4095, [1, 2])
    with pytest.raises(p.ProtocolError):
        tpu.read(4090, 7)


@pytest.mark.parametrize("m, k, n", [(1, 1, 1), (3, 4, 2), (5, 7, 9), (4, 8, 4), (130, 6, 3)])
@pytest.mark.parametrize("requant, relu, shift", [(False, False, 0), (False, True, 0), (True, False, 6),
                                                   (True, True, 9)])
def test_matmul_matches_the_reference_model(tpu, m, k, n, requant, relu, shift):
    rng = np.random.default_rng(m * 100 + k * 10 + n)
    a = rng.integers(-127, 128, (m, k))
    b = rng.integers(-127, 128, (k, n))
    c, perf = tpu.matmul(a, b, requant=requant, relu=relu, shift=shift)
    assert c.shape == (m, n)
    assert (c == matmul_ref(a, b, requant, relu, shift)).all()
    kp, np_ = -(-k // 4) * 4, -(-n // 4) * 4
    assert perf.busy_pe_cycles == m * kp * np_  # padded sizes, summed over the m chunks


def test_matmul_splits_m_into_chunks_of_max_m():
    cfg = HwConfig(max_m=8)
    fake = FakeTpu(cfg)
    c, _ = Tpu(fake, cfg).matmul(np.ones((20, 4)), np.ones((4, 4)))
    assert (c == 4).all()
    assert fake.ops.count(p.OP_MATMUL) == 3  # 8 + 8 + 4 rows


def test_matmul_refuses_matrices_that_do_not_fit(tpu):
    with pytest.raises(p.ProtocolError, match="does not fit"):
        tpu.matmul(np.ones((64, 64)), np.ones((64, 64)))


@pytest.mark.parametrize("requant", [False, True])
def test_float_matmul_is_close_to_numpy(tpu, requant):
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, (10, 12))
    b = rng.normal(0, 1, (12, 6))
    res = tpu.matmul_float(a, b, requant=requant)
    # Acceptance criterion: each input is off by at most half a step (s/2), so
    # each product by at most |a| sb/2 + |b| sa/2 + sa sb/4, summed over K;
    # requantization adds at most half an output step.
    sa, sb = choose_scale(a), choose_scale(b)
    per_product = np.max(np.abs(a)) * sb / 2 + np.max(np.abs(b)) * sa / 2 + sa * sb / 4
    bound = a.shape[1] * per_product + (res.scale / 2 if requant else 0)
    assert np.max(np.abs(res.c - a @ b)) <= bound
