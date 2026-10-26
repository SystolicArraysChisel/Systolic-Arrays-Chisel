import numpy as np
import pytest

from tpu_host.quant import (choose_scale, choose_shift, dequantize, quantize, requant_scale,
                            requantize, round_half_up, rounding_shift)


@pytest.mark.parametrize("v, expected", [
    (2.5, 3), (-2.5, -2), (0.5, 1), (-0.5, 0), (1.5, 2), (-1.5, -1),
    (0.49999999999999994, 0), (-0.49999999999999994, 0), (2.4, 2), (-2.6, -3),
])
def test_round_half_up_sends_ties_toward_plus_infinity(v, expected):
    assert round_half_up(v) == expected


def test_quantize_worked_example_from_the_spec():
    # A = [0.5, -0.2], B = [0.3, 0.4] with s_A = s_B = 0.005
    assert list(quantize([0.5, -0.2], 0.005)) == [100, -40]
    assert list(quantize([0.3, 0.4], 0.005)) == [60, 80]


def test_quantize_saturates_to_plus_minus_127_never_minus_128():
    q = quantize([200.0, -200.0, 127.5, -127.5, -128.0], 1.0)
    assert list(q) == [127, -127, 127, -127, -127]
    assert q.dtype == np.int8


def test_choose_scale_maps_the_largest_magnitude_to_127():
    x = np.array([0.3, -1.27, 0.5])
    s = choose_scale(x)
    assert s == pytest.approx(0.01)
    assert np.max(np.abs(quantize(x, s))) == 127
    assert choose_scale(np.zeros(4)) == 1.0


def test_quantization_error_is_at_most_half_a_step():
    x = np.random.default_rng(0).normal(0, 5, 1000)
    s = choose_scale(x)
    assert np.max(np.abs(dequantize(quantize(x, s), s) - x)) <= s / 2 + 1e-12


def test_requantize_worked_example_round_half_up_not_truncation():
    assert requantize(2800, 8) == 11  # (2800 + 128) >> 8; truncation would give 10
    assert requant_scale(0.005, 0.005, 8) == pytest.approx(0.0064)


@pytest.mark.parametrize("acc, shift, expected", [
    (3, 1, 2), (-3, 1, -1), (1, 1, 1), (-1, 1, 0),  # ties toward +infinity
    (5, 0, 5), (-200, 0, -127), (200, 0, 127),  # shift 0: no rounding term, still saturated
    ((1 << 31) - 1, 31, 1), (-(1 << 31), 31, -1),  # int32 extremes do not overflow
    (-129 * 4, 2, -127), (128 * 4, 2, 127),
])
def test_requantize_rounds_and_saturates(acc, shift, expected):
    assert requantize(acc, shift) == expected


@pytest.mark.parametrize("acc, expected", [
    ([0], 0), ([127], 0), ([-127], 0), ([128], 1), ([-128], 1), ([255], 2), ([-255], 1), ([2800], 5),
])
def test_choose_shift_is_the_smallest_shift_without_saturation(acc, expected):
    k = choose_shift(acc)
    assert k == expected
    assert np.max(np.abs(rounding_shift(acc, k))) <= 127
    if k > 0:  # one less would saturate
        assert np.max(np.abs(rounding_shift(acc, k - 1))) > 127
