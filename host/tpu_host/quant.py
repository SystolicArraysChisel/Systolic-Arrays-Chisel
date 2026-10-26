"""Quantization: real matrices <-> int8, with one scale per matrix.

Symmetric scheme, no zero point: x ~= scale * q, q in [-127, 127] (-128 is
never produced). Every rounding is round-half-up (ties go toward +infinity:
2.5 -> 3, -2.5 -> -2), the same rule as the hardware requantization. Python's round() and numpy.round()
round half to even and must not be used here.

The Scala copy is tpu.ref.Quant; ref_vectors.txt checks they agree.
"""

import numpy as np

QMAX = 127


def round_half_up(v):
    """floor(v + 0.5), elementwise, as float64.

    Computed as floor(v), plus 1 if the fraction is at least 0.5: v - floor(v)
    is exact, whereas v + 0.5 can round up (0.49999999999999994 + 0.5 == 1.0).
    """
    v = np.asarray(v, dtype=np.float64)
    f = np.floor(v)
    return np.where(v - f >= 0.5, f + 1.0, f)


def choose_scale(x) -> float:
    """Scale that maps the largest |x| to 127 (1.0 for an all-zero matrix)."""
    m = float(np.max(np.abs(np.asarray(x, dtype=np.float64)), initial=0.0))
    return m / QMAX if m > 0 else 1.0


def quantize(x, scale: float) -> np.ndarray:
    """q = clamp(floor(x / scale + 0.5), -127, 127), as int8."""
    q = round_half_up(np.asarray(x, dtype=np.float64) / scale)
    return np.clip(q, -QMAX, QMAX).astype(np.int8)


def dequantize(q, scale: float) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * scale


def rounding_shift(acc, shift: int) -> np.ndarray:
    """(acc + 2^(shift-1)) >> shift, arithmetic, no rounding term if shift = 0."""
    acc = np.asarray(acc, dtype=np.int64)
    if shift == 0:
        return acc
    return (acc + (1 << (shift - 1))) >> shift


def requantize(acc, shift: int) -> np.ndarray:
    """Hardware requantization: rounding shift, then saturation to [-127, 127]."""
    return np.clip(rounding_shift(acc, shift), -QMAX, QMAX)


def choose_shift(acc, acc_width: int = 32) -> int:
    """Smallest shift for which no element of acc saturates."""
    acc = np.asarray(acc, dtype=np.int64)
    for shift in range(acc_width):
        if acc.size == 0 or np.max(np.abs(rounding_shift(acc, shift))) <= QMAX:
            return shift
    return acc_width - 1


def requant_scale(scale_a: float, scale_b: float, shift: int) -> float:
    """Scale of the int8 results of a requantized C = A . B."""
    return scale_a * scale_b * 2.0 ** shift
