"""Bit-exact model of what the accelerator computes for one MATMUL.

C = A . B on int8 inputs with an int32 accumulator, then the post-processing
of the Accumulators section of the spec: optional requantization (rounding
shift and saturation to [-127, 127]), then optional ReLU.

The Scala copy is tpu.ref.RefModel; ref_vectors.txt checks they agree.
"""

import numpy as np

from .quant import requantize


def matmul_acc(a_q, b_q, acc_width: int = 32) -> np.ndarray:
    """Exact integer product, checked to fit in the accumulator."""
    a = np.asarray(a_q, dtype=np.int64)
    b = np.asarray(b_q, dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"cannot multiply {a.shape} by {b.shape}")
    acc = a @ b
    lo, hi = -(1 << (acc_width - 1)), (1 << (acc_width - 1)) - 1
    if acc.size and (acc.min() < lo or acc.max() > hi):
        raise OverflowError(f"a sum does not fit in {acc_width} bits")
    return acc


def post_process(acc, requant: bool, relu: bool, shift: int) -> np.ndarray:
    y = requantize(acc, shift) if requant else np.asarray(acc, dtype=np.int64)
    return np.maximum(y, 0) if relu else y


def matmul_ref(a_q, b_q, requant: bool = False, relu: bool = False, shift: int = 0,
               acc_width: int = 32) -> np.ndarray:
    return post_process(matmul_acc(a_q, b_q, acc_width), requant, relu, shift)
