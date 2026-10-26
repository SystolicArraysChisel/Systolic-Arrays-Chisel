import numpy as np
import pytest

from tpu_host.reference import matmul_acc, matmul_ref, post_process

# Example of the workload document: R = C = 2, A 3x4, B 4x2.
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 0, -1, 2]])
B = np.array([[1, 0], [0, 1], [1, 1], [2, -1]])
C = np.array([[12, 1], [28, 5], [4, -3]])


def test_example_product():
    assert (matmul_ref(A, B) == C).all()


def test_relu_without_requant_clamps_the_raw_sums():
    assert (matmul_ref(A, B, relu=True) == np.maximum(C, 0)).all()


def test_requant_then_relu():
    acc = np.array([[300, -300, 5, -5]])
    # shift 1: (acc + 1) >> 1 = [150, -150, 3, -2], saturated, then ReLU
    assert list(post_process(acc, requant=True, relu=False, shift=1)[0]) == [127, -127, 3, -2]
    assert list(post_process(acc, requant=True, relu=True, shift=1)[0]) == [127, 0, 3, 0]


def test_matches_numpy_on_random_int8():
    rng = np.random.default_rng(1)
    a = rng.integers(-128, 128, (7, 13))
    b = rng.integers(-128, 128, (13, 5))
    assert (matmul_acc(a, b) == a.astype(np.int64) @ b.astype(np.int64)).all()


def test_rejects_an_accumulator_overflow_and_bad_shapes():
    with pytest.raises(OverflowError):
        matmul_acc(np.full((1, 3), 127), np.full((3, 1), 127), acc_width=16)  # 48387 > 32767
    with pytest.raises(ValueError):
        matmul_acc(np.ones((2, 3)), np.ones((2, 3)))
