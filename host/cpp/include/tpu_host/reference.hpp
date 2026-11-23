// Bit-exact model of what the accelerator computes for one MATMUL.
//
// C = A . B on int8 inputs with an int32 accumulator, then the post-processing
// of the Accumulators section of the spec: optional requantization (rounding
// shift and saturation to [-127, 127]), then optional ReLU (leaky if leak_shift != 0).
// Port of tpu_host/reference.py; ref_vectors.txt checks they agree.
#pragma once
#include <cstdint>
#include <span>

#include "tpu_host/matrix.hpp"

namespace tpu_host {

// Integer product, plus bias[j] in column j if bias is not empty, wrapped to
// acc_width bits like the accumulator. Throws std::invalid_argument if the
// shapes do not match, std::overflow_error if a sum (before the bias) does not
// fit in acc_width bits.
Matrix<std::int64_t> matmul_acc(const Matrix<std::int8_t>& a, const Matrix<std::int8_t>& b,
                                int acc_width = 32, std::span<const std::int32_t> bias = {});

// Requantization (if requant), then ReLU (if relu): a negative value becomes
// 0, or value >> leak_shift (floor) if leak_shift != 0.
std::int64_t post_process(std::int64_t acc, bool requant, bool relu, int shift,
                          int leak_shift = 0);
Matrix<std::int64_t> post_process(const Matrix<std::int64_t>& acc, bool requant, bool relu,
                                  int shift, int leak_shift = 0);

Matrix<std::int64_t> matmul_ref(const Matrix<std::int8_t>& a, const Matrix<std::int8_t>& b,
                                bool requant = false, bool relu = false, int shift = 0,
                                int acc_width = 32, std::span<const std::int32_t> bias = {},
                                int leak_shift = 0);

}  // namespace tpu_host
