// Bit-exact model of what the accelerator computes for one MATMUL.
//
// C = A . B on int8 inputs with an int32 accumulator, then the post-processing
// of the Accumulators section of the spec: optional requantization (rounding
// shift and saturation to [-127, 127]), then optional ReLU.
// Port of tpu_host/reference.py; ref_vectors.txt checks they agree.
#pragma once
#include <cstdint>

#include "tpu_host/matrix.hpp"

namespace tpu_host {

// Exact integer product. Throws std::invalid_argument if the shapes do not
// match, std::overflow_error if a sum does not fit in acc_width bits.
Matrix<std::int64_t> matmul_acc(const Matrix<std::int8_t>& a, const Matrix<std::int8_t>& b,
                                int acc_width = 32);

// Requantization (if requant), then ReLU (if relu).
std::int64_t post_process(std::int64_t acc, bool requant, bool relu, int shift);
Matrix<std::int64_t> post_process(const Matrix<std::int64_t>& acc, bool requant, bool relu,
                                  int shift);

Matrix<std::int64_t> matmul_ref(const Matrix<std::int8_t>& a, const Matrix<std::int8_t>& b,
                                bool requant = false, bool relu = false, int shift = 0,
                                int acc_width = 32);

}  // namespace tpu_host
