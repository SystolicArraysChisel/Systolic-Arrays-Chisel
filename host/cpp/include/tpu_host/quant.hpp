// Quantization: real matrices <-> int8, with one scale per matrix.
//
// Symmetric scheme, no zero point: x ~= scale * q, q in [-127, 127] (-128 is
// never produced). Every rounding is round-half-up (ties go toward +infinity:
// 2.5 -> 3, -2.5 -> -2), the same rule as the hardware requantization.
// Port of tpu_host/quant.py; ref_vectors.txt checks they agree.
#pragma once
#include <cstdint>
#include <span>

#include "tpu_host/matrix.hpp"

namespace tpu_host {

inline constexpr int QMAX = 127;

// floor(v + 0.5), computed without the rounding error of v + 0.5.
double round_half_up(double v);

// q = clamp(round_half_up(x / scale), -127, 127).
std::int8_t quantize(double x, double scale);
Matrix<std::int8_t> quantize(const Matrix<double>& m, double scale);

Matrix<double> dequantize(const Matrix<std::int8_t>& m, double scale);
// For accumulators (int32 sums without requant), which do not fit in int8.
Matrix<double> dequantize(const Matrix<std::int64_t>& m, double scale);

// Scale that maps the largest |x| to 127 (1.0 if every value is 0).
double choose_scale(std::span<const double> values);

// (acc + 2^(shift-1)) >> shift, arithmetic, no rounding term if shift = 0.
std::int64_t rounding_shift(std::int64_t acc, int shift);

// Hardware requantization: rounding shift, then saturation to [-127, 127].
std::int64_t requantize(std::int64_t acc, int shift);

// Smallest shift for which no value saturates.
int choose_shift(std::span<const std::int64_t> acc, int acc_width = 32);

// Scale of the int8 results of a requantized C = A . B.
double requant_scale(double scale_a, double scale_b, int shift);

}  // namespace tpu_host
