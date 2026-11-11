#include "tpu_host/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "tpu_host/matrix.hpp"

namespace tpu_host {

double round_half_up(double v) {
    // v - floor(v) is exact, whereas v + 0.5 can round up
    // (0.49999999999999994 + 0.5 == 1.0).
    const double f = std::floor(v);
    return v - f >= 0.5 ? f + 1 : f;
}

std::int8_t quantize(double x, double scale) {
    const double qmax = QMAX;
    // Clamp before the cast: converting an out-of-range double to int8 is undefined.
    return static_cast<std::int8_t>(std::clamp(round_half_up(x / scale), -qmax, qmax));
}

Matrix<std::int8_t> quantize(const Matrix<double>& m, double scale) {
    Matrix<std::int8_t> result_m(m.rows(), m.cols());
    std::ranges::transform(m, result_m.begin(), [scale](double x) { return quantize(x, scale); });
    return result_m;
}

Matrix<double> dequantize(const Matrix<std::int8_t>& m, double scale) {
    Matrix<double> result_m(m.rows(), m.cols());
    std::ranges::transform(m, result_m.begin(), [scale](std::int8_t q) { return q * scale; });
    return result_m;
}

Matrix<double> dequantize(const Matrix<std::int64_t>& m, double scale) {
    Matrix<double> result_m(m.rows(), m.cols());
    std::ranges::transform(m, result_m.begin(), [scale](std::int64_t q) {
        return static_cast<double>(q) * scale;  // exact: |q| < 2^31 fits in a double's 53 bits
    });
    return result_m;
}

double choose_scale(std::span<const double> values) {
    const double max = std::accumulate(values.begin(), values.end(), 0.0, [](double best, double x) {
        return std::max(best, std::abs(x));
    });
    return max > 0 ? max / QMAX : 1.0;
}

std::int64_t rounding_shift(std::int64_t acc, int shift) {
    if (shift == 0) {
        return acc;
    }
    // 64-bit 1: 1 << 31 would overflow an int. >> of a negative value is an
    // arithmetic shift (defined since C++20), like numpy's.
    return (acc + (std::int64_t{1} << (shift - 1))) >> shift;
}

std::int64_t requantize(std::int64_t acc, int shift) {
    return std::clamp(rounding_shift(acc, shift), std::int64_t{-QMAX}, std::int64_t{QMAX});
}

int choose_shift(std::span<const std::int64_t> acc, int acc_width) {
    for (int shift = 0; shift < acc_width; ++shift) {
        const bool fits = std::ranges::all_of(
            acc, [shift](std::int64_t v) { return std::abs(rounding_shift(v, shift)) <= QMAX; });
        if (fits) {
            return shift;
        }
    }
    return acc_width - 1;
}

double requant_scale(double scale_a, double scale_b, int shift) {
    return std::ldexp(scale_a * scale_b, shift);  // * 2^shift, exactly
}

}  // namespace tpu_host
