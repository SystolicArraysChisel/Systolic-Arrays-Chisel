#include "tpu_host/reference.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "tpu_host/quant.hpp"

namespace tpu_host {

namespace {

std::string shape(std::size_t rows, std::size_t cols) {
    return std::to_string(rows) + "x" + std::to_string(cols);
}

}  // namespace

Matrix<std::int64_t> matmul_acc(const Matrix<std::int8_t>& a, const Matrix<std::int8_t>& b,
                                int acc_width) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("cannot multiply " + shape(a.rows(), a.cols()) + " by " +
                                    shape(b.rows(), b.cols()));
    }
    const std::int64_t hi = (std::int64_t{1} << (acc_width - 1)) - 1;
    const std::int64_t lo = -(std::int64_t{1} << (acc_width - 1));

    Matrix<std::int64_t> c(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            // 64-bit sum: an int would overflow (undefined) before the check below.
            std::int64_t sum = 0;
            for (std::size_t p = 0; p < a.cols(); ++p) {
                sum += std::int64_t{a(i, p)} * std::int64_t{b(p, j)};
            }
            if (sum < lo || sum > hi) {
                throw std::overflow_error("a sum does not fit in " + std::to_string(acc_width) +
                                          " bits");
            }
            c(i, j) = sum;
        }
    }
    return c;
}

std::int64_t post_process(std::int64_t acc, bool requant, bool relu, int shift) {
    const std::int64_t y = requant ? requantize(acc, shift) : acc;
    return relu ? std::max(y, std::int64_t{0}) : y;
}

Matrix<std::int64_t> post_process(const Matrix<std::int64_t>& acc, bool requant, bool relu,
                                  int shift) {
    Matrix<std::int64_t> result(acc.rows(), acc.cols());
    std::ranges::transform(acc, result.begin(), [requant, relu, shift](std::int64_t v) {
        return post_process(v, requant, relu, shift);
    });
    return result;
}

Matrix<std::int64_t> matmul_ref(const Matrix<std::int8_t>& a, const Matrix<std::int8_t>& b,
                                bool requant, bool relu, int shift, int acc_width) {
    return post_process(matmul_acc(a, b, acc_width), requant, relu, shift);
}

}  // namespace tpu_host
