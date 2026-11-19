// Port of tests/test_reference.py.

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <stdexcept>

#include "tpu_host/matrix.hpp"
#include "tpu_host/reference.hpp"

using namespace tpu_host;

namespace {

// Example of the workload document: R = C = 2, A 3x4, B 4x2.
const Matrix<std::int8_t> A(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 1, 0, -1, 2});
const Matrix<std::int8_t> B(4, 2, {1, 0, 0, 1, 1, 1, 2, -1});
const Matrix<std::int64_t> C(3, 2, {12, 1, 28, 5, 4, -3});

Matrix<std::int8_t> random_int8(std::mt19937& rng, std::size_t rows, std::size_t cols) {
    std::uniform_int_distribution<int> dist(-128, 127);  // no int8_t distribution exists
    Matrix<std::int8_t> m(rows, cols);
    for (std::int8_t& v : m) {
        v = static_cast<std::int8_t>(dist(rng));
    }
    return m;
}

}  // namespace

TEST(Reference, ExampleProduct) { EXPECT_EQ(matmul_ref(A, B), C); }

TEST(Reference, ReluWithoutRequantClampsTheRawSums) {
    EXPECT_EQ(matmul_ref(A, B, /*requant=*/false, /*relu=*/true),
              (Matrix<std::int64_t>(3, 2, {12, 1, 28, 5, 4, 0})));
}

TEST(Reference, RequantThenRelu) {
    const Matrix<std::int64_t> acc(1, 4, {300, -300, 5, -5});
    // shift 1: (acc + 1) >> 1 = [150, -150, 3, -2], saturated, then ReLU
    EXPECT_EQ(post_process(acc, /*requant=*/true, /*relu=*/false, /*shift=*/1),
              (Matrix<std::int64_t>(1, 4, {127, -127, 3, -2})));
    EXPECT_EQ(post_process(acc, /*requant=*/true, /*relu=*/true, /*shift=*/1),
              (Matrix<std::int64_t>(1, 4, {127, 0, 3, 0})));
}

TEST(Reference, MatchesANaiveProductOnRandomInt8) {
    std::mt19937 rng(1);
    const Matrix<std::int8_t> a = random_int8(rng, 7, 13);
    const Matrix<std::int8_t> b = random_int8(rng, 13, 5);
    const Matrix<std::int64_t> c = matmul_acc(a, b);
    for (std::size_t i = 0; i < 7; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            std::int64_t expected = 0;
            for (std::size_t p = 0; p < 13; ++p) {
                expected += a(i, p) * b(p, j);
            }
            EXPECT_EQ(c(i, j), expected);
        }
    }
}

TEST(Reference, RejectsAnAccumulatorOverflow) {
    const Matrix<std::int8_t> a(1, 3, {127, 127, 127});
    const Matrix<std::int8_t> b(3, 1, {127, 127, 127});
    EXPECT_THROW(matmul_acc(a, b, /*acc_width=*/16), std::overflow_error);  // 48387 > 32767
    EXPECT_NO_THROW(matmul_acc(a, b));                                       // fits in 32 bits
}

TEST(Reference, RejectsBadShapes) {
    const Matrix<std::int8_t> m(2, 3);
    EXPECT_THROW(matmul_acc(m, m), std::invalid_argument);
}
