// Port of tests/test_quant.py.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "tpu_host/matrix.hpp"
#include "tpu_host/quant.hpp"

using namespace tpu_host;

TEST(Quant, RoundHalfUpSendsTiesTowardPlusInfinity) {
    const std::vector<std::pair<double, double>> cases = {
        {2.5, 3},  {-2.5, -2}, {0.5, 1},
        {-0.5, 0}, {1.5, 2},   {-1.5, -1},
        {0.49999999999999994, 0}, {-0.49999999999999994, 0},
        {2.4, 2},  {-2.6, -3},
    };
    for (const auto& [v, expected] : cases) {
        SCOPED_TRACE(v);
        EXPECT_EQ(round_half_up(v), expected);
    }
}

TEST(Quant, QuantizeWorkedExampleFromTheSpec) {
    // A = [0.5, -0.2], B = [0.3, 0.4] with s_A = s_B = 0.005
    EXPECT_EQ(quantize(Matrix<double>(1, 2, {0.5, -0.2}), 0.005),
              (Matrix<std::int8_t>(1, 2, {100, -40})));
    EXPECT_EQ(quantize(Matrix<double>(1, 2, {0.3, 0.4}), 0.005),
              (Matrix<std::int8_t>(1, 2, {60, 80})));
}

TEST(Quant, QuantizeSaturatesToPlusMinus127NeverMinus128) {
    EXPECT_EQ(quantize(Matrix<double>(1, 5, {200.0, -200.0, 127.5, -127.5, -128.0}), 1.0),
              (Matrix<std::int8_t>(1, 5, {127, -127, 127, -127, -127})));
}

TEST(Quant, ChooseScaleMapsTheLargestMagnitudeTo127) {
    const Matrix<double> x(1, 3, {0.3, -1.27, 0.5});
    const double s = choose_scale(x.data());
    EXPECT_NEAR(s, 0.01, 1e-12);
    const Matrix<std::int8_t> q = quantize(x, s);
    const auto biggest =
        std::ranges::max(q, {}, [](std::int8_t v) { return std::abs(static_cast<int>(v)); });
    EXPECT_EQ(std::abs(biggest), 127);

    EXPECT_EQ(choose_scale(std::vector<double>(4, 0.0)), 1.0);
    EXPECT_EQ(choose_scale(std::vector<double>{}), 1.0);
}

TEST(Quant, QuantizationErrorIsAtMostHalfAStep) {
    std::mt19937 rng(0);
    std::normal_distribution<double> dist(0.0, 5.0);
    Matrix<double> x(1, 1000);
    for (double& v : x) {
        v = dist(rng);
    }
    const double s = choose_scale(x.data());
    const Matrix<double> back = dequantize(quantize(x, s), s);
    for (std::size_t i = 0; i < x.cols(); ++i) {
        EXPECT_LE(std::abs(back(0, i) - x(0, i)), s / 2 + 1e-12);
    }
}

TEST(Quant, DequantizeAccumulatorsBeyondInt8) {
    // Raw int32 sums: a cast to int8 first would have wrapped them.
    const Matrix<std::int64_t> acc(1, 4, {1000, -1000, 2'147'483'647, -2'147'483'648});
    EXPECT_EQ(dequantize(acc, 0.5),
              (Matrix<double>(1, 4, {500.0, -500.0, 1'073'741'823.5, -1'073'741'824.0})));
}

TEST(Quant, RequantizeWorkedExampleRoundHalfUpNotTruncation) {
    EXPECT_EQ(requantize(2800, 8), 11);  // (2800 + 128) >> 8; truncation would give 10
    EXPECT_NEAR(requant_scale(0.005, 0.005, 8), 0.0064, 1e-15);
}

TEST(Quant, RequantizeRoundsAndSaturates) {
    const std::int64_t i32_max = (std::int64_t{1} << 31) - 1;
    const std::int64_t i32_min = -(std::int64_t{1} << 31);
    const std::vector<std::tuple<std::int64_t, int, std::int64_t>> cases = {
        {3, 1, 2},        {-3, 1, -1},     {1, 1, 1},   {-1, 1, 0},  // ties toward +infinity
        {5, 0, 5},        {-200, 0, -127}, {200, 0, 127},  // shift 0: no rounding, still saturated
        {i32_max, 31, 1}, {i32_min, 31, -1},  // int32 extremes do not overflow
        {-129 * 4, 2, -127}, {128 * 4, 2, 127},
    };
    for (const auto& [acc, shift, expected] : cases) {
        SCOPED_TRACE(std::to_string(acc) + " >> " + std::to_string(shift));
        EXPECT_EQ(requantize(acc, shift), expected);
    }
}

TEST(Quant, ChooseShiftIsTheSmallestShiftWithoutSaturation) {
    const std::vector<std::pair<std::int64_t, int>> cases = {
        {0, 0}, {127, 0}, {-127, 0}, {128, 1}, {-128, 1}, {255, 2}, {-255, 1}, {2800, 5},
    };
    for (const auto& [value, expected] : cases) {
        SCOPED_TRACE(value);
        const std::vector<std::int64_t> acc{value};
        const int k = choose_shift(acc);
        EXPECT_EQ(k, expected);
        EXPECT_LE(std::abs(rounding_shift(value, k)), QMAX);
        if (k > 0) {  // one less would saturate
            EXPECT_GT(std::abs(rounding_shift(value, k - 1)), QMAX);
        }
    }
}
