// Checks on the real board, skipped unless TPU_PORT is set:
//
//   TPU_PORT=COM3 TPU_CONFIG=../../build/4x4/config.json ./tpu_host_tests --gtest_filter=Board.*
//
// TPU_CONFIG is the config.json of the programmed bitstream (default: the 4x4
// defaults of HwConfig). The cases are the Python run_board smoke cases.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "tpu_host/config.hpp"
#include "tpu_host/matrix.hpp"
#include "tpu_host/reference.hpp"
#include "tpu_host/tpu_driver.hpp"

using namespace tpu_host;

namespace {

class Board : public testing::Test {
   protected:
    void SetUp() override {
        const char* port = std::getenv("TPU_PORT");
        if (port == nullptr) {
            GTEST_SKIP() << "set TPU_PORT (e.g. COM3) to run the board checks";
        }
        const char* config = std::getenv("TPU_CONFIG");
        cfg_ = config != nullptr ? HwConfig::load(config) : HwConfig{};
        port_ = port;
    }

    TpuDriver open() { return TpuDriver::open(port_, cfg_); }

    HwConfig cfg_;
    std::string port_;
};

Matrix<std::int8_t> random_int8(std::mt19937& rng, std::size_t rows, std::size_t cols) {
    std::uniform_int_distribution<int> dist(-128, 127);
    Matrix<std::int8_t> m(rows, cols);
    for (std::int8_t& v : m) {
        v = static_cast<std::int8_t>(dist(rng));
    }
    return m;
}

}  // namespace

TEST_F(Board, StatusAnswers) {
    TpuDriver tpu = open();
    const Status s = tpu.status();  // may report flags left from before: they clear here
    const Status again = tpu.status();
    EXPECT_FALSE(again.busy);
    EXPECT_FALSE(again.error);
    EXPECT_FALSE(again.overrun);
    (void)s;
}

TEST_F(Board, ProductsMatchTheReferenceModel) {
    struct Case {
        std::string name;
        Matrix<std::int8_t> a, b;
        bool requant = false, relu = false;
        int shift = 0;
    };
    std::mt19937 rng(0);
    const std::size_t r = static_cast<std::size_t>(cfg_.rows);
    const std::size_t max_m = static_cast<std::size_t>(cfg_.max_m);
    std::vector<Case> cases;
    cases.push_back({"worked example", Matrix<std::int8_t>(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 1, 0, -1, 2}),
                     Matrix<std::int8_t>(4, 2, {1, 0, 0, 1, 1, 1, 2, -1})});
    cases.push_back({"one tile", random_int8(rng, r, r), random_int8(rng, r, r)});
    cases.push_back({"extreme values", Matrix<std::int8_t>(r, r, std::vector<std::int8_t>(r * r, -128)),
                     Matrix<std::int8_t>(r, r, std::vector<std::int8_t>(r * r, -128))});
    cases.push_back({"tiles over K and N", random_int8(rng, 20, 2 * r), random_int8(rng, 2 * r, 3 * r)});
    cases.push_back({"padding of K and N", random_int8(rng, 5, 7), random_int8(rng, 7, 6)});
    cases.push_back({"m = max_m", random_int8(rng, max_m, r), random_int8(rng, r, 2 * r)});
    cases.push_back({"m > max_m (split)", random_int8(rng, max_m + 9, r), random_int8(rng, r, r)});
    cases.push_back({"requant", random_int8(rng, 6, 2 * r), random_int8(rng, 2 * r, r), true, false, 9});
    cases.push_back(
        {"requant + relu", random_int8(rng, 6, 2 * r), random_int8(rng, 2 * r, r), true, true, 7});
    cases.push_back({"32x32x32", random_int8(rng, 32, 32), random_int8(rng, 32, 32)});

    TpuDriver tpu = open();
    tpu.status();  // clear flags left from before
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        const auto [result, perf] = tpu.matmul(c.a, c.b, c.requant, c.relu, c.shift);
        EXPECT_EQ(result, matmul_ref(c.a, c.b, c.requant, c.relu, c.shift));
        EXPECT_GT(perf.cycles, 0u);
    }
}

TEST_F(Board, FloatProductIsClose) {
    std::mt19937 rng(1);
    std::normal_distribution<double> dist(0.0, 1.0);
    Matrix<double> x(16, 24), w(24, 8);
    for (double& v : x) v = dist(rng);
    for (double& v : w) v = dist(rng);

    TpuDriver tpu = open();
    const FloatResult res = tpu.matmul_float(x, w, /*requant=*/true);
    double peak = 0, err = 0;
    for (std::size_t i = 0; i < 16; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            double exact = 0;
            for (std::size_t p = 0; p < 24; ++p) {
                exact += x(i, p) * w(p, j);
            }
            peak = std::max(peak, std::abs(exact));
            err = std::max(err, std::abs(res.c(i, j) - exact));
        }
    }
    EXPECT_LT(err / peak, 0.05) << "max error relative to the largest |C|";
}
