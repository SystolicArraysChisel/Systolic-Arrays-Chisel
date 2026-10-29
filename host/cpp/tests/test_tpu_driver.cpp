// Port of tests/test_device.py: the driver against FakeTpu.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tpu_host/config.hpp"
#include "tpu_host/fake_tpu.hpp"
#include "tpu_host/link.hpp"
#include "tpu_host/matrix.hpp"
#include "tpu_host/protocol.hpp"
#include "tpu_host/quant.hpp"
#include "tpu_host/reference.hpp"
#include "tpu_host/tpu_driver.hpp"

using namespace tpu_host;
using namespace std::chrono_literals;
using testing::HasSubstr;
using testing::ThrowsMessage;

namespace {

struct Setup {
    FakeTpu* fake;  // observer: owned by tpu, valid as long as tpu lives
    TpuDriver tpu;
};

Setup make(int busy_polls = 3) {
    auto owner = std::make_unique<FakeTpu>(HwConfig{}, busy_polls);
    FakeTpu* fake = owner.get();
    return {fake, TpuDriver(std::move(owner), HwConfig{}, 500ms)};
}

// Uniform in [-127, 127], the range quantize produces.
Matrix<std::int8_t> random_int8(std::mt19937& rng, std::size_t rows, std::size_t cols) {
    std::uniform_int_distribution<int> dist(-127, 127);
    Matrix<std::int8_t> m(rows, cols);
    for (std::int8_t& v : m) {
        v = static_cast<std::int8_t>(dist(rng));
    }
    return m;
}

// A link that never answers.
class SilentLink : public Link {
   public:
    void write(ByteView) override {}
    Bytes read(std::size_t) override { return {}; }
};

}  // namespace

TEST(TpuDriver, StatusReadsTheReply) {
    auto [fake, tpu] = make();
    const Status s = tpu.status();
    EXPECT_FALSE(s.busy);
    EXPECT_FALSE(s.error);
    EXPECT_EQ(fake->ops, std::vector<Op>{Op::Status});
}

TEST(TpuDriver, ErrorsRaise) {
    auto [fake, tpu] = make();
    fake->write(Bytes{0x99});  // unknown opcode sets error on the FPGA
    EXPECT_THAT([&] { tpu.check(); }, ThrowsMessage<TpuError>(HasSubstr("error=true")));
    fake->overrun = true;  // as if RxFifo had lost a byte
    EXPECT_THAT([&] { tpu.check(); }, ThrowsMessage<TpuError>(HasSubstr("overrun=true")));
    EXPECT_NO_THROW(tpu.check());  // both flags were cleared by the STATUS that reported them
}

TEST(TpuDriver, TimeoutWhenNothingAnswers) {
    TpuDriver tpu(std::make_unique<SilentLink>(), HwConfig{}, 50ms);
    EXPECT_THAT([&] { tpu.status(); }, ThrowsMessage<TpuError>(HasSubstr("timeout: got 0 of 9")));
}

TEST(TpuDriver, WaitIdlePollsUntilNotBusy) {
    auto [fake, tpu] = make(/*busy_polls=*/3);
    fake->write(Matmul{.m = 4, .k = 4, .n = 4, .ld_a = 4, .ld_b = 4, .ld_c = 4}.frame());
    const Status s = tpu.wait_idle();
    EXPECT_FALSE(s.busy);
    EXPECT_EQ(s.busy_pe_cycles, 64u);
    // 3 busy replies, then the idle one
    EXPECT_EQ(fake->ops, (std::vector<Op>{Op::Matmul, Op::Status, Op::Status, Op::Status,
                                          Op::Status}));
}

TEST(TpuDriver, WriteThenCheckStatus) {
    auto [fake, tpu] = make();
    const std::vector<std::int8_t> values{1, -1, 127, -128};
    tpu.write(8, values);
    EXPECT_EQ(std::vector<std::int8_t>(fake->inp.begin() + 8, fake->inp.begin() + 12), values);
    EXPECT_EQ(fake->ops, (std::vector<Op>{Op::Write, Op::Status}));  // STATUS after each WRITE
}

TEST(TpuDriver, WriteSplitsIntoFramesOfMaxField) {
    HwConfig cfg;
    cfg.in_depth = 70'000;  // more than one 2-byte count
    auto owner = std::make_unique<FakeTpu>(cfg);
    FakeTpu* fake = owner.get();
    TpuDriver tpu(std::move(owner), cfg);
    std::vector<std::int8_t> values(66'000, 0);
    values.back() = 9;
    tpu.write(0, values);
    EXPECT_EQ(fake->ops, (std::vector<Op>{Op::Write, Op::Status, Op::Write, Op::Status}));
    EXPECT_EQ(fake->inp[65'999], 9);
}

TEST(TpuDriver, ReadSignedResultsInBothSizes) {
    auto [fake, tpu] = make();
    const std::vector<std::int64_t> words{-1, INT32_MAX, INT32_MIN, 100};
    std::ranges::copy(words, fake->res.begin());
    EXPECT_EQ(tpu.read(0, 4), words);
    const std::vector<std::int64_t> bytes{-127, 127, 0};
    std::ranges::copy(bytes, fake->res.begin());
    EXPECT_EQ(tpu.read(0, 3, /*size=*/1), bytes);
    // STATUS before each READ
    EXPECT_EQ(fake->ops, (std::vector<Op>{Op::Status, Op::Read, Op::Status, Op::Read}));
}

TEST(TpuDriver, RunPollsStatusUntilNotBusy) {
    auto [fake, tpu] = make(/*busy_polls=*/3);
    const Perf perf =
        tpu.run(Matmul{.b_addr = 16, .m = 4, .k = 4, .n = 4, .ld_a = 4, .ld_b = 4, .ld_c = 4});
    EXPECT_EQ(fake->ops, (std::vector<Op>{Op::Matmul, Op::Status, Op::Status, Op::Status,
                                          Op::Status}));  // 3 busy replies, then idle
    EXPECT_EQ(perf.busy_pe_cycles, 64u);
}

TEST(TpuDriver, RefusesOutsideTheMemory) {
    auto [fake, tpu] = make();
    const std::vector<std::int8_t> two{1, 2};
    EXPECT_THROW(tpu.write(4095, two), ProtocolError);
    EXPECT_THROW(tpu.write(-1, two), ProtocolError);
    EXPECT_THROW(tpu.read(4090, 7), ProtocolError);
    EXPECT_TRUE(fake->ops.empty());  // nothing sent
}

TEST(TpuDriver, RunRefusesAnInvalidMatmul) {
    auto [fake, tpu] = make();
    EXPECT_THROW(tpu.run(Matmul{.m = 0, .k = 4, .n = 4, .ld_a = 4, .ld_b = 4, .ld_c = 4}),
                 ProtocolError);
    EXPECT_TRUE(fake->ops.empty());  // refused before anything was sent
}

TEST(TpuDriver, PerfAddsUp) {
    Perf total;
    total += Perf{10, 40};
    total += Perf{6, 24};
    EXPECT_EQ(total.cycles, 16u);
    EXPECT_EQ(total.busy_pe_cycles, 64u);
    EXPECT_DOUBLE_EQ(total.utilization(HwConfig{}), 64.0 / (16 * 16));  // 4x4 PEs
    EXPECT_EQ(Perf{}.utilization(HwConfig{}), 0.0);                     // no division by 0
}

TEST(TpuDriver, MatmulOfTheWorkloadExample) {
    auto [fake, tpu] = make();
    const Matrix<std::int8_t> a(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 1, 0, -1, 2});
    const Matrix<std::int8_t> b(4, 2, {1, 0, 0, 1, 1, 1, 2, -1});
    const auto [c, perf] = tpu.matmul(a, b);
    EXPECT_EQ(c, (Matrix<std::int64_t>(3, 2, {12, 1, 28, 5, 4, -3})));
    EXPECT_EQ(perf.busy_pe_cycles, 3u * 4u * 4u);  // n padded from 2 to 4
}

TEST(TpuDriver, MatmulMatchesTheReferenceModel) {
    struct Shape {
        std::size_t m, k, n;
    };
    struct Flags {
        bool requant, relu;
        int shift;
    };
    const std::vector<Shape> shapes{{1, 1, 1}, {3, 4, 2}, {5, 7, 9}, {4, 8, 4}, {130, 6, 3}};
    const std::vector<Flags> flags{{false, false, 0}, {false, true, 0}, {true, false, 6},
                                   {true, true, 9}};
    for (const auto [m, k, n] : shapes) {
        for (const auto [requant, relu, shift] : flags) {
            SCOPED_TRACE(testing::Message() << m << "x" << k << "x" << n << " requant=" << requant
                                            << " relu=" << relu << " shift=" << shift);
            auto [fake, tpu] = make(/*busy_polls=*/1);
            std::mt19937 rng(static_cast<unsigned>(m * 100 + k * 10 + n));
            const Matrix<std::int8_t> a = random_int8(rng, m, k);
            const Matrix<std::int8_t> b = random_int8(rng, k, n);

            const auto [c, perf] = tpu.matmul(a, b, requant, relu, shift);
            EXPECT_EQ(c, matmul_ref(a, b, requant, relu, shift));
            const std::size_t kp = (k + 3) / 4 * 4, np = (n + 3) / 4 * 4;
            EXPECT_EQ(perf.busy_pe_cycles, m * kp * np);  // padded sizes, summed over the strips
        }
    }
}

TEST(TpuDriver, MatmulSplitsMIntoChunksOfMaxM) {
    HwConfig cfg;
    cfg.max_m = 8;
    auto owner = std::make_unique<FakeTpu>(cfg);
    FakeTpu* fake = owner.get();
    TpuDriver tpu(std::move(owner), cfg);
    const Matrix<std::int8_t> a(20, 4, std::vector<std::int8_t>(80, 1));
    const Matrix<std::int8_t> b(4, 4, std::vector<std::int8_t>(16, 1));
    const auto [c, perf] = tpu.matmul(a, b);
    EXPECT_TRUE(std::ranges::all_of(c, [](std::int64_t v) { return v == 4; }));
    EXPECT_EQ(std::ranges::count(fake->ops, Op::Matmul), 3);  // 8 + 8 + 4 rows
}

TEST(TpuDriver, MatmulRefusesMatricesThatDoNotFit) {
    auto [fake, tpu] = make();
    const Matrix<std::int8_t> big(64, 64);  // A and B need 2 * 4096 > 4096 entries
    EXPECT_THAT([&] { tpu.matmul(big, big); },
                ThrowsMessage<ProtocolError>(HasSubstr("does not fit")));
    EXPECT_TRUE(fake->ops.empty());  // refused before anything was sent
}

TEST(TpuDriver, MatmulRejectsBadShapes) {
    auto [fake, tpu] = make();
    const Matrix<std::int8_t> m(2, 3);
    EXPECT_THAT([&] { tpu.matmul(m, m); },
                ThrowsMessage<std::invalid_argument>(HasSubstr("cannot multiply 2x3 by 2x3")));
}

TEST(TpuDriver, FloatMatmulIsCloseToTheExactProduct) {
    std::mt19937 rng(3);
    std::normal_distribution<double> dist(0.0, 1.0);
    Matrix<double> a(10, 12), b(12, 6);
    for (double& v : a) v = dist(rng);
    for (double& v : b) v = dist(rng);

    // Exact double product, the reference.
    Matrix<double> exact(10, 6);
    for (std::size_t i = 0; i < 10; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            for (std::size_t p = 0; p < 12; ++p) {
                exact(i, j) += a(i, p) * b(p, j);
            }
        }
    }
    auto max_abs = [](const Matrix<double>& m) {
        return std::ranges::max(m, {}, [](double v) { return std::abs(v); });
    };

    for (const bool requant : {false, true}) {
        SCOPED_TRACE(requant ? "requant" : "no requant");
        auto [fake, tpu] = make(/*busy_polls=*/1);
        const FloatResult res = tpu.matmul_float(a, b, requant);
        if (requant) {
            EXPECT_GT(res.shift, 0);  // the requant path really ran
        } else {
            EXPECT_EQ(res.shift, 0);
        }

        // Each input is off by at most half a step (s/2), so each product by at
        // most |a| sb/2 + |b| sa/2 + sa sb/4, summed over K; requantization adds
        // at most half an output step.
        const double sa = choose_scale(a.data()), sb = choose_scale(b.data());
        const double per_product =
            std::abs(max_abs(a)) * sb / 2 + std::abs(max_abs(b)) * sa / 2 + sa * sb / 4;
        const double bound = 12 * per_product + (requant ? res.scale / 2 : 0);
        for (std::size_t i = 0; i < 10; ++i) {
            for (std::size_t j = 0; j < 6; ++j) {
                EXPECT_LE(std::abs(res.c(i, j) - exact(i, j)), bound) << i << "," << j;
            }
        }
    }
}

TEST(TpuDriver, WaitIdleTimesOutIfAlwaysBusy) {
    auto owner = std::make_unique<FakeTpu>(HwConfig{}, /*busy_polls=*/1'000'000);
    FakeTpu* fake = owner.get();
    TpuDriver tpu(std::move(owner), HwConfig{}, 20ms, 1ms);
    fake->write(Matmul{.m = 4, .k = 4, .n = 4, .ld_a = 4, .ld_b = 4, .ld_c = 4}.frame());
    EXPECT_THAT([&] { tpu.wait_idle(); }, ThrowsMessage<TpuError>(HasSubstr("still busy")));
}
