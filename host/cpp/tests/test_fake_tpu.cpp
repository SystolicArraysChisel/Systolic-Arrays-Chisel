// FakeTpu driven directly with the frame builders, without Tpu.

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include "tpu_host/fake_tpu.hpp"
#include "tpu_host/matrix.hpp"
#include "tpu_host/protocol.hpp"
#include "tpu_host/reference.hpp"

using namespace tpu_host;

namespace {

Bytes as_bytes(const std::vector<int>& values) {
    Bytes out;
    for (int v : values) {
        out.push_back(static_cast<std::uint8_t>(v));  // int8 kept modulo 256
    }
    return out;
}

Status status_of(FakeTpu& fake) {
    fake.write(status_frame());
    return parse_status(fake.read(STATUS_REPLY_BYTES));
}

}  // namespace

TEST(FakeTpu, WriteStoresSignedBytes) {
    FakeTpu fake;
    fake.write(write_frame(10, as_bytes({1, -1, 127, -128})));
    EXPECT_EQ(fake.inp[10], 1);
    EXPECT_EQ(fake.inp[11], -1);
    EXPECT_EQ(fake.inp[12], 127);
    EXPECT_EQ(fake.inp[13], -128);
    EXPECT_EQ(fake.ops, std::vector<Op>{Op::Write});
}

TEST(FakeTpu, ReadReturnsFullWordsOrLowBytes) {
    FakeTpu fake;
    fake.res[0] = 5;
    fake.res[1] = -2;
    fake.write(read_frame(0, 2, 4));
    EXPECT_EQ(fake.read(100), as_bytes({5, 0, 0, 0, 0xFE, 0xFF, 0xFF, 0xFF}));
    fake.write(read_frame(0, 2, 1));
    EXPECT_EQ(fake.read(100), as_bytes({5, 0xFE}));
}

TEST(FakeTpu, WaitsForTheRestOfASplitFrame) {
    FakeTpu fake;
    const Bytes frame = write_frame(0, as_bytes({7, 8, 9}));
    const ByteView view = frame;
    fake.write(view.first(4));  // count not even complete
    EXPECT_TRUE(fake.ops.empty());
    fake.write(view.subspan(4, 2));
    EXPECT_TRUE(fake.ops.empty());
    fake.write(view.subspan(6));
    EXPECT_EQ(fake.ops, std::vector<Op>{Op::Write});
    EXPECT_EQ(fake.inp[2], 9);
}

TEST(FakeTpu, DecodesSeveralFramesInOneWrite) {
    FakeTpu fake;
    Bytes stream = write_frame(0, as_bytes({3}));
    const Bytes status = status_frame();
    stream.insert(stream.end(), status.begin(), status.end());
    fake.write(stream);
    EXPECT_EQ(fake.ops, (std::vector<Op>{Op::Write, Op::Status}));
    EXPECT_EQ(fake.read(100).size(), STATUS_REPLY_BYTES);
}

TEST(FakeTpu, WritesWrapAroundTheMemory) {
    FakeTpu fake;
    const int last = HwConfig{}.in_depth - 1;
    fake.write(write_frame(last, as_bytes({4, 5})));
    EXPECT_EQ(fake.inp[static_cast<std::size_t>(last)], 4);
    EXPECT_EQ(fake.inp[0], 5);
}

TEST(FakeTpu, MatmulMatchesTheReference) {
    // A 3x4 at 0 with ld_a = 6 (padded rows), B 4x2 at 100 with ld_b = 2, C at 50 with ld_c = 3.
    const Matrix<std::int8_t> a(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 1, 0, -1, 2});
    const Matrix<std::int8_t> b(4, 2, {1, 0, 0, 1, 1, 1, 2, -1});
    FakeTpu fake;
    for (int i = 0; i < 3; ++i) {
        const Bytes row(a.data().begin() + 4 * i, a.data().begin() + 4 * i + 4);
        fake.write(write_frame(6 * i, row));
    }
    fake.write(write_frame(100, Bytes(b.data().begin(), b.data().end())));
    fake.write(Matmul{.a_addr = 0, .b_addr = 100, .c_addr = 50, .m = 3, .k = 4, .n = 2,
                      .ld_a = 6, .ld_b = 2, .ld_c = 3}
                   .frame());

    const Matrix<std::int64_t> c = matmul_ref(a, b);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(fake.res[50 + 3 * i + j], c(i, j)) << i << "," << j;
        }
    }
    EXPECT_EQ(fake.busy_pe_cycles, 3u * 4u * 2u);
}

TEST(FakeTpu, MatmulAppliesRequantAndRelu) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(-128, 127);
    Matrix<std::int8_t> a(4, 8), b(8, 4);
    for (auto& v : a) v = static_cast<std::int8_t>(dist(rng));
    for (auto& v : b) v = static_cast<std::int8_t>(dist(rng));
    FakeTpu fake;
    fake.write(write_frame(0, Bytes(a.data().begin(), a.data().end())));
    fake.write(write_frame(32, Bytes(b.data().begin(), b.data().end())));
    fake.write(Matmul{.a_addr = 0, .b_addr = 32, .c_addr = 0, .m = 4, .k = 8, .n = 4,
                      .ld_a = 8, .ld_b = 4, .ld_c = 4, .requant = true, .relu = true, .shift = 6}
                   .frame());
    const Matrix<std::int64_t> c = matmul_ref(a, b, true, true, 6);
    EXPECT_EQ(std::vector<std::int64_t>(fake.res.begin(), fake.res.begin() + 16), c.data());
}

TEST(FakeTpu, StatusIsBusyForBusyPollsAfterAMatmul) {
    FakeTpu fake(HwConfig{}, /*busy_polls=*/2);
    EXPECT_FALSE(status_of(fake).busy);
    fake.write(Matmul{.m = 1, .k = 4, .n = 4, .ld_a = 4, .ld_b = 4, .ld_c = 4}.frame());
    const Status s = status_of(fake);
    EXPECT_TRUE(s.busy);
    EXPECT_EQ(s.busy_pe_cycles, 16u);
    EXPECT_GT(s.cycles, 0u);
    EXPECT_TRUE(status_of(fake).busy);
    EXPECT_FALSE(status_of(fake).busy);
}

TEST(FakeTpu, ErrorsAreStickyUntilStatus) {
    FakeTpu fake;
    fake.write(Bytes{0x7F});  // unknown opcode
    EXPECT_TRUE(status_of(fake).error);
    EXPECT_FALSE(status_of(fake).error);  // cleared by the previous STATUS

    Bytes bad_read = read_frame(0, 1, 4);
    bad_read[5] = 2;  // read_frame refuses size 2, so patch it in
    fake.write(bad_read);
    EXPECT_TRUE(fake.read(100).empty());
    EXPECT_TRUE(status_of(fake).error);

    Bytes bad_shift = Matmul{.m = 1, .k = 4, .n = 4, .ld_a = 4, .ld_b = 4, .ld_c = 4}.frame();
    bad_shift[20] = 32;  // >= acc_width
    fake.write(bad_shift);
    EXPECT_TRUE(status_of(fake).error);
}
