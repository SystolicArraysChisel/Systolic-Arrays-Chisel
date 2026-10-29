#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "tpu_host/config.hpp"
#include "tpu_host/protocol.hpp"

using namespace tpu_host;
using testing::HasSubstr;
using testing::StrEq;
using testing::ThrowsMessage;

namespace {

const HwConfig CFG;

// A valid product: 4x8 . 8x4, A at 0, B right after it, C at 0.
Matmul dense() {
    return Matmul{.a_addr = 0, .b_addr = 64, .c_addr = 0, .m = 4, .k = 8, .n = 4,
                  .ld_a = 8, .ld_b = 4, .ld_c = 4};
}

}  // namespace

// ---- simple frames ----

TEST(Protocol, StatusFrame) { EXPECT_EQ(status_frame(), Bytes{0x04}); }

TEST(Protocol, WriteFrame) {
    EXPECT_EQ(write_frame(300, Bytes{5, 0xFF, 0x80}),
              (Bytes{0x01, 0x2C, 0x01, 0x03, 0x00, 0x05, 0xFF, 0x80}));
    EXPECT_EQ(write_frame(0x1234, Bytes{1, 0xFF}),
              (Bytes{0x01, 0x34, 0x12, 0x02, 0x00, 0x01, 0xFF}));
}

TEST(Protocol, WriteFrameRejectsAnAddressOver2Bytes) {
    EXPECT_THAT([] { write_frame(70000, Bytes{}); },
                ThrowsMessage<ProtocolError>(StrEq("70000 does not fit in a 2-byte field")));
}

TEST(Protocol, ReadFrame) {
    EXPECT_EQ(read_frame(0x0102, 3, 4), (Bytes{0x02, 0x02, 0x01, 0x03, 0x00, 0x04}));
    EXPECT_EQ(read_frame(16, 1000, 4), (Bytes{0x02, 0x10, 0x00, 0xE8, 0x03, 0x04}));
}

TEST(Protocol, ReadFrameRejectsASizeOtherThan1Or4) {
    EXPECT_THAT([] { read_frame(0, 1, 2); },
                ThrowsMessage<ProtocolError>(StrEq("READ size must be 1 or 4, got 2")));
}

// ---- MATMUL frame ----

TEST(Protocol, MatmulFrameFollowsTheByteTable) {
    // Every field has distinct bytes, so a swapped pair or byte order shows up.
    const Matmul mm{.a_addr = 0x0102, .b_addr = 0x0304, .c_addr = 0x0506, .m = 0x0708,
                    .k = 0x090a, .n = 0x0b0c, .ld_a = 0x0d0e, .ld_b = 0x0f10, .ld_c = 0x1112,
                    .requant = true, .relu = true, .shift = 9};
    const Bytes f = mm.frame();
    ASSERT_EQ(f.size(), MATMUL_FRAME_BYTES);
    EXPECT_EQ(f[0], static_cast<std::uint8_t>(Op::Matmul));
    const std::vector<std::pair<std::size_t, int>> fields = {
        {1, 0x0102},  {3, 0x0304},  {5, 0x0506},  {7, 0x0708},  {9, 0x090a},
        {11, 0x0b0c}, {13, 0x0d0e}, {15, 0x0f10}, {17, 0x1112},
    };
    for (const auto& [off, value] : fields) {
        SCOPED_TRACE("offset " + std::to_string(off));
        EXPECT_EQ(f[off] | f[off + 1] << 8, value);  // low byte first
    }
    EXPECT_EQ(f[19], 0b11);  // requant and relu
    EXPECT_EQ(f[20], 9);     // shift
}

TEST(Protocol, MatmulFrameReference) {
    const Matmul mm{.a_addr = 0, .b_addr = 256, .c_addr = 0, .m = 4, .k = 8, .n = 12,
                    .ld_a = 8, .ld_b = 12, .ld_c = 12, .requant = true, .relu = true,
                    .shift = 7};
    const Bytes expected{0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x04, 0x00, 0x08, 0x00,
                         0x0c, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x0c, 0x00, 0x03, 0x07};
    EXPECT_EQ(mm.frame(), expected);
}

// ---- STATUS reply ----

TEST(Protocol, ParseStatus) {
    const Status s = parse_status(Bytes{0b111, 0xEF, 0xBE, 0xAD, 0xDE, 64, 0, 0, 0});
    EXPECT_TRUE(s.busy);
    EXPECT_TRUE(s.error);
    EXPECT_TRUE(s.overrun);
    EXPECT_EQ(s.cycles, 0xDEADBEEFu);
    EXPECT_EQ(s.busy_pe_cycles, 64u);
}

TEST(Protocol, ParseStatusWithHighBytes) {
    // Bytes >= 0x80 catch a shift into the sign bit in read_le32.
    const Status s = parse_status(Bytes{0x05, 0x10, 0x27, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF});
    EXPECT_TRUE(s.busy);
    EXPECT_FALSE(s.error);
    EXPECT_TRUE(s.overrun);
    EXPECT_EQ(s.cycles, 10000u);
    EXPECT_EQ(s.busy_pe_cycles, 4294967295u);
}

TEST(Protocol, ParseStatusRejectsAShortReply) {
    EXPECT_THAT([] { parse_status(Bytes(8)); },
                ThrowsMessage<ProtocolError>(StrEq("STATUS reply has 8 bytes, expected 9")));
}

// ---- MATMUL check ----

TEST(MatmulCheck, AcceptsADenseProduct) {
    EXPECT_NO_THROW(dense().check(CFG));
    Matmul mm = dense();
    mm.m = CFG.max_m;
    mm.shift = 31;
    mm.requant = true;
    EXPECT_NO_THROW(mm.check(CFG));
}

TEST(MatmulCheck, RejectsBrokenGuarantees) {
    struct Case {
        Matmul mm;
        std::string reason;
    };
    std::vector<Case> cases;
    auto add = [&cases](auto change, std::string reason) {
        Matmul mm = dense();
        change(mm);
        cases.push_back({mm, std::move(reason)});
    };
    add([](Matmul& mm) { mm.m = 0; }, "m");
    add([](Matmul& mm) { mm.m = CFG.max_m + 1; }, "m");
    add([](Matmul& mm) { mm.k = 6; mm.ld_a = 6; }, "k");
    add([](Matmul& mm) { mm.n = 2; mm.ld_b = 2; mm.ld_c = 2; }, "n");
    add([](Matmul& mm) { mm.shift = 32; }, "shift");
    add([](Matmul& mm) { mm.ld_a = 4; mm.ld_b = 4; }, "ldA");
    add([](Matmul& mm) { mm.a_addr = 2; }, "aAddr");
    add([](Matmul& mm) { mm.c_addr = 2; }, "cAddr");
    add([](Matmul& mm) { mm.b_addr = 4090; }, "B does not fit");

    for (const Case& c : cases) {
        SCOPED_TRACE(c.reason);
        EXPECT_THAT([&c] { c.mm.check(CFG); }, ThrowsMessage<ProtocolError>(HasSubstr(c.reason)));
    }
}

TEST(MatmulCheck, ReportsEveryBrokenRule) {
    const Matmul mm{.a_addr = 2, .m = 4, .k = 6, .n = 4, .ld_a = 4, .ld_b = 4, .ld_c = 2,
                    .shift = 32};
    EXPECT_THAT(
        [&mm] { mm.check(CFG); },
        ThrowsMessage<ProtocolError>(StrEq(
            "k = 6 must be a positive multiple of banks (4); ldA = 4 < k = 6; ldC = 2 < n = 4; "
            "aAddr and bAddr must be multiples of banks (4); cAddr and ldC must be multiples "
            "of cols (4); shift = 32 must be in 0..31")));
}

TEST(MatmulCheck, ExactMessages) {
    const Matmul zero_m{.m = 0, .k = 4, .n = 4, .ld_a = 4, .ld_b = 4, .ld_c = 4};
    EXPECT_THAT([&zero_m] { zero_m.check(CFG); },
                ThrowsMessage<ProtocolError>(StrEq("m = 0 must be in 1..64")));

    const Matmul a_too_far{.a_addr = 4000, .m = 64, .k = 64, .n = 4, .ld_a = 64, .ld_b = 4,
                           .ld_c = 4};
    EXPECT_THAT([&a_too_far] { a_too_far.check(CFG); },
                ThrowsMessage<ProtocolError>(StrEq("A does not fit in the input memory")));
}
