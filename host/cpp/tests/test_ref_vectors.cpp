// The shared reference vectors (src/test/resources/ref_vectors.txt), also
// checked by the Scala tests. Every line is checked: FM/FW/FR (protocol), Q/S/K
// (quant), P/M (reference). The format is in host/README.md.

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tpu_host/protocol.hpp"
#include "tpu_host/quant.hpp"
#include "tpu_host/reference.hpp"

using namespace tpu_host;

namespace {

Bytes from_hex(const std::string& hex) {
    Bytes out;
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        out.push_back(static_cast<std::uint8_t>(std::stoi(hex.substr(i, 2), nullptr, 16)));
    }
    return out;
}

}  // namespace

TEST(RefVectors, Frames) {
    std::ifstream file(REF_VECTORS);  // set by CMake
    ASSERT_TRUE(file) << "cannot open " << REF_VECTORS;

    int checked = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream in(line);
        std::string kind, hex;
        in >> kind;
        if (kind != "FM" && kind != "FW" && kind != "FR") {
            continue;  // Q S P K M: quant and reference model, step 4
        }
        in >> hex;
        const Bytes expected = from_hex(hex);
        SCOPED_TRACE(line);

        if (kind == "FM") {
            Matmul mm;
            int requant = 0, relu = 0;
            in >> mm.a_addr >> mm.b_addr >> mm.c_addr >> mm.m >> mm.k >> mm.n >> mm.ld_a >>
                mm.ld_b >> mm.ld_c >> requant >> relu >> mm.shift;
            mm.requant = requant != 0;
            mm.relu = relu != 0;
            EXPECT_EQ(mm.frame(), expected);
        } else if (kind == "FW") {
            Addr addr = 0;
            in >> addr;
            Bytes data;
            for (int v; in >> v;) {
                data.push_back(static_cast<std::uint8_t>(v));  // signed int8, kept modulo 256
            }
            EXPECT_EQ(write_frame(addr, data), expected);
        } else {  // FR
            Addr addr = 0;
            Dim count = 0;
            int size = 0;
            in >> addr >> count >> size;
            EXPECT_EQ(read_frame(addr, count, size), expected);
        }
        ++checked;
    }
    EXPECT_EQ(checked, 15);  // 8 FM + 4 FW + 3 FR: catches a wrong file or skipped lines
}

TEST(RefVectors, Quant) {
    std::ifstream file(REF_VECTORS);
    ASSERT_TRUE(file) << "cannot open " << REF_VECTORS;

    int checked = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream in(line);
        std::string kind;
        in >> kind;
        SCOPED_TRACE(line);

        // Floats were written with Python's repr(), the shortest text that reads
        // back to the same double, so the comparisons below are exact.
        if (kind == "Q") {  // Q x scale q
            std::string x, scale;
            int q = 0;
            in >> x >> scale >> q;
            EXPECT_EQ(quantize(std::stod(x), std::stod(scale)), q);
        } else if (kind == "S") {  // S scale x...
            std::string scale, x;
            in >> scale;
            std::vector<double> xs;
            while (in >> x) {
                xs.push_back(std::stod(x));
            }
            EXPECT_EQ(choose_scale(xs), std::stod(scale));
        } else if (kind == "K") {  // K shift acc...
            int shift = 0;
            in >> shift;
            std::vector<std::int64_t> acc;
            for (std::int64_t v; in >> v;) {
                acc.push_back(v);
            }
            EXPECT_EQ(choose_shift(acc), shift);
        } else {
            continue;
        }
        ++checked;
    }
    EXPECT_EQ(checked, 94 + 8 + 15);  // Q + S + K lines
}

TEST(RefVectors, Reference) {
    std::ifstream file(REF_VECTORS);
    ASSERT_TRUE(file) << "cannot open " << REF_VECTORS;

    int checked = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream in(line);
        std::string kind;
        in >> kind;
        SCOPED_TRACE(line);

        if (kind == "P") {  // P acc requant relu shift y
            std::int64_t acc = 0, y = 0;
            int requant = 0, relu = 0, shift = 0;
            in >> acc >> requant >> relu >> shift >> y;
            EXPECT_EQ(post_process(acc, requant != 0, relu != 0, shift), y);
        } else if (kind == "M") {  // M m k n requant relu shift | a... | b... | c...
            std::size_t m = 0, k = 0, n = 0;
            int requant = 0, relu = 0, shift = 0;
            in >> m >> k >> n >> requant >> relu >> shift;
            // Read word by word: >> into an int would stop at each "|".
            std::vector<std::int64_t> parts[3];
            int part = -1;
            for (std::string word; in >> word;) {
                if (word == "|") {
                    ++part;
                } else {
                    parts[part].push_back(std::stoll(word));
                }
            }
            auto to_int8 = [](const std::vector<std::int64_t>& v) {
                return std::vector<std::int8_t>(v.begin(), v.end());
            };
            const Matrix<std::int8_t> a(m, k, to_int8(parts[0]));
            const Matrix<std::int8_t> b(k, n, to_int8(parts[1]));
            const Matrix<std::int64_t> c(m, n, parts[2]);
            EXPECT_EQ(matmul_ref(a, b, requant != 0, relu != 0, shift), c);
        } else {
            continue;
        }
        ++checked;
    }
    EXPECT_EQ(checked, 792 + 30);  // P + M lines
}
