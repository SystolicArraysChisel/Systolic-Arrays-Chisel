#include "tpu_host/protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tpu_host/config.hpp"

namespace tpu_host {
namespace {
void append_le16(Bytes& out, int value) {
    if (value < 0 || value > MAX_FIELD) {
        throw ProtocolError(std::to_string(value) + " does not fit in a 2-byte field");
    }
    out.push_back(static_cast<std::uint8_t>(value & 0xFF));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
}
std::uint32_t read_le32(ByteView bytes, std::size_t offset) {
    return (static_cast<uint32_t>(bytes[offset]) | static_cast<uint32_t>(bytes[offset + 1]) << 8 |
            static_cast<uint32_t>(bytes[offset + 2]) << 16 |
            static_cast<uint32_t>(bytes[offset + 3]) << 24);
}
}  // namespace
Bytes status_frame() { return Bytes{static_cast<std::uint8_t>(Op::Status)}; }

Bytes write_frame(Addr addr, ByteView data) {
    Bytes frame;
    frame.reserve(sizeof(Op) + 2 * FIELD_BYTES + data.size());
    frame.push_back(static_cast<std::uint8_t>(Op::Write));
    append_le16(frame, addr);
    append_le16(frame, static_cast<int>(data.size()));
    frame.insert(frame.end(), data.begin(), data.end());
    return frame;
}

Bytes read_frame(Addr addr, Dim count, int size) {
    if (!((size == 1) || (size == 4))) {
        throw ProtocolError("READ size must be 1 or 4, got " + std::to_string(size));
    }
    Bytes frame;
    frame.reserve(sizeof(Op) + 2 * FIELD_BYTES +
                  1);  // Opcode (1) + addr (2) + count (2) + size (1)
    frame.push_back(static_cast<std::uint8_t>(Op::Read));
    append_le16(frame, addr);
    append_le16(frame, count);
    frame.push_back(static_cast<std::uint8_t>(size));
    return frame;
}
Bytes Matmul::frame() const {
    Bytes frame;
    frame.reserve(MATMUL_FRAME_BYTES);
    frame.push_back(static_cast<std::uint8_t>(Op::Matmul));
    append_le16(frame, a_addr);
    append_le16(frame, b_addr);
    append_le16(frame, c_addr);
    append_le16(frame, m);
    append_le16(frame, k);
    append_le16(frame, n);
    append_le16(frame, ld_a);
    append_le16(frame, ld_b);
    append_le16(frame, ld_c);
    int flags = (requant << REQUANT_BIT) | (relu << RELU_BIT);
    frame.push_back(static_cast<std::uint8_t>(flags));
    frame.push_back(static_cast<std::uint8_t>(shift));
    return frame;
}
Status parse_status(ByteView reply) {
    if (reply.size() != STATUS_REPLY_BYTES) {
        throw ProtocolError("STATUS reply has " + std::to_string(reply.size()) +
                            " bytes, expected " + std::to_string(STATUS_REPLY_BYTES));
    }
    const std::uint8_t flags = reply[0];
    return Status{
        .busy = ((flags >> BUSY_BIT) & 1) != 0,
        .error = ((flags >> ERROR_BIT) & 1) != 0,
        .overrun = ((flags >> OVERRUN_BIT) & 1) != 0,
        .cycles = read_le32(reply, 1),
        .busy_pe_cycles = read_le32(reply, 5),
    };
}
void Matmul::check(const HwConfig& cfg) const {
    const int banks = cfg.banks();
    const int cols = cfg.cols;
    std::vector<std::string> broken_rules;

    auto need = [&broken_rules](bool ok, std::string msg) {
        if (!ok) {
            broken_rules.push_back(msg);
        }
    };
    const std::string b = std::to_string(banks);
    need(1 <= m && m <= cfg.max_m,
         "m = " + std::to_string(m) + " must be in 1.." + std::to_string(cfg.max_m));
    need(k > 0 && k % banks == 0,
         "k = " + std::to_string(k) + " must be a positive multiple of banks (" + b + ")");
    need(n > 0 && n % banks == 0,
         "n = " + std::to_string(n) + " must be a positive multiple of banks (" + b + ")");
    need(ld_a >= k, "ldA = " + std::to_string(ld_a) + " < k = " + std::to_string(k));
    need(ld_b >= n, "ldB = " + std::to_string(ld_b) + " < n = " + std::to_string(n));
    need(ld_c >= n, "ldC = " + std::to_string(ld_c) + " < n = " + std::to_string(n));
    need(a_addr % banks == 0 && b_addr % banks == 0,
         "aAddr and bAddr must be multiples of banks (" + b + ")");
    need(ld_a % banks == 0 && ld_b % banks == 0,
         "ldA and ldB must be multiples of banks (" + b + ")");
    need(c_addr % cols == 0 && ld_c % cols == 0,
         "cAddr and ldC must be multiples of cols (" + std::to_string(cols) + ")");
    need(a_addr + (m - 1) * ld_a + k <= cfg.in_depth, "A does not fit in the input memory");
    need(b_addr + (k - 1) * ld_b + n <= cfg.in_depth, "B does not fit in the input memory");
    need(c_addr + (m - 1) * ld_c + n <= cfg.out_depth, "C does not fit in the result memory");
    need(
        0 <= shift && shift < cfg.acc_width,
        "shift = " + std::to_string(shift) + " must be in 0.." + std::to_string(cfg.acc_width - 1));

    if (broken_rules.empty()) {
        return;
    }
    std::string joined = broken_rules[0];
    for (std::size_t i = 1; i < broken_rules.size(); ++i) {
        joined += "; " + broken_rules[i];
    }
    throw ProtocolError(joined);
}

}  // namespace tpu_host