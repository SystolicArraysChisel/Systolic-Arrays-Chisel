#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tpu_host/types.hpp"

namespace tpu_host {
enum class Op : std::uint8_t { Write = 0x01, Read = 0x02, Matmul = 0x03, Status = 0x04 };

struct HwConfig;

inline constexpr std::size_t FIELD_BYTES = 2;
inline constexpr int MAX_FIELD =
    0xFFFF;  // largest value of a 2-byte field (addr, count, dimension)

inline constexpr std::size_t MATMUL_FRAME_BYTES = 21;  // opcode + 20 payload bytes
inline constexpr int REQUANT_BIT = 0;                  // bit positions in the MATMUL flags byte
inline constexpr int RELU_BIT = 1;

inline constexpr std::size_t STATUS_REPLY_BYTES = 9;
inline constexpr int BUSY_BIT = 0;  // bit positions in the STATUS flags byte
inline constexpr int ERROR_BIT = 1;
inline constexpr int OVERRUN_BIT = 2;

using Bytes = std::vector<std::uint8_t>;
using ByteView = std::span<const std::uint8_t>;

class ProtocolError : public std::invalid_argument {
   public:
    using std::invalid_argument::invalid_argument;
};

Bytes write_frame(Addr addr, ByteView data);
Bytes read_frame(Addr addr, Dim count, int size);
Bytes status_frame();

struct Matmul {
    Addr a_addr = 0;
    Addr b_addr = 0;
    Addr c_addr = 0;
    Dim m = 0;
    Dim k = 0;
    Dim n = 0;
    Dim ld_a = 0;
    Dim ld_b = 0;
    Dim ld_c = 0;
    bool requant = false;
    bool relu = false;
    int shift = 0;

    Bytes frame() const;
    void check(const HwConfig& cfg) const;
};
struct Status {
    bool busy = false;
    bool error = false;
    bool overrun = false;
    std::uint32_t cycles = 0;          // cycles of the last MATMUL (so far, while busy)
    std::uint32_t busy_pe_cycles = 0;  // m * k * n of the last MATMUL
};

Status parse_status(ByteView reply);
}  // namespace tpu_host