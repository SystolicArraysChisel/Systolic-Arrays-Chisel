#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tpu_host/types.hpp"

namespace tpu_host {
enum class Op : std::uint8_t { Write = 0x01, Read = 0x02, Matmul = 0x03, Status = 0x04, Bias = 0x05 };

struct HwConfig;

inline constexpr std::size_t FIELD_BYTES = 2;
inline constexpr int MAX_FIELD =
    0xFFFF;  // largest value of a 2-byte field (addr, count, dimension)

inline constexpr std::size_t MATMUL_FRAME_BYTES = 23;  // opcode + 22 payload bytes
inline constexpr int REQUANT_BIT = 0;                  // bit positions in the MATMUL flags byte
inline constexpr int RELU_BIT = 1;
inline constexpr int TO_INPUT_BIT = 2;  // write C back into the input memory
inline constexpr int BIAS_BIT = 3;      // start the accumulators from the bias memory
inline constexpr int LEAK_SHIFT_BIT = 4;  // bits 4 to 6: leaky ReLU shift
inline constexpr int LEAK_SHIFT_MAX = 7;

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
// int32 entries addr, addr + 1, ... of the bias memory, little-endian.
Bytes bias_frame(Addr addr, std::span<const std::int32_t> data);

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
    // Write C (requantized, int8) into the input memory at c_addr instead of the
    // result memory, so the next MATMUL can use it as A without going through
    // the host. Needs requant.
    bool to_input = false;
    int shift = 0;
    // Start each accumulator from bias[bias_addr + j] (int32, loaded with BIAS)
    // instead of 0; the sum wraps in acc_width bits. bias_addr must be 0 without bias.
    bool bias = false;
    int bias_addr = 0;
    // Leaky ReLU: with relu, a negative value x becomes x >> leak_shift (floor)
    // instead of 0. 0 is the plain ReLU; must be 0 without relu.
    int leak_shift = 0;

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