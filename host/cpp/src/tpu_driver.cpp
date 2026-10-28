#include "tpu_host/tpu_driver.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "tpu_host/protocol.hpp"
#include "tpu_host/quant.hpp"
#include "tpu_host/reference.hpp"
#include "tpu_host/serial_link.hpp"

namespace tpu_host {
namespace {
std::uint32_t read_le32(ByteView bytes, std::size_t offset) {
    return static_cast<std::uint32_t>(bytes[offset]) |
           static_cast<std::uint32_t>(bytes[offset + 1]) << 8 |
           static_cast<std::uint32_t>(bytes[offset + 2]) << 16 |
           static_cast<std::uint32_t>(bytes[offset + 3]) << 24;
}

// Smallest multiple of `multiple` >= x: adding multiple - 1 turns the
// truncating division into a rounding-up one.
int round_up(int x, int multiple) { return (x + multiple - 1) / multiple * multiple; }

std::string shape(int rows, int cols) { return std::to_string(rows) + "x" + std::to_string(cols); }
}  // namespace

TpuDriver::TpuDriver(std::unique_ptr<Link> link, HwConfig cfg, std::chrono::milliseconds timeout,
                     std::chrono::milliseconds poll_interval)
    : link_(std::move(link)),
      cfg_(std::move(cfg)),
      timeout_(timeout),
      poll_interval_(poll_interval) {}

TpuDriver TpuDriver::open(const std::string& port, HwConfig cfg) {
    auto link = std::make_unique<SerialLink>(port, cfg.baud);
    return TpuDriver(std::move(link), std::move(cfg));
}

void TpuDriver::send(ByteView frame) { link_->write(frame); }

Bytes TpuDriver::recv(std::size_t n) {
    Bytes data;
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (data.size() < n) {
        const Bytes chunk = link_->read(n - data.size());
        if (!chunk.empty()) {
            data.insert(data.end(), chunk.begin(), chunk.end());
        } else if (std::chrono::steady_clock::now() > deadline) {
            throw TpuError("timeout: got " + std::to_string(data.size()) + " of " +
                           std::to_string(n) + " reply bytes");
        }
    }
    return data;
}
Status TpuDriver::status() {
    send(status_frame());
    return parse_status(recv(STATUS_REPLY_BYTES));
}
Status TpuDriver::check() {
    Status current_status = status();
    if (current_status.error || current_status.overrun) {
        throw TpuError(std::string("accelerator reported error=") +
                       (current_status.error ? "true" : "false") +
                       " overrun=" + (current_status.overrun ? "true" : "false") +
                       ": reset it and resend the matrices");
    }
    return current_status;
}
Status TpuDriver::wait_idle() {
    const auto deadline = std::chrono::steady_clock::now() + timeout_;
    while (true) {
        const Status s = check();  // check, not status: an error must stop the wait
        if (!s.busy) return s;
        if (std::chrono::steady_clock::now() > deadline) throw TpuError("timeout: still busy");
        std::this_thread::sleep_for(poll_interval_);
    }
}

void TpuDriver::write(Addr addr, std::span<const std::int8_t> data) {
    const int len = static_cast<int>(data.size());
    if (addr < 0 || addr + len > cfg_.in_depth) {
        throw ProtocolError("WRITE of " + std::to_string(len) + " elements at " +
                            std::to_string(addr) + " leaves the input memory");
    }
    for (int off = 0; off < len; off += MAX_FIELD) {
        const auto chunk = data.subspan(static_cast<std::size_t>(off),
                                        static_cast<std::size_t>(std::min(MAX_FIELD, len - off)));
        // Same bits, seen as unsigned: -1 goes on the wire as 0xFF.
        const ByteView bytes(reinterpret_cast<const std::uint8_t*>(chunk.data()), chunk.size());
        send(write_frame(addr + off, bytes));
        check();  // after each WRITE: a set flag then points to this transfer
    }
}

std::vector<std::int64_t> TpuDriver::read(Addr addr, Dim count, int size) {
    if (addr < 0 || addr + count > cfg_.out_depth) {
        throw ProtocolError("READ of " + std::to_string(count) + " elements at " +
                            std::to_string(addr) + " leaves the result memory");
    }
    std::vector<std::int64_t> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int off = 0; off < count; off += MAX_FIELD) {
        const int n = std::min(MAX_FIELD, count - off);
        check();  // before each READ
        send(read_frame(addr + off, n, size));
        const Bytes reply = recv(static_cast<std::size_t>(n * size));
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
            if (size == 1) {
                out.push_back(static_cast<std::int8_t>(reply[i]));
            } else {
                // Through int32_t, so bit 31 is the sign: 0xFFFFFFFE -> -2, not 4294967294.
                out.push_back(static_cast<std::int32_t>(read_le32(reply, 4 * i)));
            }
        }
    }
    return out;
}

Perf TpuDriver::run(const Matmul& mm) {
    mm.check(cfg_);  // throws before anything is sent
    send(mm.frame());
    const Status s = wait_idle();
    return Perf{s.cycles, s.busy_pe_cycles};
}

IntResult TpuDriver::matmul(const Matrix<std::int8_t>& a, const Matrix<std::int8_t>& b,
                            bool requant, bool relu, int shift) {
    const int m = static_cast<int>(a.rows());
    const int k = static_cast<int>(a.cols());
    const int n = static_cast<int>(b.cols());
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("cannot multiply " + shape(m, k) + " by " +
                                    shape(static_cast<int>(b.rows()), n));
    }

    // 1. Pad K and N to multiples of banks; M needs no padding (any m <= max_m is fine).
    const int banks = cfg_.banks();
    const int kp = round_up(k, banks);
    const int np = round_up(n, banks);

    // 2. Layout: A at 0, B right after it, C at 0 of the result memory. Checked
    //    before anything is sent.
    const Addr a_addr = 0;
    const Addr b_addr = m * kp;
    const Addr c_addr = 0;
    if (b_addr + kp * np > cfg_.in_depth || m * np > cfg_.out_depth) {
        throw ProtocolError(shape(m, k) + " . " + shape(k, n) + " does not fit in the memories");
    }

    // 3. Zero-padded copies: the extra zeros add nothing to the sums, and the
    //    extra columns of C are cropped in step 6.
    const auto um = static_cast<std::size_t>(m), uk = static_cast<std::size_t>(k),
               un = static_cast<std::size_t>(n), ukp = static_cast<std::size_t>(kp),
               unp = static_cast<std::size_t>(np);
    Matrix<std::int8_t> a_pad(um, ukp);
    Matrix<std::int8_t> b_pad(ukp, unp);
    for (std::size_t i = 0; i < um; ++i) {
        for (std::size_t j = 0; j < uk; ++j) {
            a_pad(i, j) = a(i, j);
        }
    }
    for (std::size_t i = 0; i < uk; ++i) {
        for (std::size_t j = 0; j < un; ++j) {
            b_pad(i, j) = b(i, j);
        }
    }

    // 4. Upload: a Matrix is row-major and contiguous, exactly the memory layout.
    write(a_addr, a_pad.data());
    write(b_addr, b_pad.data());

    // 5. Compute in strips of at most max_m rows. Each strip of C needs the
    //    same rows of A and C shifted by r0, but the whole of B.
    Perf perf;
    for (int r0 = 0; r0 < m; r0 += cfg_.max_m) {
        const int rows = std::min(cfg_.max_m, m - r0);
        perf += run(Matmul{.a_addr = a_addr + r0 * kp,
                           .b_addr = b_addr,
                           .c_addr = c_addr + r0 * np,
                           .m = rows,
                           .k = kp,
                           .n = np,
                           .ld_a = kp,
                           .ld_b = np,
                           .ld_c = np,
                           .requant = requant,
                           .relu = relu,
                           .shift = shift});
    }

    // 6. Download and crop. With requant every value fits in one byte: 4x less
    //    UART traffic.
    const std::vector<std::int64_t> flat = read(c_addr, m * np, requant ? 1 : 4);
    Matrix<std::int64_t> c(um, un);
    for (std::size_t i = 0; i < um; ++i) {
        for (std::size_t j = 0; j < un; ++j) {
            c(i, j) = flat[i * unp + j];  // row i of the padded result starts at i * np
        }
    }
    return {std::move(c), perf};
}

FloatResult TpuDriver::matmul_float(const Matrix<double>& a, const Matrix<double>& b, bool requant,
                                    bool relu) {
    // 1. One scale per matrix: the largest |x| maps to 127
    const double sa = choose_scale(a.data()), sb = choose_scale(b.data());
    // 2. Quantize to int8
    const Matrix<std::int8_t> qa = quantize(a, sa), qb = quantize(b, sb);
    // 3. With requant, the smallest shift that saturates nothing (needs the exact accumulators)
    const int shift =
        requant ? choose_shift(matmul_acc(qa, qb, cfg_.acc_width).data(), cfg_.acc_width) : 0;
    // 4. Integer product on the FPGA
    auto [c_q, perf] = matmul(qa, qb, requant, relu, shift);
    // 5. Back to reals
    const double scale = requant_scale(sa, sb, shift);
    return {dequantize(c_q, scale), scale, shift, perf};
}

Perf& Perf::operator+=(const Perf& other) {
    cycles += other.cycles;
    busy_pe_cycles += other.busy_pe_cycles;
    return *this;
}

double Perf::utilization(const HwConfig& cfg) const {
    if (cycles == 0) {
        return 0.0;
    }
    return static_cast<double>(busy_pe_cycles) /
           (static_cast<double>(cycles) * cfg.rows * cfg.cols);
}
}  // namespace tpu_host
