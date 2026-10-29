#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "tpu_host/config.hpp"
#include "tpu_host/link.hpp"
#include "tpu_host/matrix.hpp"
#include "tpu_host/protocol.hpp"

namespace tpu_host {
class TpuError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};
struct Perf {
    std::uint32_t cycles = 0;
    std::uint32_t busy_pe_cycles = 0;
    Perf& operator+=(const Perf&);
    double utilization(const HwConfig&) const;
};
struct FloatResult {
    Matrix<double> c;
    double scale;
    int shift;
    Perf perf;
};
struct IntResult {
    Matrix<std::int64_t> c;
    Perf perf;
};
class TpuDriver {
   public:
    TpuDriver(std::unique_ptr<Link> link, HwConfig cfg = {},
              std::chrono::milliseconds timeout = std::chrono::seconds(5),
              std::chrono::milliseconds poll_interval = {});
    // The board on a COM port ("COM5"), at cfg.baud.
    static TpuDriver open(const std::string& port, HwConfig cfg = {});
    Status status();
    Status check();
    Status wait_idle();
    void write(Addr addr, std::span<const std::int8_t> data);
    std::vector<std::int64_t> read(Addr addr, Dim count, int size = 4);
    Perf run(const Matmul& mm);
    IntResult matmul(const Matrix<std::int8_t>& a, const Matrix<std::int8_t>& b,
                     bool requant = false, bool relu = false, int shift = 0);
    FloatResult matmul_float(const Matrix<double>& a, const Matrix<double>& b, bool requant = false,
                             bool relu = false);
    const HwConfig& cfg() const { return cfg_; };

   private:
    std::unique_ptr<Link> link_;
    HwConfig cfg_;
    std::chrono::milliseconds timeout_;
    std::chrono::milliseconds poll_interval_;
    void send(ByteView frame);
    Bytes recv(std::size_t n);
};
}  // namespace tpu_host