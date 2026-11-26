// Performance report: random int8 square products, checked against the reference model.
//
//   TPU_PORT=COM3 TPU_CONFIG=../../build/4x4/config.json ./build/run_board   # the board
//   ./build/run_board                                                        # FakeTpu
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include "tpu_host/fake_tpu.hpp"
#include "tpu_host/reference.hpp"
#include "tpu_host/tpu_driver.hpp"

using namespace tpu_host;

int main() {
    const char* port = std::getenv("TPU_PORT");
    const char* config = std::getenv("TPU_CONFIG");
    const HwConfig cfg = config != nullptr ? HwConfig::load(config) : HwConfig{};
    TpuDriver tpu = port != nullptr ? TpuDriver::open(port, cfg)
                                    : TpuDriver(std::make_unique<FakeTpu>(cfg), cfg);
    std::cout << (port != nullptr ? "board" : "FakeTpu (cycle model, not hardware)") << ", " << cfg.name
              << " at " << cfg.clock_hz / 1'000'000 << " MHz\n"
              << "size  cycles  busy PE-cycles  util %  MAC/s\n";

    std::mt19937 rng(0);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (int n : {4, 8, 16, 32}) {
        if (n > cfg.max_m || 2 * n * n > cfg.in_depth || n * n > cfg.out_depth) continue;
        const std::size_t s = static_cast<std::size_t>(n);
        Matrix<std::int8_t> a(s, s), b(s, s);
        for (std::int8_t& v : a) v = static_cast<std::int8_t>(dist(rng));
        for (std::int8_t& v : b) v = static_cast<std::int8_t>(dist(rng));
        const auto [c, perf] = tpu.matmul(a, b);
        if (!(c == matmul_ref(a, b))) {
            std::cerr << "mismatch at size " << n << "\n";
            return 1;
        }
        const double macs = static_cast<double>(n) * n * n * cfg.clock_hz / perf.cycles;
        std::cout << std::setw(4) << n << std::setw(8) << perf.cycles << std::setw(16) << perf.busy_pe_cycles
                  << std::fixed << std::setprecision(1) << std::setw(8) << 100 * perf.utilization(cfg)
                  << std::scientific << std::setprecision(2) << std::setw(10) << macs << "\n";
    }
}
