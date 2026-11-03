#pragma once
#include <algorithm>
#include <string>

namespace tpu_host {
struct HwConfig {
    int rows = 4;               // PE rows (= banks = cols in the MVP)
    int cols = 4;               // PE columns
    int in_depth = 4096;        // int8 elements in the input memory (A and B)
    int out_depth = 4096;       // int32 words in the result memory (C)
    int max_m = 64;             // largest m one MATMUL accepts
    int acc_width = 32;         // bits of an accumulator
    int baud = 2'000'000;       // UART speed
    int clock_hz = 50'000'000;  // core clock (BoardConfig.nexysTpu), to turn cycles into time
    int rx_fifo_depth = 16;     // bytes RxFifo holds while the decoder is busy

    std::string name = "4x4";  // configuration name (EmitBoard.configs)

    int banks() const { return std::max(rows, cols); }
    static HwConfig load(const std::string& path);
};
}  // namespace tpu_host