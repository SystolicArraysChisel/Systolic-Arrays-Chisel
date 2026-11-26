// Handwritten digits (64 -> 16 -> 10 MLP) on the accelerator, end to end.
//
//   ./build/run_mlp [demo/digits_model.json]      FakeTpu, or the board if TPU_PORT is set
//
// Model and test set come from scripts/train_digits.py. Layer 1 is bias + requant + leaky
// ReLU and stays on the TPU (to_input); layer 2 is bias + raw int32 logits.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "tpu_host/config.hpp"
#include "tpu_host/fake_tpu.hpp"
#include "tpu_host/reference.hpp"
#include "tpu_host/tpu_driver.hpp"

using namespace tpu_host;
using nlohmann::json;

// Memory layout (input memory): X 32x64, W1 64x16, H 32x16, W2 16x12; bias: b1 at 0, b2 at 16.
constexpr int BATCH = 32, IN = 64, HID = 16, OUT = 10, OUT_PAD = 12;
constexpr int X_ADDR = 0, W1_ADDR = BATCH * IN, H_ADDR = W1_ADDR + IN * HID,
              W2_ADDR = H_ADDR + BATCH * HID, B2_ADDR = HID;

// A JSON matrix (array of rows) as a flat vector, rows zero-padded to `cols` entries.
template <typename T>
std::vector<T> flat(const json& rows, std::size_t cols) {
    std::vector<T> v;
    for (const json& row : rows.is_array() && rows[0].is_array() ? rows : json::array({rows})) {
        for (std::size_t j = 0; j < cols; ++j) {
            v.push_back(j < row.size() ? static_cast<T>(row[j].get<int>()) : T{0});
        }
    }
    return v;
}

int main(int argc, char** argv) {
    std::ifstream file(argc > 1 ? argv[1] : "demo/digits_model.json");
    if (!file) {
        std::cerr << "cannot open the model (run from host/cpp or pass its path)\n";
        return 1;
    }
    const json j = json::parse(file);
    const auto w1 = flat<std::int8_t>(j["w1"], HID), w2 = flat<std::int8_t>(j["w2"], OUT_PAD),
               x = flat<std::int8_t>(j["x"], IN);
    const auto b1 = flat<std::int32_t>(j["b1"], HID), b2 = flat<std::int32_t>(j["b2"], OUT_PAD);
    const std::vector<int> labels = j["labels"], expected = j["expected"];
    const int shift = j["shift1"], leak = j["leak_shift"];

    const char* port = std::getenv("TPU_PORT");
    const char* config = std::getenv("TPU_CONFIG");
    const HwConfig cfg = config != nullptr ? HwConfig::load(config) : HwConfig{};
    TpuDriver tpu = port != nullptr ? TpuDriver::open(port, cfg)
                                    : TpuDriver(std::make_unique<FakeTpu>(cfg), cfg);

    tpu.write(W1_ADDR, w1);
    tpu.write(W2_ADDR, w2);
    tpu.write_bias(0, b1);
    tpu.write_bias(B2_ADDR, b2);
    const Matrix<std::int8_t> w1m(IN, HID, w1), w2m(HID, OUT_PAD, w2);

    Perf p1, p2;
    int correct = 0;
    const int n = static_cast<int>(labels.size());
    for (int start = 0; start < n; start += BATCH) {
        const int m = std::min(BATCH, n - start);
        const std::vector<std::int8_t> xb(x.begin() + start * IN, x.begin() + (start + m) * IN);
        tpu.write(X_ADDR, xb);
        p1 += tpu.run(Matmul{.a_addr = X_ADDR, .b_addr = W1_ADDR, .c_addr = H_ADDR, .m = m,
                             .k = IN, .n = HID, .ld_a = IN, .ld_b = HID, .ld_c = HID,
                             .requant = true, .relu = true, .to_input = true, .shift = shift,
                             .bias = true, .bias_addr = 0, .leak_shift = leak});
        p2 += tpu.run(Matmul{.a_addr = H_ADDR, .b_addr = W2_ADDR, .c_addr = 0, .m = m, .k = HID,
                             .n = OUT_PAD, .ld_a = HID, .ld_b = OUT_PAD, .ld_c = OUT_PAD,
                             .bias = true, .bias_addr = B2_ADDR});
        const std::vector<std::int64_t> logits = tpu.read(0, m * OUT_PAD, 4);

        // Reference: layer 1 output is int8 (to_input), as on the TPU.
        const Matrix<std::int64_t> h_wide =
            matmul_ref(Matrix<std::int8_t>(m, IN, xb), w1m, true, true, shift, cfg.acc_width, b1, leak);
        Matrix<std::int8_t> h(h_wide.rows(), h_wide.cols());
        std::ranges::transform(h_wide, h.begin(),
                               [](std::int64_t v) { return static_cast<std::int8_t>(v); });
        const auto ref = matmul_ref(h, w2m, false, false, 0, cfg.acc_width, b2).data();
        if (logits != ref) {
            std::cerr << "logits differ from the reference in the batch at image " << start << "\n";
            return 1;
        }
        for (int i = 0; i < m; ++i) {
            const auto row = logits.begin() + i * OUT_PAD;
            const int pred = static_cast<int>(std::max_element(row, row + OUT) - row);
            if (pred != expected[static_cast<std::size_t>(start + i)]) {
                std::cerr << "image " << start + i << ": predicted " << pred
                          << ", the Python simulation says " << expected[static_cast<std::size_t>(start + i)] << "\n";
                return 1;
            }
            correct += pred == labels[static_cast<std::size_t>(start + i)];
        }
    }

    std::cout << (port != nullptr ? "board" : "FakeTpu (cycle model, not hardware)") << "\n"
              << "accuracy " << correct << "/" << n << " = " << 100.0 * correct / n << " %\n";
    for (const auto& [name, p] : {std::pair{"layer 1", p1}, std::pair{"layer 2", p2}}) {
        std::cout << name << ": " << p.cycles << " cycles, utilization "
                  << 100.0 * p.utilization(cfg) << " %\n";
    }
    return 0;
}
