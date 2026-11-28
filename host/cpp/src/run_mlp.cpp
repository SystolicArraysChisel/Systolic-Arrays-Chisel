// A drawing classifier (64 -> 16 -> classes MLP) on the accelerator, end to end.
//
//   ./build/run_mlp [demo/digits_model.json]      FakeTpu, or the board if TPU_PORT is set
//   ./build/run_mlp demo/shapes_model.json
//
// Model and test set come from scripts/train_digits.py or train_shapes.py; the network is in
// digits.hpp.
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "tpu_host/config.hpp"
#include "tpu_host/digits.hpp"
#include "tpu_host/fake_tpu.hpp"
#include "tpu_host/tpu_driver.hpp"

using namespace tpu_host;

constexpr int BATCH = 32;

int main(int argc, char** argv) try {
    const digits::Model model = digits::Model::load(argc > 1 ? argv[1] : "demo/digits_model.json");

    const char* port = std::getenv("TPU_PORT");
    const char* config = std::getenv("TPU_CONFIG");
    const HwConfig cfg = config != nullptr ? HwConfig::load(config) : HwConfig{};
    TpuDriver tpu = port != nullptr ? TpuDriver::open(port, cfg)
                                    : TpuDriver(std::make_unique<FakeTpu>(cfg), cfg);
    digits::Mlp mlp(tpu, model, BATCH);

    Perf p1, p2;
    int correct = 0;
    const int n = static_cast<int>(model.labels.size());
    for (int start = 0; start < n; start += BATCH) {
        const int m = std::min(BATCH, n - start);
        const auto first = model.x.data().begin() + start * digits::PIXELS;
        const Matrix<std::int8_t> xb(static_cast<std::size_t>(m), digits::PIXELS,
                                     {first, first + m * digits::PIXELS});
        const digits::Mlp::Result r = mlp.run(xb);
        p1 += r.layer1;
        p2 += r.layer2;

        if (r.logits != digits::reference(model, xb, cfg.acc_width)) {
            std::cerr << "logits differ from the reference in the batch at image " << start << "\n";
            return 1;
        }
        for (int i = 0; i < m; ++i) {
            const auto img = static_cast<std::size_t>(start + i);
            const int pred =
                digits::predict(r.logits, static_cast<std::size_t>(i), model.classes.size());
            if (pred != model.expected[img]) {
                std::cerr << "image " << img << ": predicted " << pred
                          << ", the Python simulation says " << model.expected[img] << "\n";
                return 1;
            }
            correct += pred == model.labels[img];
        }
    }

    std::cout << model.name << " on "
              << (port != nullptr ? "the board" : "FakeTpu (cycle model, not hardware)") << "\n"
              << "accuracy " << correct << "/" << n << " = " << 100.0 * correct / n << " %\n";
    for (const auto& [name, p] : {std::pair{"layer 1", p1}, std::pair{"layer 2", p2}}) {
        std::cout << name << ": " << p.cycles << " cycles, utilization "
                  << 100.0 * p.utilization(cfg) << " %\n";
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
}
