#include "tpu_host/digits.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "tpu_host/quant.hpp"
#include "tpu_host/reference.hpp"

namespace tpu_host::digits {

using nlohmann::json;

namespace {

int round_up(int v, int to) { return (v + to - 1) / to * to; }

// A JSON matrix (array of rows) zero-padded to rows x cols.
Matrix<std::int8_t> matrix(const json& rows, std::size_t n_rows, std::size_t n_cols) {
    Matrix<std::int8_t> m(n_rows, n_cols);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (std::size_t j = 0; j < rows[i].size(); ++j) {
            m(i, j) = static_cast<std::int8_t>(rows[i][j].get<int>());
        }
    }
    return m;
}

std::vector<std::int32_t> vec(const json& v, std::size_t n) {
    std::vector<std::int32_t> out(n, 0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i] = v[i].get<std::int32_t>();
    }
    return out;
}

}  // namespace

Model Model::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const json j = json::parse(file);
    const std::size_t hidden_raw = j.at("b1").size();
    Model m;
    m.name = j.value("name", std::filesystem::path(path).stem().string());
    if (j.contains("classes")) {
        m.classes = j.at("classes").get<std::vector<std::string>>();
    } else {
        for (int c = 0; c < 10; ++c) {
            m.classes.push_back(std::to_string(c));
        }
    }
    const std::size_t n_classes = m.classes.size();
    if (j.at("w1").size() != PIXELS || j.at("b2").size() != n_classes || n_classes < 2 ||
        n_classes > MAX_CLASSES) {
        throw std::runtime_error(path + ": not a 64 -> hidden -> classes network with 2.." +
                                 std::to_string(MAX_CLASSES) + " named classes");
    }
    m.hidden_units = static_cast<int>(hidden_raw);
    m.hidden = round_up(m.hidden_units, PAD);
    m.out = round_up(static_cast<int>(n_classes), PAD);
    m.shift1 = j.at("shift1");
    m.leak_shift = j.at("leak_shift");
    const auto hid = static_cast<std::size_t>(m.hidden), out = static_cast<std::size_t>(m.out);
    m.w1 = matrix(j.at("w1"), PIXELS, hid);
    m.w2 = matrix(j.at("w2"), hid, out);
    m.b1 = vec(j.at("b1"), hid);
    m.b2 = vec(j.at("b2"), out);
    m.x = matrix(j.at("x"), j.at("x").size(), PIXELS);
    m.labels = j.at("labels").get<std::vector<int>>();
    m.expected = j.at("expected").get<std::vector<int>>();
    return m;
}

Matrix<std::int64_t> reference(const Model& model, const Matrix<std::int8_t>& x, int acc_width) {
    // Layer 1's output is int8, as when it stays on the TPU.
    const Matrix<std::int64_t> h_wide =
        matmul_ref(x, model.w1, true, true, model.shift1, acc_width, model.b1, model.leak_shift);
    Matrix<std::int8_t> h(h_wide.rows(), h_wide.cols());
    std::ranges::transform(h_wide, h.begin(),
                           [](std::int64_t v) { return static_cast<std::int8_t>(v); });
    return matmul_ref(h, model.w2, false, false, 0, acc_width, model.b2);
}

// Input memory: X (batch x 64), W1 (64 x hidden), H (batch x hidden), W2 (hidden x out).
// Bias memory: b1 at 0, b2 after it. Every address is a multiple of PAD.
Mlp::Mlp(TpuDriver& tpu, const Model& model, int batch) : tpu_(tpu), model_(model), batch_(batch) {
    const HwConfig& cfg = tpu.cfg();
    x_addr_ = 0;
    w1_addr_ = x_addr_ + batch * PIXELS;
    h_addr_ = w1_addr_ + PIXELS * model.hidden;
    w2_addr_ = h_addr_ + batch * model.hidden;
    b2_addr_ = model.hidden;
    if (batch < 1 || batch > cfg.max_m) {
        throw std::invalid_argument("batch must be in 1.." + std::to_string(cfg.max_m));
    }
    if (w2_addr_ + model.hidden * model.out > cfg.in_depth ||
        b2_addr_ + model.out > cfg.bias_depth) {
        throw std::invalid_argument(model.name + " does not fit in the TPU's memory");
    }
    tpu.write(w1_addr_, model.w1.data());
    tpu.write(w2_addr_, model.w2.data());
    tpu.write_bias(0, model.b1);
    tpu.write_bias(b2_addr_, model.b2);
}

Mlp::Result Mlp::run(const Matrix<std::int8_t>& x) {
    const int m = static_cast<int>(x.rows());
    if (m < 1 || m > batch_ || x.cols() != PIXELS) {
        throw std::invalid_argument("run takes 1.." + std::to_string(batch_) + " images of " +
                                    std::to_string(PIXELS) + " pixels");
    }
    const int hid = model_.hidden, out = model_.out;
    Result r;
    tpu_.write(x_addr_, x.data());
    r.layer1 = tpu_.run(Matmul{.a_addr = x_addr_,
                               .b_addr = w1_addr_,
                               .c_addr = h_addr_,
                               .m = m,
                               .k = PIXELS,
                               .n = hid,
                               .ld_a = PIXELS,
                               .ld_b = hid,
                               .ld_c = hid,
                               .requant = true,
                               .relu = true,
                               .to_input = true,
                               .shift = model_.shift1,
                               .bias = true,
                               .bias_addr = 0,
                               .leak_shift = model_.leak_shift});
    r.layer2 = tpu_.run(Matmul{.a_addr = h_addr_,
                               .b_addr = w2_addr_,
                               .c_addr = 0,
                               .m = m,
                               .k = hid,
                               .n = out,
                               .ld_a = hid,
                               .ld_b = out,
                               .ld_c = out,
                               .bias = true,
                               .bias_addr = b2_addr_});
    r.logits = Matrix<std::int64_t>(static_cast<std::size_t>(m), static_cast<std::size_t>(out),
                                    tpu_.read(0, m * out, 4));
    return r;
}

int predict(const Matrix<std::int64_t>& logits, std::size_t row, std::size_t classes) {
    std::size_t best = 0;
    for (std::size_t c = 1; c < classes; ++c) {
        if (logits(row, c) > logits(row, best)) {
            best = c;
        }
    }
    return static_cast<int>(best);
}

std::vector<std::int8_t> features(std::span<const std::uint8_t> bitmap) {
    if (bitmap.size() != BITMAP * BITMAP) {
        throw std::invalid_argument("features takes a 32x32 bitmap");
    }
    constexpr int BLOCK = BITMAP / SIDE;
    std::vector<std::int8_t> x(PIXELS);
    for (int by = 0; by < SIDE; ++by) {
        for (int bx = 0; bx < SIDE; ++bx) {
            int count = 0;
            for (int y = by * BLOCK; y < (by + 1) * BLOCK; ++y) {
                for (int xx = bx * BLOCK; xx < (bx + 1) * BLOCK; ++xx) {
                    count += bitmap[static_cast<std::size_t>(y * BITMAP + xx)] >= 128;
                }
            }
            x[static_cast<std::size_t>(by * SIDE + bx)] = quantize(count / 16.0, 1.0 / QMAX);
        }
    }
    return x;
}

}  // namespace tpu_host::digits
