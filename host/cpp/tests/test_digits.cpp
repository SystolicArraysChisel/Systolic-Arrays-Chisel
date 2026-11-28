// The drawing MLPs (digits.hpp) on FakeTpu, and the drawing -> input conversion.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "tpu_host/config.hpp"
#include "tpu_host/digits.hpp"
#include "tpu_host/fake_tpu.hpp"
#include "tpu_host/tpu_driver.hpp"

using namespace tpu_host;

namespace {

const digits::Model& model(const std::string& path = DIGITS_MODEL) {
    static std::map<std::string, digits::Model> cache;
    auto it = cache.find(path);
    if (it == cache.end()) {
        it = cache.emplace(path, digits::Model::load(path)).first;
    }
    return it->second;
}

HwConfig square(int n) {
    HwConfig cfg;
    cfg.rows = cfg.cols = n;
    return cfg;
}

}  // namespace

TEST(Digits, LoadsAndPadsTheModel) {
    const digits::Model& m = model();
    EXPECT_EQ(m.name, "Handwritten digits");
    EXPECT_EQ(m.classes,
              (std::vector<std::string>{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}));
    EXPECT_EQ(m.hidden_units, 16);
    EXPECT_EQ(m.hidden, 16);
    EXPECT_EQ(m.out, 16);
    EXPECT_EQ(m.w1.rows(), digits::PIXELS);
    EXPECT_EQ(m.w2.cols(), 16u);
    EXPECT_EQ(m.b2[m.classes.size()], 0);  // padding
    EXPECT_EQ(m.x.rows(), m.labels.size());
}

TEST(Digits, LoadsTheShapesModel) {
    const digits::Model& m = model(SHAPES_MODEL);
    EXPECT_EQ(m.name, "Shapes and symbols");
    ASSERT_EQ(m.classes.size(), 10u);
    EXPECT_EQ(m.classes[0], "circle");
    EXPECT_EQ(m.out, 16);
    EXPECT_GT(m.x.rows(), 0u);
}

// (model file, array size)
class ModelOnArray : public testing::TestWithParam<std::tuple<std::string, int>> {};

TEST_P(ModelOnArray, MatchesReferenceAndPythonSimulation) {
    const auto& [path, n] = GetParam();
    const HwConfig cfg = square(n);
    TpuDriver tpu(std::make_unique<FakeTpu>(cfg), cfg);
    const digits::Model& m = model(path);
    digits::Mlp mlp(tpu, m, 32);

    const auto& x = m.x.data();
    const Matrix<std::int8_t> xb(32, digits::PIXELS, {x.begin(), x.begin() + 32 * digits::PIXELS});
    const digits::Mlp::Result r = mlp.run(xb);
    EXPECT_EQ(r.logits, digits::reference(m, xb));
    for (std::size_t i = 0; i < 32; ++i) {
        EXPECT_EQ(digits::predict(r.logits, i, m.classes.size()), m.expected[i]) << "image " << i;
    }
    EXPECT_GT(r.layer1.cycles, 0u);
    EXPECT_GT(r.layer2.cycles, 0u);
}

TEST_P(ModelOnArray, OneImageAtATime) {
    const auto& [path, n] = GetParam();
    const HwConfig cfg = square(n);
    TpuDriver tpu(std::make_unique<FakeTpu>(cfg), cfg);
    const digits::Model& m = model(path);
    digits::Mlp mlp(tpu, m, 1);
    for (std::size_t i = 0; i < 5; ++i) {
        const auto first = m.x.data().begin() + static_cast<std::ptrdiff_t>(i) * digits::PIXELS;
        const Matrix<std::int8_t> img(1, digits::PIXELS, {first, first + digits::PIXELS});
        EXPECT_EQ(digits::predict(mlp.run(img).logits, 0, m.classes.size()), m.expected[i]);
    }
}

INSTANTIATE_TEST_SUITE_P(Models, ModelOnArray,
                         testing::Combine(testing::Values(std::string(DIGITS_MODEL),
                                                          std::string(SHAPES_MODEL)),
                                          testing::Values(2, 4, 8)));

TEST(Digits, RejectsAModelThatDoesNotFit) {
    HwConfig cfg;
    cfg.in_depth = 1024;
    TpuDriver tpu(std::make_unique<FakeTpu>(cfg), cfg);
    EXPECT_THROW(digits::Mlp(tpu, model(), 32), std::invalid_argument);
}

TEST(Digits, FeaturesCountInkPerBlock) {
    std::vector<std::uint8_t> bitmap(digits::BITMAP * digits::BITMAP, 0);
    EXPECT_EQ(digits::features(bitmap), std::vector<std::int8_t>(digits::PIXELS, 0));

    // Top-left 4x4 block: 8 of 16 pixels set (two rows); faint ink does not count.
    for (int i = 0; i < 2 * digits::BITMAP; ++i) {
        if (i % digits::BITMAP < 4) {
            bitmap[static_cast<std::size_t>(i)] = 255;
        }
    }
    bitmap[2 * digits::BITMAP] = 100;
    // Bottom-right block full.
    for (int y = 28; y < 32; ++y) {
        for (int x = 28; x < 32; ++x) {
            bitmap[static_cast<std::size_t>(y * digits::BITMAP + x)] = 200;
        }
    }
    const std::vector<std::int8_t> f = digits::features(bitmap);
    EXPECT_EQ(f[0], 64);  // 8 / 16 * 127 = 63.5, rounded half up
    EXPECT_EQ(f[1], 0);
    EXPECT_EQ(f[digits::PIXELS - 1], 127);
}

TEST(Digits, FeaturesRejectAWrongSize) {
    EXPECT_THROW(digits::features(std::vector<std::uint8_t>(64)), std::invalid_argument);
}
