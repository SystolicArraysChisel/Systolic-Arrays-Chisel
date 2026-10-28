#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "tpu_host/config.hpp"

using namespace tpu_host;

namespace {

// TEST_DATA_DIR is set by CMake to the absolute path of tests/data.
std::string data(const std::string& file) { return std::string(TEST_DATA_DIR) + "/" + file; }

}  // namespace

TEST(Config, LoadsEveryField) {
    const HwConfig cfg = HwConfig::load(data("config_8x8.json"));
    EXPECT_EQ(cfg.name, "8x8");
    EXPECT_EQ(cfg.rows, 8);
    EXPECT_EQ(cfg.cols, 8);
    EXPECT_EQ(cfg.banks(), 8);
    EXPECT_EQ(cfg.in_depth, 4096);
    EXPECT_EQ(cfg.out_depth, 4096);
    EXPECT_EQ(cfg.max_m, 64);
    EXPECT_EQ(cfg.acc_width, 32);
    EXPECT_EQ(cfg.baud, 115'200);
    EXPECT_EQ(cfg.rx_fifo_depth, 16);
    EXPECT_EQ(cfg.clock_hz, 50'000'000);
}

TEST(Config, MissingKeysKeepTheirDefault) {
    const HwConfig cfg = HwConfig::load(data("config_partial.json"));
    const HwConfig defaults;
    EXPECT_EQ(cfg.rows, 8);
    EXPECT_EQ(cfg.cols, defaults.cols);
    EXPECT_EQ(cfg.max_m, defaults.max_m);
    EXPECT_EQ(cfg.name, defaults.name);
}

TEST(Config, UnknownKeysAreIgnored) {
    EXPECT_NO_THROW(HwConfig::load(data("config_unknown_key.json")));
}

TEST(Config, MissingFileThrows) {
    EXPECT_THROW(HwConfig::load(data("does_not_exist.json")), std::runtime_error);
}

TEST(Config, WrongTypeThrows) {
    EXPECT_THROW(HwConfig::load(data("config_wrong_type.json")), std::exception);
}

TEST(Config, InvalidJsonThrows) {
    EXPECT_THROW(HwConfig::load(data("config_invalid.json")), std::exception);
}
