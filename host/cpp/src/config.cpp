#include "tpu_host/config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace tpu_host {

HwConfig HwConfig::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const nlohmann::json j = nlohmann::json::parse(file);
    HwConfig cfg;
    cfg.name = j.value("name", cfg.name);
    cfg.rows = j.value("rows", cfg.rows);
    cfg.cols = j.value("cols", cfg.cols);
    cfg.in_depth = j.value("in_depth", cfg.in_depth);
    cfg.out_depth = j.value("out_depth", cfg.out_depth);
    cfg.max_m = j.value("max_m", cfg.max_m);
    cfg.acc_width = j.value("acc_width", cfg.acc_width);
    cfg.baud = j.value("baud", cfg.baud);
    cfg.rx_fifo_depth = j.value("rx_fifo_depth", cfg.rx_fifo_depth);
    cfg.clock_hz = j.value("clock_hz", cfg.clock_hz);
    return cfg;
}
}  // namespace tpu_host