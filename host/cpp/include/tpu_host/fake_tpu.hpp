// A software stand-in for the FPGA, for testing the host without hardware.
//
// FakeTpu is a Link that answers the host protocol the way the CommandDecoder
// does: same frames, same error rules, sticky error and overrun flags cleared
// by STATUS. MATMUL is computed with the reference model. Its cycle count is
// only a rough estimate. Port of tpu_host/sim.py.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "tpu_host/config.hpp"
#include "tpu_host/link.hpp"
#include "tpu_host/protocol.hpp"

namespace tpu_host {

class FakeTpu : public Link {
   public:
    // busy_polls: STATUS replies with busy = 1 after each MATMUL.
    explicit FakeTpu(HwConfig cfg = {}, int busy_polls = 1);

    void write(ByteView data) override;
    Bytes read(std::size_t n) override;

    // Public so tests can preload and inspect them.
    std::vector<std::int8_t> inp;   // input memory, cfg.in_depth entries
    std::vector<std::int64_t> res;  // result memory, cfg.out_depth entries
    bool error = false;
    bool overrun = false;
    std::uint32_t cycles = 0;
    std::uint32_t busy_pe_cycles = 0;
    std::vector<Op> ops;  // opcode of every frame received

   private:
    HwConfig cfg_;
    int busy_polls_;
    int busy_left_ = 0;
    Bytes rx_;  // received, not decoded yet
    Bytes tx_;  // replies, not read yet

    void process();
    void do_write(ByteView f);
    void do_read(ByteView f);
    void do_matmul(ByteView f);
    void do_status();
};

}  // namespace tpu_host
