#include "tpu_host/fake_tpu.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "tpu_host/matrix.hpp"
#include "tpu_host/reference.hpp"

namespace tpu_host {

namespace {

int le16(ByteView f, std::size_t offset) { return f[offset] | f[offset + 1] << 8; }

void append_le(Bytes& out, std::uint32_t value, int size) {
    for (int i = 0; i < size; ++i) {
        out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
    }
}

// Removes and returns the first n bytes of buf.
Bytes take_front(Bytes& buf, std::size_t n) {
    const auto k = static_cast<std::ptrdiff_t>(std::min(n, buf.size()));
    Bytes out(buf.begin(), buf.begin() + k);
    buf.erase(buf.begin(), buf.begin() + k);
    return out;
}

}  // namespace

FakeTpu::FakeTpu(HwConfig cfg, int busy_polls)
    : inp(static_cast<std::size_t>(cfg.in_depth)),
      res(static_cast<std::size_t>(cfg.out_depth)),
      cfg_(std::move(cfg)),
      busy_polls_(busy_polls) {}

void FakeTpu::write(ByteView data) {
    rx_.insert(rx_.end(), data.begin(), data.end());
    process();
}

Bytes FakeTpu::read(std::size_t n) { return take_front(tx_, n); }

void FakeTpu::process() {
    while (!rx_.empty()) {
        const auto op = static_cast<Op>(rx_[0]);
        std::size_t len = 0;
        switch (op) {
            case Op::Write:
                if (rx_.size() < 5) {
                    return;  // count not known yet
                }
                len = 5 + static_cast<std::size_t>(le16(rx_, 3));
                break;
            case Op::Read:
                len = 6;
                break;
            case Op::Matmul:
                len = MATMUL_FRAME_BYTES;
                break;
            case Op::Status:
                len = 1;
                break;
            default:
                error = true;  // unknown opcode: ignored
                rx_.erase(rx_.begin());
                continue;
        }
        if (rx_.size() < len) {
            return;  // wait for the rest of the frame
        }
        const Bytes frame = take_front(rx_, len);
        ops.push_back(op);
        switch (op) {
            case Op::Write:
                do_write(frame);
                break;
            case Op::Read:
                do_read(frame);
                break;
            case Op::Matmul:
                do_matmul(frame);
                break;
            case Op::Status:
                do_status();
                break;
        }
    }
}

void FakeTpu::do_write(ByteView f) {
    const int addr = le16(f, 1);
    for (std::size_t i = 5; i < f.size(); ++i) {
        const auto at = static_cast<std::size_t>(addr + static_cast<int>(i - 5)) % inp.size();
        inp[at] = static_cast<std::int8_t>(f[i]);
    }
}

void FakeTpu::do_read(ByteView f) {
    const int addr = le16(f, 1);
    const int count = le16(f, 3);
    const int size = f[5];
    if (size != 1 && size != 4) {
        error = true;
        return;
    }
    for (int i = 0; i < count; ++i) {
        const auto at = static_cast<std::size_t>(addr + i) % res.size();
        // The low size bytes; a negative value comes out in two's complement.
        append_le(tx_, static_cast<std::uint32_t>(res[at]), size);
    }
}

void FakeTpu::do_matmul(ByteView f) {
    const int a_addr = le16(f, 1), b_addr = le16(f, 3), c_addr = le16(f, 5);
    const int m = le16(f, 7), k = le16(f, 9), n = le16(f, 11);
    const int ld_a = le16(f, 13), ld_b = le16(f, 15), ld_c = le16(f, 17);
    const int flags = f[19];
    const int shift = f[20];
    if (shift >= cfg_.acc_width || (flags & ~0b11) != 0) {
        error = true;
        return;
    }

    // Element (i, j) of a matrix at addr with leading dimension ld.
    auto in_at = [this](int addr, int ld, int i, int j) {
        return inp[static_cast<std::size_t>(addr + i * ld + j) % inp.size()];
    };
    const auto um = static_cast<std::size_t>(m), uk = static_cast<std::size_t>(k),
               un = static_cast<std::size_t>(n);
    Matrix<std::int8_t> a(um, uk), b(uk, un);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            a(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = in_at(a_addr, ld_a, i, j);
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            b(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = in_at(b_addr, ld_b, i, j);
        }
    }

    const Matrix<std::int64_t> c =
        matmul_ref(a, b, (flags >> REQUANT_BIT & 1) != 0, (flags >> RELU_BIT & 1) != 0, shift,
                   cfg_.acc_width);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            res[static_cast<std::size_t>(c_addr + i * ld_c + j) % res.size()] =
                c(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }

    const int rows = cfg_.rows, cols = cfg_.cols;
    const int tiles = (k / rows) * (n / cols);
    cycles = static_cast<std::uint32_t>(tiles * (rows + m + rows + cols - 1));  // load + stream + drain
    busy_pe_cycles = static_cast<std::uint32_t>(m) * static_cast<std::uint32_t>(k) *
                     static_cast<std::uint32_t>(n);
    busy_left_ = busy_polls_;
}

void FakeTpu::do_status() {
    const bool busy = busy_left_ > 0;
    busy_left_ = std::max(0, busy_left_ - 1);
    const auto flags = static_cast<std::uint32_t>(busy << BUSY_BIT | error << ERROR_BIT |
                                                  overrun << OVERRUN_BIT);
    append_le(tx_, flags, 1);
    append_le(tx_, cycles, 4);
    append_le(tx_, busy_pe_cycles, 4);
    error = overrun = false;  // cleared when the reply is captured
}

}  // namespace tpu_host
