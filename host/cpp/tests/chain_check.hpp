// Shared by the FakeTpu and board tests: two layers chained on the TPU, compared with the reference.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "tpu_host/matrix.hpp"
#include "tpu_host/protocol.hpp"
#include "tpu_host/reference.hpp"
#include "tpu_host/tpu_driver.hpp"

// Y = requant(relu(requant(X . W1)) . W2), the hidden layer H stays on the TPU.
inline void check_two_layer_chain(tpu_host::TpuDriver& tpu, const tpu_host::Matrix<std::int8_t>& x,
                                  const tpu_host::Matrix<std::int8_t>& w1,
                                  const tpu_host::Matrix<std::int8_t>& w2) {
    using namespace tpu_host;
    const int m = static_cast<int>(x.rows()), k = static_cast<int>(x.cols());
    const int h_n = static_cast<int>(w1.cols()), n = static_cast<int>(w2.cols());
    const int x_addr = 0, w1_addr = m * k, w2_addr = w1_addr + k * h_n, h_addr = w2_addr + h_n * n;

    tpu.write(x_addr, x.data());
    tpu.write(w1_addr, w1.data());
    tpu.write(w2_addr, w2.data());
    tpu.run(Matmul{.a_addr = x_addr, .b_addr = w1_addr, .c_addr = h_addr, .m = m, .k = k,
                   .n = h_n, .ld_a = k, .ld_b = h_n, .ld_c = h_n, .requant = true, .relu = true,
                   .to_input = true, .shift = 10});
    tpu.run(Matmul{.a_addr = h_addr, .b_addr = w2_addr, .c_addr = 0, .m = m, .k = h_n, .n = n,
                   .ld_a = h_n, .ld_b = n, .ld_c = n, .requant = true, .shift = 10});
    const std::vector<std::int64_t> y = tpu.read(0, m * n, 1);

    Matrix<std::int8_t> h(x.rows(), w1.cols());
    const Matrix<std::int64_t> h_wide = matmul_ref(x, w1, true, true, 10);
    std::transform(h_wide.begin(), h_wide.end(), h.begin(),
                   [](std::int64_t v) { return static_cast<std::int8_t>(v); });
    EXPECT_EQ(y, matmul_ref(h, w2, true, false, 10).data());
}
