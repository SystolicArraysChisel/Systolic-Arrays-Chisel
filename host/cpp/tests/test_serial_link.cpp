// SerialLink without a board: only the error path can be tested. The real
// check is run_board against the FPGA.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <system_error>

#include "tpu_host/serial_link.hpp"
#include "tpu_host/tpu_driver.hpp"

using namespace tpu_host;
using testing::HasSubstr;
using testing::ThrowsMessage;

TEST(SerialLink, OpeningAMissingPortThrows) {
    EXPECT_THAT([] { SerialLink link("COM99", 115'200); },
                ThrowsMessage<std::system_error>(HasSubstr("cannot open COM99")));
}

TEST(SerialLink, TpuDriverOpenReportsAMissingPort) {
    EXPECT_THROW(TpuDriver::open("COM99"), std::system_error);
}
