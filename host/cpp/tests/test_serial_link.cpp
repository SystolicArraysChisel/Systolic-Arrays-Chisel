// SerialLink without a board: only the error path can be tested. The real
// check is run_board against the FPGA.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>
#include <system_error>

#include "tpu_host/serial_link.hpp"
#include "tpu_host/tpu_driver.hpp"

using namespace tpu_host;
using testing::HasSubstr;
using testing::ThrowsMessage;

#ifdef _WIN32
constexpr const char* kMissingPort = "COM99";
#else
constexpr const char* kMissingPort = "/dev/does-not-exist";
#endif

TEST(SerialLink, OpeningAMissingPortThrows) {
    EXPECT_THAT([] { SerialLink link(kMissingPort, 115'200); },
                ThrowsMessage<std::system_error>(
                    HasSubstr(std::string("cannot open ") + kMissingPort)));
}

TEST(SerialLink, TpuDriverOpenReportsAMissingPort) {
    EXPECT_THROW(TpuDriver::open(kMissingPort), std::system_error);
}
