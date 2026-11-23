// The real link: a serial port (Windows COM port, or termios device on
// macOS/Linux), 8N1, no flow control.
#pragma once
#include <cstddef>
#include <string>

#include "tpu_host/link.hpp"
#include "tpu_host/protocol.hpp"

namespace tpu_host {

class SerialLink : public Link {
   public:
    // port: "COM5", "/dev/cu.usbserial-X" or "/dev/ttyUSB0". Throws std::system_error if it cannot be opened or set up.
    SerialLink(const std::string& port, int baud);
    ~SerialLink() override;  // closes the port
    SerialLink(const SerialLink&) = delete;  // one owner of the port
    SerialLink& operator=(const SerialLink&) = delete;

    void write(ByteView data) override;
    // Returns at once with what is waiting; if nothing is, waits up to 100 ms
    // for a first byte and may return none.
    Bytes read(std::size_t n) override;

   private:
#ifdef _WIN32
    void* handle_;  // a Win32 HANDLE; <windows.h> stays in the .cpp
#else
    int fd_;
#endif
};

}  // namespace tpu_host
