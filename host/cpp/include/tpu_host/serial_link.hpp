// The real link: a Windows COM port, 8N1, no flow control. The Win32
// equivalent of serial.Serial(port, baud, timeout=...).
#pragma once
#include <cstddef>
#include <string>

#include "tpu_host/link.hpp"
#include "tpu_host/protocol.hpp"

namespace tpu_host {

class SerialLink : public Link {
   public:
    // port: "COM5". Throws std::system_error if it cannot be opened or set up.
    SerialLink(const std::string& port, int baud);
    ~SerialLink() override;  // closes the port
    SerialLink(const SerialLink&) = delete;  // one owner of the port
    SerialLink& operator=(const SerialLink&) = delete;

    void write(ByteView data) override;
    // Returns at once with what is waiting; if nothing is, waits up to 100 ms
    // for a first byte and may return none.
    Bytes read(std::size_t n) override;

   private:
    void* handle_;  // a Win32 HANDLE; <windows.h> stays in the .cpp
};

}  // namespace tpu_host
