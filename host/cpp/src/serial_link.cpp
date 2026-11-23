#include "tpu_host/serial_link.hpp"

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN  // skip the rarely used parts of the API
#endif
#ifndef NOMINMAX  // MinGW's C++ headers may already define it
#define NOMINMAX  // no min/max macros clashing with std::min
#endif
#include <windows.h>

#include <string>
#include <system_error>

namespace tpu_host {

namespace {

// Throws the last Win32 error with a readable message, e.g.
// "cannot open COM5: The system cannot find the file specified."
[[noreturn]] void fail(const std::string& what) {
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), what);
}

}  // namespace

SerialLink::SerialLink(const std::string& port, int baud) {
    // \\.\COMn is the device path; plain "COMn" only works for COM1 to COM9.
    handle_ = CreateFileA(("\\\\.\\" + port).c_str(), GENERIC_READ | GENERIC_WRITE,
                          0,  // exclusive: fails if a terminal has the port open
                          nullptr, OPEN_EXISTING, 0, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
        fail("cannot open " + port);
    }

    // A throwing constructor never runs the destructor: close the port ourselves.
    auto close_and_fail = [&](const std::string& what) {
        const DWORD err = GetLastError();  // CloseHandle may overwrite it
        CloseHandle(handle_);
        SetLastError(err);
        fail(what + " on " + port);
    };

    // Line format: 8N1, no flow control. XON/XOFF especially must stay off:
    // 0x11 and 0x13 are ordinary data bytes in our frames.
    DCB dcb{};
    dcb.DCBlength = sizeof(dcb);
    if (!GetCommState(handle_, &dcb)) {
        close_and_fail("GetCommState");
    }
    dcb.BaudRate = static_cast<DWORD>(baud);
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = FALSE;
    dcb.fOutxCtsFlow = FALSE;
    dcb.fOutxDsrFlow = FALSE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    dcb.fOutX = FALSE;
    dcb.fInX = FALSE;
    if (!SetCommState(handle_, &dcb)) {
        close_and_fail("SetCommState");
    }

    // MAXDWORD / MAXDWORD / constant: ReadFile returns at once with the bytes
    // waiting, or waits up to the constant for a first byte. TpuDriver::recv
    // loops until its own deadline.
    COMMTIMEOUTS timeouts{};
    timeouts.ReadIntervalTimeout = MAXDWORD;
    timeouts.ReadTotalTimeoutMultiplier = MAXDWORD;
    timeouts.ReadTotalTimeoutConstant = 100;   // ms
    timeouts.WriteTotalTimeoutConstant = 1000;  // ms
    if (!SetCommTimeouts(handle_, &timeouts)) {
        close_and_fail("SetCommTimeouts");
    }

    // Drop bytes left over from an earlier run (a reply nobody read).
    PurgeComm(handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);
}

SerialLink::~SerialLink() { CloseHandle(handle_); }  // a destructor must not throw

void SerialLink::write(ByteView data) {
    DWORD written = 0;
    if (!WriteFile(handle_, data.data(), static_cast<DWORD>(data.size()), &written, nullptr)) {
        fail("WriteFile");
    }
    if (written < data.size()) {
        SetLastError(ERROR_TIMEOUT);
        fail("WriteFile: sent " + std::to_string(written) + " of " +
             std::to_string(data.size()) + " bytes");
    }
}

Bytes SerialLink::read(std::size_t n) {
    Bytes buf(n);
    DWORD got = 0;
    if (!ReadFile(handle_, buf.data(), static_cast<DWORD>(n), &got, nullptr)) {
        fail("ReadFile");
    }
    buf.resize(got);  // 0 bytes is a timeout, not an error: recv decides
    return buf;
}

}  // namespace tpu_host

#else  // POSIX (macOS, Linux)

#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <string>
#include <system_error>
#ifdef __APPLE__
#include <IOKit/serial/ioss.h>  // IOSSIOSPEED: arbitrary baud rates
#endif

namespace tpu_host {

namespace {

// Throws errno with a readable message, e.g.
// "cannot open /dev/ttyUSB0: No such file or directory"
[[noreturn]] void fail(const std::string& what) {
    throw std::system_error(errno, std::system_category(), what);
}

}  // namespace

SerialLink::SerialLink(const std::string& port, int baud) {
    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0) {
        fail("cannot open " + port);
    }

    // A throwing constructor never runs the destructor: close the port ourselves.
    auto close_and_fail = [&](const std::string& what) {
        const int err = errno;  // close may overwrite it
        ::close(fd_);
        errno = err;
        fail(what + " on " + port);
    };

    // Raw 8N1, no flow control (0x11 / 0x13 are ordinary data bytes in our frames).
    termios tio{};
    if (tcgetattr(fd_, &tio) != 0) {
        close_and_fail("tcgetattr");
    }
    cfmakeraw(&tio);
    tio.c_cflag |= CLOCAL | CREAD;
    tio.c_cflag &= ~static_cast<tcflag_t>(CSTOPB | CRTSCTS);
    // VMIN 0 / VTIME 1: read returns at once with the bytes waiting, or waits
    // up to 100 ms for a first byte. TpuDriver::recv loops until its own deadline.
    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 1;
#ifdef __APPLE__
    cfsetspeed(&tio, B9600);  // placeholder: the real rate is set by IOSSIOSPEED below
#else
    // shortcut: only the two rates the project uses, add a Bxxx case for others
    if (baud != 115'200 && baud != 2'000'000) {
        errno = EINVAL;
        close_and_fail("unsupported baud rate");
    }
    cfsetspeed(&tio, baud == 115'200 ? B115200 : B2000000);
#endif
    if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
        close_and_fail("tcsetattr");
    }
#ifdef __APPLE__
    speed_t speed = static_cast<speed_t>(baud);
    if (ioctl(fd_, IOSSIOSPEED, &speed) != 0) {
        close_and_fail("IOSSIOSPEED");
    }
#endif

    // Drop bytes left over from an earlier run (a reply nobody read).
    tcflush(fd_, TCIOFLUSH);
}

SerialLink::~SerialLink() { ::close(fd_); }  // a destructor must not throw

void SerialLink::write(ByteView data) {
    std::size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t n = ::write(fd_, data.data() + sent, data.size() - sent);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("write");
        }
        sent += static_cast<std::size_t>(n);
    }
}

Bytes SerialLink::read(std::size_t n) {
    Bytes buf(n);
    ssize_t got;
    while ((got = ::read(fd_, buf.data(), n)) < 0 && errno == EINTR) {
    }
    if (got < 0) {
        fail("read");
    }
    buf.resize(static_cast<std::size_t>(got));  // 0 bytes is a timeout, not an error: recv decides
    return buf;
}

}  // namespace tpu_host

#endif  // _WIN32
