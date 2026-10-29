#pragma once
#include <cstddef>

#include "tpu_host/protocol.hpp"

namespace tpu_host {
class Link {
   public:
    virtual void write(ByteView data) = 0;
    virtual Bytes read(std::size_t n) = 0;
    virtual ~Link() = default;
};
}  // namespace tpu_host
