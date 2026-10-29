// The Link interface, through a minimal implementation: everything written
// can be read back, in order.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "tpu_host/link.hpp"

using namespace tpu_host;

namespace {

class LoopbackLink : public Link {
   public:
    void write(ByteView data) override { buffer_.insert(buffer_.end(), data.begin(), data.end()); }

    Bytes read(std::size_t n) override {
        const auto k = static_cast<std::ptrdiff_t>(std::min(n, buffer_.size()));
        Bytes out(buffer_.begin(), buffer_.begin() + k);
        buffer_.erase(buffer_.begin(), buffer_.begin() + k);
        return out;
    }

   private:
    Bytes buffer_;
};

}  // namespace

TEST(Link, IsAbstract) { EXPECT_TRUE(std::is_abstract_v<Link>); }

TEST(Link, ReadsBackWhatWasWritten) {
    LoopbackLink loop;
    Link& link = loop;  // Tpu only ever sees a Link
    link.write(Bytes{1, 2, 3});
    EXPECT_EQ(link.read(2), (Bytes{1, 2}));
    EXPECT_EQ(link.read(5), (Bytes{3}));  // fewer than asked
    EXPECT_TRUE(link.read(1).empty());    // nothing left: what a timeout looks like
}

TEST(Link, KeepsTheOrderAcrossWrites) {
    LoopbackLink loop;
    Link& link = loop;
    link.write(Bytes{1});
    link.write(Bytes{2, 3});
    EXPECT_EQ(link.read(3), (Bytes{1, 2, 3}));
}

TEST(Link, WorksThroughAUniquePtr) {
    // How Tpu will own its link; the virtual destructor makes the delete safe.
    std::unique_ptr<Link> link = std::make_unique<LoopbackLink>();
    link->write(Bytes{0xFF});
    EXPECT_EQ(link->read(1), (Bytes{0xFF}));
}
