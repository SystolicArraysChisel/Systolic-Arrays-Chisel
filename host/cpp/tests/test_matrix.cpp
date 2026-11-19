#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "tpu_host/matrix.hpp"

using namespace tpu_host;

TEST(Matrix, DefaultIsEmpty) {
    const Matrix<int> m;
    EXPECT_EQ(m.rows(), 0u);
    EXPECT_EQ(m.cols(), 0u);
    EXPECT_TRUE(m.data().empty());
}

TEST(Matrix, NewMatrixIsZero) {
    const Matrix<int> m(2, 3);
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);
    ASSERT_EQ(m.data().size(), 6u);
    for (int v : m.data()) {
        EXPECT_EQ(v, 0);
    }
}

TEST(Matrix, RowMajor) {
    Matrix<int> m(2, 3);
    m(1, 2) = 7;
    EXPECT_EQ(m.data()[5], 7);  // row 1 * 3 columns + column 2
    EXPECT_EQ(m(1, 2), 7);
}

TEST(Matrix, ValuesConstructor) {
    const Matrix<int> m(2, 3, {1, 2, 3, 4, 5, 6});
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 2), 3);
    EXPECT_EQ(m(1, 0), 4);
    EXPECT_EQ(m(1, 2), 6);
}

TEST(Matrix, WrongValueCountThrows) {
    // The parentheses keep the commas away from the macro.
    EXPECT_THROW((Matrix<int>(2, 3, {1, 2})), std::invalid_argument);
}

TEST(Matrix, ConstAccessReads) {
    const Matrix<int> m(1, 2, {8, 9});
    const Matrix<int>& view = m;  // only the const operator() can be called here
    EXPECT_EQ(view(0, 1), 9);
}

TEST(Matrix, Equality) {
    Matrix<int> a(2, 2, {1, 2, 3, 4});
    const Matrix<int> b(2, 2, {1, 2, 3, 4});
    EXPECT_EQ(a, b);
    a(1, 1) = 0;
    EXPECT_NE(a, b);
    // Same elements, different shape: not equal.
    EXPECT_NE((Matrix<int>(1, 4, {1, 2, 3, 4})), (Matrix<int>(2, 2, {1, 2, 3, 4})));
}

TEST(Matrix, WorksForInt8) {
    Matrix<std::int8_t> m(1, 2);
    m(0, 0) = -127;
    m(0, 1) = 127;
    EXPECT_EQ(m(0, 0), -127);
    EXPECT_EQ(m(0, 1), 127);
}
