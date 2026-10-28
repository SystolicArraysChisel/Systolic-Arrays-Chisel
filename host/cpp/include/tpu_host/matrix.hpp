#pragma once
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tpu_host {

template <typename T>
class Matrix {
   public:
    Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}
    Matrix(std::size_t rows, std::size_t cols, std::vector<T> values)
        : rows_(rows), cols_(cols), data_(std::move(values)) {
        if (data_.size() != rows_ * cols_) {
            throw std::invalid_argument(
                "data size doesn't correspond to provided matrix dimensions");
        }
    }
    Matrix() = default;
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    const std::vector<T>& data() const { return data_; }
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
    T& operator()(std::size_t r, std::size_t c) {
        assert(r < rows_ && c < cols_);
        return data_[r * cols_ + c];
    }
    const T& operator()(std::size_t r, std::size_t c) const {
        assert(r < rows_ && c < cols_);
        return data_[r * cols_ + c];
    }
    bool operator==(const Matrix&) const = default;

   private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T> data_;  // row-major: element (r, c) is at data_[r * cols_ + c]
};

}  // namespace tpu_host
