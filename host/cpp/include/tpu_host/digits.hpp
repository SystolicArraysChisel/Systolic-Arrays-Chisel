// Drawings on the accelerator: a 64 -> hidden -> classes MLP on 8x8 images, trained by
// scripts/train_digits.py (sklearn's digits) or scripts/train_shapes.py (shapes, symbols).
//
// Layer 1 is bias + requant + leaky ReLU and stays on the TPU (to_input);
// layer 2 is bias + raw int32 logits. Used by run_mlp and the Qt demo.
#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "tpu_host/matrix.hpp"
#include "tpu_host/tpu_driver.hpp"

namespace tpu_host::digits {

inline constexpr int SIDE = 8;              // the network sees 8x8 images
inline constexpr int PIXELS = SIDE * SIDE;  // 64 inputs
inline constexpr int MAX_CLASSES = 16;      // the most classes a model may have
inline constexpr int BITMAP = 32;  // features() takes a 32x32 bitmap, like sklearn's source
inline constexpr int PAD = 16;     // hidden and output widths are padded to a multiple of
                                   // this, a multiple of banks for 2x2, 4x4 and 8x8

struct Model {
    std::string name;                  // shown in the demo; the file name if the JSON has none
    std::vector<std::string> classes;  // class names; "0".."9" if the JSON has none
    int hidden_units = 0;              // hidden neurons of the trained network
    int hidden = 0;                    // padded hidden width (multiple of PAD)
    int out = 0;                       // padded output width (multiple of PAD)
    int shift1 = 0;                    // requant shift of layer 1
    int leak_shift = 0;
    Matrix<std::int8_t> w1;  // PIXELS x hidden
    Matrix<std::int8_t> w2;  // hidden x out
    std::vector<std::int32_t> b1, b2;
    // Test set, written by the training script: inputs, true labels and the
    // classes the Python simulation of the integer pipeline predicts.
    Matrix<std::int8_t> x;
    std::vector<int> labels, expected;

    static Model load(const std::string& path);
};

// Bit-exact reference of both layers: int32 logits, one row of `out` per image.
Matrix<std::int64_t> reference(const Model& model, const Matrix<std::int8_t>& x,
                               int acc_width = 32);

// The model on a TPU: weights and biases are written once, by the constructor.
class Mlp {
   public:
    struct Result {
        Matrix<std::int64_t> logits;  // images x out; columns >= classes are padding
        Perf layer1, layer2;
    };

    // batch: the most images one run() takes (<= cfg.max_m). Throws
    // std::invalid_argument if the model does not fit in the TPU's memories.
    Mlp(TpuDriver& tpu, const Model& model, int batch = 32);

    // x: images x PIXELS, at most batch images.
    Result run(const Matrix<std::int8_t>& x);

   private:
    TpuDriver& tpu_;
    const Model& model_;
    int batch_;
    int x_addr_ = 0, w1_addr_ = 0, h_addr_ = 0, w2_addr_ = 0, b2_addr_ = 0;
};

// Index of the largest of the first `classes` logits of `row`.
int predict(const Matrix<std::int64_t>& logits, std::size_t row, std::size_t classes);

// A drawn digit as network input. bitmap: BITMAP x BITMAP ink values (0 blank,
// 255 full), row-major, already cropped and scaled to the digit. Like sklearn's
// digits: threshold at half ink, count the set pixels of each 4x4 block (0..16),
// then quantize count / 16 with the input scale 1/127.
std::vector<std::int8_t> features(std::span<const std::uint8_t> bitmap);

}  // namespace tpu_host::digits
