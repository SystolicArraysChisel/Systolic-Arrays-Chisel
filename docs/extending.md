# Running your own network

What the accelerator can run today, its limits, and the three ways to add a
model: a new model file for `tpu_app`, a new network written against the host
library, or a change to the hardware. For the overview see the
[README](../README.md); for how the blocks work, [architecture](architecture.md).

## What the hardware computes

One operation, MATMUL: `C = post(A · B + bias)`, with

- A (m × k) and B (k × n) int8 matrices in the input memory,
- int32 accumulation and an optional int32 bias per output column,
- `post`: optional requantization to int8 (rounding right shift, saturation),
  then optional ReLU or Leaky ReLU,
- C written to the result memory (int32 or int8), or back into the input memory
  (`toInput`), where the next MATMUL can use it as its A without a round trip
  through the host.

Everything else a network needs (softmax, normalisation, pooling, reshaping,
argmax) runs on the host. Matrix products are where nearly all the arithmetic
of a fully connected or convolutional network is, and a convolution becomes a
matrix product by unrolling its input patches (im2col).

## Current limits

Values for the bitstreams built by `EmitBoard` (2x2, 4x4, 8x8). The parameter
names are those of `SAConfig` and `BoardConfig`.

| Limit | Value | Where it comes from | Can it change? |
|---|---|---|---|
| Number types | int8 inputs and weights, int32 sums and biases | `inWidth`, `accWidth`; the protocol sends one byte per element | Hardware and protocol work |
| Input memory (A, B and chained layers) | 4096 int8 elements | `inDepth` | Up to 65,536 by changing the parameter (2-byte addresses); beyond that, protocol and decoder changes |
| Result memory | 4096 int32 elements | `outDepth` | Parameter |
| Bias memory | 256 int32 entries | `biasDepth` | Parameter |
| Rows of A per MATMUL | m ≤ 64 | `maxM` (accumulator memory) | Parameter; the host can also split A into several MATMULs |
| Shapes and addresses | k and n padded to a multiple of the bank count (= array size); addresses and leading dimensions of A and B multiples of it | banked input memory | No: pad on the host |
| Requantization | one power-of-two shift per MATMUL, symmetric, per tensor | MATMUL `shift` field | Hardware work for per-channel scales |
| Activations | ReLU, Leaky ReLU with slope 2^−s (s = 0..7) | Accumulators | Hardware work for others (tanh, sigmoid...) |
| Array | square, power-of-two side, one product per PE per cycle | `rows == cols`, `vecWidth = 1` | Hardware work |
| Control | the host sends every command; each MATMUL is a 23-byte frame, then STATUS polling | protocol | Hardware work for an on-chip program |
| Link | 2 Mbaud UART, about 200 KB/s | `BoardConfig.baud` | Limited by the USB-UART chip |
| Programming the FPGA | Vivado on Windows or Linux; lost at power off | Vivado, Nexys4 DDR | No |

In practice the input memory decides what fits: weights, the input batch and
any chained intermediate layer must fit in 4096 bytes together.

## 1. A new model for `tpu_app`

The app runs any **64 → hidden → classes MLP on 8×8 drawings**, the network
shape of the two included models. Adding one needs no C++:

1. **Train** it in Python. `scripts/train_shapes.py` is the template: build
   inputs in 0..1 (64 values per drawing), then call `train()` and `export()`
   from `scripts/train_digits.py`. They train in float, quantize to the
   hardware's integer pipeline, check the integer accuracy and write the model
   file.
2. **Put the file** in `host/cpp/demo/` (`<name>.json`) and, if you like, a
   logo next to it (`<name>.png`, square). The app lists every model in that
   folder.
3. **Check** it bit-exact: `./build/run_mlp demo/<name>.json` runs the stored
   test set on FakeTpu, or on the board with `TPU_PORT` set, and compares every
   result with the C++ reference and the Python simulation.

If your inputs are drawings, generate or convert them exactly the way the app
converts the canvas (crop to the ink, scale the longer side to 32 pixels, count
the inked pixels in each 4×4 block): `features()` in `train_shapes.py` is that
conversion in Python.

**Model file.** JSON with `name`, `classes` (2 to 16 names), the int8 weights
`w1` (64 × hidden) and `w2` (hidden × classes), the int32 biases `b1` and `b2`,
`shift1` (requantization of layer 1), `leak_shift`, and a test set: `x` (int8
inputs), `labels` and `expected` (what the Python simulation predicts).

**Size.** The hidden and output widths are padded to 16. The app runs one image
at a time, which fits up to 48 hidden neurons in 4096 bytes. `run_mlp` processes
batches of 32 images, which fits 16.

## 2. A different network, with the host library

For another network shape (more layers, a CNN through im2col...), write the
host side in C++ against `TpuDriver` (`host/cpp/include/tpu_host/tpu_driver.hpp`);
`host/cpp/src/digits.cpp` is a complete example of a two-layer network.

```cpp
TpuDriver tpu = TpuDriver::open("COM3", HwConfig::load("build/4x4/config.json"));
tpu.write(W1_ADDR, w1);                 // int8 weights, once
tpu.write_bias(0, b1);                  // int32 biases, once
tpu.write(X_ADDR, x);                   // the input batch
tpu.run(Matmul{.a_addr = X_ADDR, .b_addr = W1_ADDR, .c_addr = H_ADDR,
               .m = m, .k = 64, .n = 16, .ld_a = 64, .ld_b = 16, .ld_c = 16,
               .requant = true, .relu = true, .to_input = true, .shift = shift1,
               .bias = true, .bias_addr = 0});      // layer 1 stays on chip
tpu.run(Matmul{/* layer 2 reads H_ADDR */});
auto logits = tpu.read(0, m * 16, 4);   // int32 results
```

Steps that work:

- **Quantize** as the included models do: symmetric int8 per tensor
  (`max|w|` maps to 127), biases as int32 at the scale `s_x · s_w`, and for a
  chained layer the smallest shift at which no activation saturates
  (`choose_shift` in `quant.hpp`). Simulate this integer pipeline in Python to
  know the accuracy before touching the board.
- **Lay out the memory** by hand: every address and leading dimension of A and
  B a multiple of 16 works on all three array sizes. `Matmul::check` rejects
  anything the hardware would not accept, with the reason.
- **Check** each layer against `matmul_ref` (`reference.hpp`) on `FakeTpu`
  first; it answers the same protocol in software. Then run on the board.

If the network does not fit in 4096 bytes, raise `inDepth` (up to 65,536) for a
new configuration in `EmitBoard.configs` and build its bitstream: it is a
parameter, not new hardware. The input memory uses block RAM, of which the
FPGA has plenty (the 8x8 build uses 8 of 135).

## 3. Changing the hardware

Some networks need more than the current datapath: higher precision (int16),
other activations (tanh, sigmoid), per-channel scales, memories above 64 KB
(wider protocol addresses), or running a whole sequence of layers without the
host (an on-chip program). These are changes to the Chisel generator. The
rules the project follows for them are in
[design decisions](design-decisions.md#5-engineering-rules-followed): new
features default off, and every hardware change updates both reference models
and `FakeTpu` with tests on both sides.
