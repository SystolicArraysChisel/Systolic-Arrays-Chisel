# C++ host

Sends matrices to the FPGA over the USB-UART link, starts products, reads the
results back and checks them. Protocol: `src/main/scala/tpu/io/HostProtocol.scala`.

## Build and test

Needs CMake ≥ 3.20 and a C++20 compiler (MSYS2 MinGW g++ on Windows).
nlohmann/json and GoogleTest are downloaded by CMake.

On Windows, after installing MSYS2: `pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make`
in the MSYS2 shell, then add `C:\msys64\ucrt64\bin` to `PATH` (the executables need its DLLs).

```
cd host/cpp
cmake -S . -B build -G "MinGW Makefiles"
cmake --build build
./build/tpu_host_tests        # runs without hardware: the board checks are skipped
```

On macOS/Linux drop the `-G` option: `cmake -S . -B build && cmake --build build`.

## Use

```cpp
#include "tpu_host/tpu_driver.hpp"
using namespace tpu_host;

TpuDriver tpu = TpuDriver::open("COM3", HwConfig::load("build/4x4/config.json"));
auto [c, perf] = tpu.matmul(a_int8, b_int8);             // exact int32 results
FloatResult res = tpu.matmul_float(a, b, /*requant=*/true);  // quantize, run, dequantize
double u = res.perf.utilization(tpu.cfg());
```

`HwConfig` must match the bitstream: load the `config.json` written next to it.
Without a board, `TpuDriver(std::make_unique<FakeTpu>(cfg), cfg)` answers the
same protocol in software.

## On the board

From the repository root:

```
sbt "runMain tpu.board.EmitBoard 4x4"                     # generated/4x4/BoardTop.v (2x2, 4x4 or 8x8)
vivado -mode batch -source fpga/build.tcl -tclargs 4x4    # build/4x4/BoardTop.bit, config.json
vivado -mode batch -source fpga/program.tcl -tclargs 4x4  # lost at power off: reprogram
```

`program.tcl` also takes the path of a `.bit` file instead of a configuration name.

Then, from `host/cpp/`:

```
TPU_PORT=COM3 TPU_CONFIG=../../build/4x4/config.json ./build/tpu_host_tests --gtest_filter=Board.*
```

In PowerShell set the variables first: `$env:TPU_PORT = "COM3"; $env:TPU_CONFIG = "../../build/4x4/config.json"`.
`TPU_CONFIG` must be the `config.json` of the bitstream on the board.

`./build/run_board` prints a performance table (cycles, utilization, MAC/s) for
random square products, checked against the reference model: on the board with
`TPU_PORT` (and `TPU_CONFIG`) set as above, on `FakeTpu` without `TPU_PORT`.

The port is the "USB Serial Port" the Nexys4 DDR creates: `COM3` on Windows
(Device Manager, FTDI), `/dev/cu.usbserial-*` on macOS, `/dev/ttyUSB*` on Linux. A `timeout: got 0 of 9 reply bytes` usually means the FPGA is not
programmed.

## Digits demo

A 64-16-10 MLP for sklearn's handwritten digits: bias + leaky ReLU (slope 1/8) in layer 1,
chained on the TPU without going back to the host, bias and raw int32 logits in layer 2.
`demo/digits_model.json` is committed, so the demo runs without Python. To retrain it
(from the repository root):

```
python3 -m venv .venv; .venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python scripts/train_digits.py     # float 96.7 %, integer pipeline 96.4 % on the 450 test digits
.venv/bin/python scripts/train_shapes.py     # demo/shapes_model.json, about 3 minutes
```

`demo/shapes_model.json` is the same network for 10 shapes and symbols, trained on shapes
the script draws itself with the GUI's canvas and pen, converted to input the way the GUI
converts a drawing (99.9 % on 3,000 generated test shapes; generated, so this is optimistic
for human drawings). `./build/run_mlp demo/shapes_model.json` runs it like the digits.

A model file holds `name`, `classes` (2 to 16 names; `"0"`..`"9"` if absent), the int8
weights `w1` (64 x hidden) and `w2` (hidden x classes), the int32 biases `b1` and `b2`,
`shift1` and `leak_shift`, and a test set: `x` (int8 inputs), `labels` and `expected` (the
classes the Python simulation of the integer pipeline predicts). The model screen shows the PNG with the same name
(`demo/digits_model.png`) as its logo, or a generic icon if there is none.

Then, from `host/cpp/`: `./build/run_mlp` (FakeTpu, so the cycles are a model, not hardware),
or `TPU_PORT=COM3 TPU_CONFIG=../../build/4x4/config.json ./build/run_mlp` on the board.
It checks every batch's logits against the C++ reference and the predictions against the
Python simulation, then prints the accuracy (96.4 %), the cycles and the utilization per layer.

## User application

`tpu_app` (sources in `cpp/gui/`) is the graphical application: choose a model, draw with the
mouse, and the network classifies the drawing on the board or on the simulator.
It is built with the rest when CMake finds Qt 6; otherwise CMake prints
`Qt 6 not found: tpu_app is not built` and builds the other targets.

```
pacman -Syu                                  # MSYS2: Qt needs an up-to-date system
pacman -S mingw-w64-ucrt-x86_64-qt6-base     # or mingw-w64-x86_64-qt6-base: same environment as g++
cmake -S . -B build -G "MinGW Makefiles"     # from host/cpp, again if build/ already exists
cmake --build build --target tpu_app
./build/tpu_app
```

On macOS/Linux install Qt 6 with the package manager (`brew install qt`, `apt install qt6-base-dev`);
if CMake still does not find it, pass `-DCMAKE_PREFIX_PATH=<Qt dir>`.

- The first screen lists the models: the `.json` files in `cpp/demo/` (copy a new one there to
  offer it). **Open** goes to the drawing screen, **Models** comes back.
- Draw in the square: the digit is classified when the mouse is released. **Example** takes a
  digit from the test set. "TPU input" shows the 8x8 image that is sent.
- **Run on**: Simulator (FakeTpu, no board needed) or a serial port; the list is refreshed when
  it opens. The board must be programmed with the bitstream of the chosen **Array** size.
- **Program FPGA...** loads a bitstream with Vivado (`fpga/program.tcl`; Vivado on `PATH` or in
  its default install folder: `C:\AMDDesignTools`, `C:\Xilinx`, `/tools/Xilinx`, `/opt/Xilinx`).
  The file dialog opens on `bitstreams/<array>/BoardTop.bit` beside a released app, or
  `build/<array>/BoardTop.bit` in the source tree; if a `config.json` lies beside the chosen
  file, **Array** is set to match it. Where the app looks for its files:
  [docs/releasing.md](../docs/releasing.md#where-the-app-looks-for-its-files).

A drawing becomes network input the way sklearn's digits were made: crop to the ink, scale the
longer side to 32 pixels, then count the set pixels in each 4x4 block (0..16).

## Modules

| File | Content |
|---|---|
| `config` | `HwConfig`: array size, memory depths, baud rate; loaded from `config.json` |
| `protocol` | frame builders, STATUS parser, checks of the host guarantees |
| `matrix` | `Matrix<T>`: row-major, contiguous |
| `quant` | quantize / dequantize, scale and shift selection (round-half-up) |
| `reference` | bit-exact model of one MATMUL (Scala copy: `tpu.ref.RefModel`) |
| `link` | `Link`: the byte stream interface below the driver |
| `fake_tpu` | `FakeTpu`: software stand-in for the FPGA, used by the tests |
| `serial_link` | `SerialLink`: the serial port (Win32 or termios) |
| `tpu_driver` | `TpuDriver`: WRITE / READ / MATMUL / STATUS, whole products |
| `digits` | the drawing MLPs (digits, shapes): model loading, `Mlp` on a `TpuDriver`, reference, drawing to input |

## Reference vectors

`src/test/resources/ref_vectors.txt` is checked by both `tests/test_ref_vectors.cpp`
and the Scala `RefVectorsSpec`, so the two models stay bit-exact. It was
generated by the former Python host (`host/tpu_host/vectors.py`, see the git
history). One vector per line, fields separated by spaces, matrices row-major:

```
Q x scale q                                    quantize
S scale x...                                   choose_scale
P acc requant relu shift y                     post_process (requant, relu: 0/1)
K shift acc...                                 choose_shift
M m k n requant relu shift | a... | b... | c...  matmul_ref
FM hexframe aAddr bAddr cAddr m k n ldA ldB ldC requant relu shift
FW hexframe addr data...                       WRITE frame (data as signed int8)
FR hexframe addr count size                    READ frame
```
