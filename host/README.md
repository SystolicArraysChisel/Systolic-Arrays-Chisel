# Python host

Sends matrices to the FPGA over the USB-UART link, starts products, reads the
results back and checks them. Protocol: `src/main/scala/tpu/io/HostProtocol.scala`;
numeric rules: `tpu_host/quant.py` and `tpu_host/reference.py`.

## Setup

```
cd host
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows
.venv\Scripts\python -m pytest                             # runs without hardware
```

## Use

```python
import numpy as np
from tpu_host import Tpu

tpu = Tpu.open("COM5")                 # 115200 baud, see HwConfig
c, perf = tpu.matmul(a_int8, b_int8)   # exact int32 results
res = tpu.matmul_float(a, b, requant=True)   # quantize, run, dequantize
print(res.c, res.perf.cycles, res.perf.utilization(tpu.cfg))
```

`HwConfig` must match the `SAConfig` / `BoardConfig` of the bitstream.

## On the board

From the repository root:

```
sbt "runMain tpu.board.EmitBoard"                # generated/BoardTop.v
vivado -mode batch -source fpga/build.tcl        # build/BoardTop.bit, build/reports/, build/summary.txt
vivado -mode batch -source fpga/program.tcl
```

Then, from `host/`:

```
.venv\Scripts\python run_board.py smoke COM5              # Tpu: results equal NumPy
.venv\Scripts\python run_board.py sweep COM5 -o perf.csv  # Tpu: cycles and utilization per size
```

Both also run with `--fake` (FakeTpu, no board). The COM port is the
"USB Serial Port" the Nexys4 DDR creates (Device Manager, FTDI).

## Modules

| File | Content |
|---|---|
| `protocol.py` | frame builders, STATUS parser, checks of the host guarantees |
| `quant.py` | quantize / dequantize, scale and shift selection (round-half-up) |
| `reference.py` | bit-exact model of one MATMUL (Scala copy: `tpu.ref.RefModel`) |
| `device.py` | `Tpu`: WRITE / READ / MATMUL / STATUS over a serial link |
| `sim.py` | `FakeTpu`: software stand-in for the FPGA, used by the tests |
| `vectors.py` | writes `src/test/resources/ref_vectors.txt`, checked by Scala |

After changing `quant.py` or `reference.py`, run `python -m tpu_host.vectors`
and commit the new `ref_vectors.txt`; both test suites fail until you do.
