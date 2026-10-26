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
