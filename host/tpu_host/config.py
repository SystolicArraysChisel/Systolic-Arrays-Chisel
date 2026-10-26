"""Hardware parameters the host needs to know.

These mirror SAConfig (src/main/scala/ifaces/SAConfig.scala) and BoardConfig
(src/main/scala/tpu/io/BoardConfig.scala). The FPGA cannot report them, so
they must be kept equal to the values the bitstream was generated with:
EmitBoard writes them next to each bitstream as config.json (HwConfig.load).
"""

import json
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class HwConfig:
    rows: int = 4  # PE rows (= banks = cols in the MVP)
    cols: int = 4  # PE columns
    in_depth: int = 4096  # int8 elements in the input memory (A and B)
    out_depth: int = 4096  # int32 words in the result memory (C)
    max_m: int = 64  # largest m one MATMUL accepts
    acc_width: int = 32  # bits of an accumulator
    baud: int = 115_200  # UART speed
    clock_hz: int = 50_000_000  # core clock (BoardConfig.nexysTpu), to turn cycles into time
    rx_fifo_depth: int = 16  # bytes RxFifo holds while the decoder is busy

    name: str = "4x4"  # configuration name (EmitBoard.configs)

    @classmethod
    def load(cls, path: str) -> "HwConfig":
        """The config.json EmitBoard writes next to a bitstream."""
        with open(path) as fh:
            data = json.load(fh)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def banks(self) -> int:
        return max(self.rows, self.cols)
