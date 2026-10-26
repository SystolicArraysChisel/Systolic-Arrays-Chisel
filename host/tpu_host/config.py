"""Hardware parameters the host needs to know.

These mirror SAConfig (src/main/scala/ifaces/SAConfig.scala) and BoardConfig
(src/main/scala/tpu/io/BoardConfig.scala). The FPGA cannot report them, so
they must be kept equal to the values the bitstream was generated with.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HwConfig:
    rows: int = 4  # PE rows (= banks = cols in the MVP)
    cols: int = 4  # PE columns
    in_depth: int = 4096  # int8 elements in the input memory (A and B)
    out_depth: int = 4096  # int32 words in the result memory (C)
    max_m: int = 64  # largest m one MATMUL accepts
    acc_width: int = 32  # bits of an accumulator
    baud: int = 115_200  # UART speed
    rx_fifo_depth: int = 16  # bytes RxFifo holds while the decoder is busy

    @property
    def banks(self) -> int:
        return max(self.rows, self.cols)
