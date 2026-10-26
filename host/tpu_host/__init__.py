"""Python host for the systolic-array accelerator."""

from .config import HwConfig
from .device import FloatResult, Perf, Tpu, TpuError
from .protocol import Matmul, ProtocolError, Status
from .quant import choose_scale, choose_shift, dequantize, quantize, requant_scale, requantize
from .reference import matmul_acc, matmul_ref, post_process
