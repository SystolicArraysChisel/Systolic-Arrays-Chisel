# Timing and resource notes (MVP build, 24 Oct 2026)

First full Vivado build of `BoardTop` + `Tpu` (4x4 array, maxM = 64, Vivado
2026.1, xc7a100tcsg324-1). Not fixed for the MVP demo: the core is run at
**50 MHz** from an MMCM instead (`ClockGen`, `BoardConfig.nexysTpu`).

## Timing at 100 MHz (routed)

- WNS **-4.57 ns** (worst path ~14.1 ns, so ~70 MHz max), TNS ~ -5400 ns:
  thousands of failing endpoints, not one path.
- Worst path: `acc/writeCounter` -> `buf_/resultMem_*` (24 logic levels,
  15 CARRY4). In `Accumulators.sWrite`, one cycle does:
  1. `accReg(writeCounter)`: a 64:1 mux over maxM rows (MUXF7/MUXF8),
  2. rounding add, variable right shift, saturation compare (carry chains),
  3. ReLU mux, then the write into the result memory.
- Same pattern in `sAccum`: `accReg(rowCounter)` read, add, write back, in
  one cycle.

### Fix ideas

- Pipeline the write phase: register the selected row, then requant, then
  write (2-3 stages; `resWr` is Decoupled, so the latency is free).
- Move `accReg` into a `SyncReadMem` (BRAM) with a pipelined
  read-modify-write, as the TODO at the end of `Accumulators.scala` says. That
  also removes the 64:1 mux and ~8k flip-flops.
- Re-check `Sequencer` (`kOff * ldB`) and `DataMover` (`(rows-1) * ldB`)
  address multipliers once the accumulators are fixed.

## Resources (post-synthesis, 100 MHz build)

| Resource | Used | Available |
|---|---|---|
| LUT | 10 672 | 63 400 (16.8 %) |
| FF | 9 835 | 126 800 (7.8 %) |
| BRAM tile | 2 | 135 |
| DSP | 2 | 240 |

- **The PE multipliers are not in DSPs**: 16 MACs, only 2 DSPs used (the
  address multipliers). Vivado builds the 8x8 products from LUTs. To match
  `SAConfig.dspCount` and for the scaling plots, force DSPs (`use_dsp = "yes"`
  on the PE, or a DSP48E1 black box).
- **The result memory is in LUTRAM, not BRAM** (`resultMem_*_reg_960_1023_*`
  are 64-deep distributed RAM slices). Probably because the host read port
  selects the column after the read. Worth checking with the BRAM fix above.
- Vivado warns that the input-memory BRAMs may collide on same-address read
  and write; it cannot happen (host writes and data-mover reads never overlap
  in time).

Reports of that build: `build/reports-100MHz/` (not committed).
