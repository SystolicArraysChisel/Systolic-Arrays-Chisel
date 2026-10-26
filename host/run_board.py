"""Bring-up and measurements on the Nexys4 DDR.

    python run_board.py smoke COM5           # Tpu bitstream: check results against NumPy
    python run_board.py sweep COM5 -o perf.csv   # Tpu bitstream: cycles and utilization per size

Use --fake instead of a port to run against FakeTpu (no board).
Exit code 0 when every check passes, 1 otherwise.
"""

import argparse
import csv
import sys
import time

import numpy as np

from tpu_host import HwConfig, Tpu, TpuError, matmul_ref
from tpu_host.sim import FakeTpu


def open_tpu(args, cfg: HwConfig) -> Tpu:
    if args.port == "--fake" or args.fake:
        return Tpu(FakeTpu(cfg), cfg)
    return Tpu.open(args.port, cfg, timeout=args.timeout)


def rand_int8(rng, shape):
    return rng.integers(-128, 128, size=shape, dtype=np.int64)


# ---- smoke ----

def smoke_cases(rng, cfg: HwConfig):
    """(name, A, B, requant, relu, shift): from one tile up to the memory limits."""
    r = cfg.rows
    yield "worked example", np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 0, -1, 2]]), \
        np.array([[1, 0], [0, 1], [1, 1], [2, -1]]), False, False, 0
    yield "one tile", rand_int8(rng, (r, r)), rand_int8(rng, (r, r)), False, False, 0
    yield "extreme values", np.full((r, r), -128), np.full((r, r), -128), False, False, 0
    yield "tiles over K and N", rand_int8(rng, (20, 2 * r)), rand_int8(rng, (2 * r, 3 * r)), False, False, 0
    yield "padding of K and N", rand_int8(rng, (5, 7)), rand_int8(rng, (7, 6)), False, False, 0
    yield "m = max_m", rand_int8(rng, (cfg.max_m, r)), rand_int8(rng, (r, 2 * r)), False, False, 0
    yield "m > max_m (split)", rand_int8(rng, (cfg.max_m + 9, r)), rand_int8(rng, (r, r)), False, False, 0
    yield "requant", rand_int8(rng, (6, 2 * r)), rand_int8(rng, (2 * r, r)), True, False, 9
    yield "requant + relu", rand_int8(rng, (6, 2 * r)), rand_int8(rng, (2 * r, r)), True, True, 7
    yield "32x32x32", rand_int8(rng, (32, 32)), rand_int8(rng, (32, 32)), False, False, 0


def cmd_smoke(args) -> bool:
    cfg = HwConfig()
    rng = np.random.default_rng(args.seed)
    tpu = open_tpu(args, cfg)
    ok = True
    try:
        s = tpu.status()
        print(f"STATUS busy={s.busy} error={s.error} overrun={s.overrun}")
        if s.busy or s.error or s.overrun:
            print("note: flags were set before the test (they clear on this STATUS)")

        for name, a, b, requant, relu, shift in smoke_cases(rng, cfg):
            try:
                c, perf = tpu.matmul(a, b, requant=requant, relu=relu, shift=shift)
            except TpuError as e:
                print(f"FAIL  {name}: {e}")
                ok = False
                continue
            want = matmul_ref(a, b, requant=requant, relu=relu, shift=shift)
            m, k, n = a.shape[0], a.shape[1], b.shape[1]
            if np.array_equal(c, want):
                print(f"PASS  {name:20s} {m}x{k}x{n}  {perf.cycles} cycles, "
                      f"utilization {perf.utilization(cfg):.1%}")
            else:
                bad = np.argwhere(c != want)
                i, j = bad[0]
                print(f"FAIL  {name:20s} {m}x{k}x{n}  {len(bad)} wrong elements, "
                      f"first C[{i}][{j}] = {c[i, j]}, expected {want[i, j]}")
                ok = False

        x = rng.standard_normal((16, 24))
        w = rng.standard_normal((24, 8))
        res = tpu.matmul_float(x, w, requant=True)
        err = np.abs(res.c - x @ w).max() / np.abs(x @ w).max()
        print(f"INFO  float 16x24x8 with requant: max error {err:.1%} of the largest |C|")
    finally:
        tpu.close()
    print("all checks passed" if ok else "SOME CHECKS FAILED")
    return ok


# ---- sweep ----

def cmd_sweep(args) -> bool:
    """Square products of growing size: cycles on the FPGA vs the ideal m*k*n / (R*C)."""
    cfg = HwConfig()
    rng = np.random.default_rng(args.seed)
    tpu = open_tpu(args, cfg)
    rows = []
    ok = True
    try:
        for size in args.sizes:
            a, b = rand_int8(rng, (size, size)), rand_int8(rng, (size, size))
            t0 = time.monotonic()
            c, perf = tpu.matmul(a, b)
            wall = time.monotonic() - t0
            correct = np.array_equal(c, matmul_ref(a, b))
            ok &= correct
            ideal = size ** 3 / (cfg.rows * cfg.cols)
            rows.append({
                "m": size, "k": size, "n": size,
                "cycles": perf.cycles, "busy_pe_cycles": perf.busy_pe_cycles,
                "ideal_cycles": ideal, "utilization": perf.utilization(cfg),
                "wall_s": round(wall, 3), "correct": correct,
            })
            print(f"{size:4d}  {perf.cycles:8d} cycles  ideal {ideal:8.0f}  "
                  f"utilization {perf.utilization(cfg):6.1%}  wall {wall:6.2f} s  "
                  f"{'ok' if correct else 'WRONG'}")
    finally:
        tpu.close()
    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"written {args.output}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("port", nargs="?", default="--fake", help="serial port, e.g. COM5 or /dev/ttyUSB1")
        p.add_argument("--fake", action="store_true", help="use FakeTpu instead of the board")
        p.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for a reply")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("smoke", help="check Tpu results against NumPy")
    common(p)
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("sweep", help="cycles and utilization for square products")
    common(p)
    p.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 12, 16, 24, 32, 40])
    p.add_argument("-o", "--output", help="CSV file for the results")
    p.set_defaults(fn=cmd_sweep)

    args = ap.parse_args(argv)
    return 0 if args.fn(args) else 1


if __name__ == "__main__":
    sys.exit(main())
