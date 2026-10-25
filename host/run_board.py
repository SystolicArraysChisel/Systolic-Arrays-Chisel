"""Bring-up and measurements on the Nexys4 DDR.

    python run_board.py smoke COM5           # Tpu bitstream: check results against NumPy
    python run_board.py sweep COM5 -o perf.csv   # Tpu bitstream: cycles and utilization per size
    python run_board.py run COM5 --float --m 8 --k 16 --n 8 --requant --json out.json
                                             # one product, every intermediate value as JSON (GUI)

Use --fake instead of a port to run against FakeTpu (no board).
Exit code 0 when every check passes, 1 otherwise.
"""

import argparse
import csv
import sys
import time

import numpy as np

from tpu_host import (HwConfig, Tpu, TpuError, choose_scale, choose_shift, dequantize, matmul_acc,
                      matmul_ref, quantize, requant_scale)
from tpu_host import protocol as p
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


# ---- run ----

class RecordingLink:
    """Wraps a serial link and records every frame written and every byte read.

    Tpu writes each frame with a single write(), so the first byte of each
    write is the frame's opcode.
    """

    def __init__(self, link):
        self.link = link
        self.frames = []  # bytes of every frame sent, in order
        self.bytes_in = 0

    def write(self, data: bytes) -> int:
        self.frames.append(bytes(data))
        return self.link.write(data)

    def read(self, n: int) -> bytes:
        data = self.link.read(n)
        self.bytes_in += len(data)
        return data

    def close(self) -> None:
        self.link.close()


def decode_matmul(frame: bytes) -> dict:
    """Fields of a MATMUL frame (see protocol.Matmul.frame)."""
    names = ["a_addr", "b_addr", "c_addr", "m", "k", "n", "ld_a", "ld_b", "ld_c"]
    f = {name: int.from_bytes(frame[1 + 2 * i:3 + 2 * i], "little") for i, name in enumerate(names)}
    f["requant"] = bool(frame[19] & 1)
    f["relu"] = bool(frame[19] & 2)
    f["shift"] = frame[20]
    f["hex"] = frame.hex(" ")
    return f


def run_inputs(args, rng):
    """(A, B, is_float) from --example, --a/--b files, or random of the requested shape."""
    if args.example:
        return (np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 0, -1, 2]]),
                np.array([[1, 0], [0, 1], [1, 1], [2, -1]]), False)
    if args.a or args.b:
        if not (args.a and args.b):
            raise SystemExit("--a and --b go together")
        load = lambda path: np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=",", ndmin=2)
        a, b = load(args.a), load(args.b)
        is_float = args.float or not (np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer))
        return a, b, is_float
    if args.float:
        return rng.standard_normal((args.m, args.k)), rng.standard_normal((args.k, args.n)), True
    return rand_int8(rng, (args.m, args.k)), rand_int8(rng, (args.k, args.n)), False


def stats(err) -> dict:
    err = np.asarray(err, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(err), initial=0.0)),
            "mean_abs": float(np.mean(np.abs(err))) if err.size else 0.0,
            "rms": float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0}


def cmd_run(args) -> bool:
    """One product, with every intermediate value, as JSON (for the GUI)."""
    import json

    cfg = HwConfig()
    rng = np.random.default_rng(args.seed)
    a, b, is_float = run_inputs(args, rng)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise SystemExit(f"cannot multiply {a.shape} by {b.shape}")
    m, k = a.shape
    n = b.shape[1]
    out = {"command": sys.argv, "hw": cfg.__dict__, "m": m, "k": k, "n": n,
           "float": is_float, "requant": args.requant, "relu": args.relu}

    # host side: quantize (float inputs only) and choose the shift
    if is_float:
        sa, sb = choose_scale(a), choose_scale(b)
        qa, qb = quantize(a, sa), quantize(b, sb)
        out["a_float"], out["b_float"] = a.tolist(), b.tolist()
        out["scale_a"], out["scale_b"] = sa, sb
        out["quant_err_a"] = (a - dequantize(qa, sa)).tolist()
        out["quant_err_b"] = (b - dequantize(qb, sb)).tolist()
        out["quant_err_a_stats"] = stats(out["quant_err_a"])
        out["quant_err_b_stats"] = stats(out["quant_err_b"])
    else:
        qa, qb = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        if qa.min(initial=0) < -128 or qa.max(initial=0) > 127 or qb.min(initial=0) < -128 or qb.max(initial=0) > 127:
            raise SystemExit("integer inputs must be int8 (-128..127)")
    acc = matmul_acc(qa, qb, cfg.acc_width)
    if args.requant:
        shift = choose_shift(acc, cfg.acc_width) if args.shift is None else args.shift
    else:
        shift = 0
    out["a_q"], out["b_q"], out["acc"], out["shift"] = qa.tolist(), qb.tolist(), acc.tolist(), shift
    out["shift_auto"] = args.requant and args.shift is None
    c_ref = matmul_ref(qa, qb, requant=args.requant, relu=args.relu, shift=shift)

    # FPGA side
    if args.port == "--fake" or args.fake:
        raw = FakeTpu(cfg)
    else:
        import serial
        try:
            raw = serial.Serial(args.port, cfg.baud, timeout=0.5)
        except serial.SerialException as e:
            out["error"] = f"cannot open {args.port}: is the board plugged in and on? ({e})"
            if args.json:
                with open(args.json, "w") as fh:
                    json.dump(out, fh)
            print(f"FAIL  {out['error']}")
            return False
    link = RecordingLink(raw)
    tpu = Tpu(link, cfg, timeout=args.timeout)
    try:
        t0 = time.monotonic()
        c_hw, perf = tpu.matmul(qa, qb, requant=args.requant, relu=args.relu, shift=shift)
        wall = time.monotonic() - t0
        st = tpu.status()
    except TpuError as e:
        out["error"] = str(e)
        c_hw, perf, wall, st = None, None, None, None
    finally:
        tpu.close()

    if c_hw is not None:
        mismatches = np.argwhere(c_hw != c_ref)
        out["c_hw"] = c_hw.tolist()
        out["mismatches"] = int(len(mismatches))
        out["mismatch_positions"] = mismatches[:20].tolist()
        kp, np_ = -(-k // cfg.banks) * cfg.banks, -(-n // cfg.banks) * cfg.banks
        ideal = m * kp * np_ / (cfg.rows * cfg.cols)
        out["perf"] = {
            "cycles": perf.cycles, "busy_pe_cycles": perf.busy_pe_cycles,
            "utilization": perf.utilization(cfg), "ideal_cycles": ideal,
            "compute_s": perf.cycles / cfg.clock_hz, "wall_s": wall,
            "k_padded": kp, "n_padded": np_,
            "tiles": (kp // cfg.rows) * (np_ // cfg.cols) * -(-m // cfg.max_m),
        }
        out["status"] = {"busy": st.busy, "error": st.error, "overrun": st.overrun}
        if is_float:
            c_float = a @ b
            if args.relu:
                c_float = np.maximum(c_float, 0)
            scale_c = requant_scale(sa, sb, shift) if args.requant else sa * sb
            c_deq = dequantize(c_hw, scale_c)
            out["scale_c"] = scale_c
            out["c_float"] = c_float.tolist()
            out["c_dequant"] = c_deq.tolist()
            out["c_err"] = (c_float - c_deq).tolist()
            out["c_err_stats"] = stats(c_float - c_deq)
            peak = float(np.max(np.abs(c_float), initial=0.0))
            out["c_err_stats"]["max_rel_to_peak"] = out["c_err_stats"]["max_abs"] / peak if peak else 0.0
    out["c_ref"] = c_ref.tolist()

    ops = {p.OP_WRITE: "WRITE", p.OP_READ: "READ", p.OP_MATMUL: "MATMUL", p.OP_STATUS: "STATUS"}
    counts = {}
    for f in link.frames:
        name = ops.get(f[0], f"0x{f[0]:02x}")
        c = counts.setdefault(name, {"frames": 0, "bytes": 0})
        c["frames"] += 1
        c["bytes"] += len(f)
    out["traffic"] = {"per_command": counts, "bytes_out": sum(len(f) for f in link.frames),
                      "bytes_in": link.bytes_in}
    out["matmuls"] = [decode_matmul(f) for f in link.frames if f[0] == p.OP_MATMUL]

    ok = c_hw is not None and out["mismatches"] == 0
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh)
    if "error" in out:
        print(f"FAIL  {out['error']}")
    else:
        print(f"{'PASS' if ok else 'FAIL'}  {m}x{k}x{n}  {out['mismatches']} mismatches, "
              f"{perf.cycles} cycles, utilization {perf.utilization(cfg):.1%}, wall {wall:.2f} s")
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

    sp = sub.add_parser("run", help="one product, every intermediate value as JSON")
    common(sp)
    sp.add_argument("--m", type=int, default=8, help="rows of A")
    sp.add_argument("--k", type=int, default=8, help="columns of A = rows of B")
    sp.add_argument("--n", type=int, default=8, help="columns of B")
    sp.add_argument("--float", action="store_true", help="real inputs: quantize on the host")
    sp.add_argument("--example", action="store_true", help="the worked example of the workload document")
    sp.add_argument("--a", help="A from a .npy or .csv file (with --b)")
    sp.add_argument("--b", help="B from a .npy or .csv file (with --a)")
    sp.add_argument("--requant", action="store_true", help="requantize the results to int8 on the FPGA")
    sp.add_argument("--relu", action="store_true", help="apply ReLU on the FPGA")
    sp.add_argument("--shift", type=int, help="requantization shift (default: smallest without saturation)")
    sp.add_argument("--json", help="write every value to this JSON file")
    sp.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    return 0 if args.fn(args) else 1


if __name__ == "__main__":
    sys.exit(main())
