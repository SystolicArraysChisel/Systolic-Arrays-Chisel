"""PE utilization vs matrix size on the board, from saved run_board outputs.

Each input is the stdout of host/cpp/build/run_board for one bitstream:
a header line "board, <rows>x<cols> at <MHz> MHz", then one row per square
product: size, cycles, busy PE-cycles, util %, MAC/s.

Utilization here counts only useful MACs, size^3 / (rows * cols * cycles), so
padded PEs do not count as busy (run_board's own "util %" includes them).

usage: python scripts/board_utilization.py [run_board outputs...] [-o png]
default: docs/data/run_board_*.txt -> docs/figures/board_utilization.png
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(path):
    """(name, PEs, MHz, [(size, cycles), ...]) from one run_board output."""
    with open(path) as f:
        lines = f.read().splitlines()
    head = re.match(r"(\w+), (\d+)x(\d+) at (\d+) MHz", lines[0])
    assert head, f"{path}: no 'board, RxC at N MHz' header"
    assert head[1] == "board", f"{path}: not a board run ({head[1]})"
    rows, cols, mhz = int(head[2]), int(head[3]), int(head[4])
    points = [tuple(int(v) for v in l.split()[:2]) for l in lines[2:] if l.strip()]
    return f"{rows}x{cols}", rows * cols, mhz, points


ap = argparse.ArgumentParser()
ap.add_argument("inputs", nargs="*", default=sorted(glob.glob("docs/data/run_board_*.txt")))
ap.add_argument("-o", "--out", default="docs/figures/board_utilization.png")
args = ap.parse_args()
runs = sorted((parse(p) for p in args.inputs), key=lambda r: r[1])
assert runs, "no run_board outputs found"

fig, ax = plt.subplots(figsize=(8, 4.5))
for k, (name, pes, mhz, points) in enumerate(runs):
    sizes = [n for n, _ in points]
    util = [100 * n**3 / (pes * c) for n, c in points]
    gmacs = [n**3 / c * mhz / 1000 for n, c in points]
    shade = 0.35 + 0.65 * k / max(len(runs) - 1, 1)  # one hue, darker = larger array
    ax.plot(sizes, util, "o-", color=(0.1, 0.4, 0.8, shade), label=name)
    ax.annotate(f"{name}: {util[-1]:.1f} % · {gmacs[-1]:.2f} of {pes * mhz / 1000:.1f} GMAC/s",
                (sizes[-1], util[-1]), xytext=(8, 0), textcoords="offset points", va="center")
    print(f"{name}: " + ", ".join(f"{n}: {u:.1f} % ({g:.2f} GMAC/s)"
                                  for n, u, g in zip(sizes, util, gmacs)))

ax.set_xscale("log", base=2)
ax.set_xticks(sizes, [f"{n}×{n}×{n}" for n in sizes])
ax.set_ylim(0, 100)
ax.set_xlabel("Square int8 MATMUL size (m × k × n)")
ax.set_ylabel("Useful MACs, % of peak")
ax.set_title(f"PE utilization on the board at {runs[0][2]} MHz (STATUS cycles, UART excluded)")
ax.grid(axis="y", alpha=0.3)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
fig.savefig(args.out, dpi=150, bbox_inches="tight")
print("->", args.out)
