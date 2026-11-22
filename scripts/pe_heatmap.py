"""Per-PE heatmap of useful cycles in one MATMUL, from the PeHeatmapSpec VCD.

PE(i,j) is busy at cycle t if the array's aIn.valid was high at t-i-j (skew)
and its effective weight is non-zero (gap cycles carry junk activations).

usage: python scripts/pe_heatmap.py [vcd] [png]
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

m, k, n = 16, 6, 5  # must match PeHeatmapSpec
vcd = sys.argv[1] if len(sys.argv) > 1 else "test_run_dir/pe_heatmap/TpuCore.vcd"
png = sys.argv[2] if len(sys.argv) > 2 else "figures/pe_heatmap.png"

# Signals of interest: array.io_aIn_valid and pes_i_j.wEff.
ids, scope = {}, []
lines = iter(open(vcd))
for l in lines:
    t = l.split() or [""]
    if t[0] == "$scope":
        scope.append(t[2])
    elif t[0] == "$upscope":
        scope.pop()
    elif t[0] == "$var":
        if scope[-1] == "array" and t[4] == "io_aIn_valid":
            ids[t[3]] = "valid"
        elif scope[-1] == "TpuCore" and t[4] == "clock":
            ids[t[3]] = "clk"
        elif t[4] == "wEff" and (p := re.fullmatch(r"pes_(\d+)_(\d+)", scope[-1])):
            ids[t[3]] = (int(p[1]), int(p[2]))
    elif t[0] == "$enddefinitions":
        break
assert "valid" in ids.values(), "io_aIn_valid not found in " + vcd

pes = [v for v in ids.values() if isinstance(v, tuple)]
rows, cols = 1 + max(i for i, _ in pes), 1 + max(j for _, j in pes)
cur = dict.fromkeys(ids.values(), 0)  # x/unset counts as 0
valid, weff = [], []  # one sample per clock cycle (VCD has a timestamp per clock edge)
for l in lines:
    if l[0] == "#" and cur["clk"]:
        valid.append(cur["valid"])
        weff.append(dict(cur))
    elif l[0] == "b":
        v, i = l[1:].split()
        if i in ids:
            cur[ids[i]] = int(v, 2) if set(v) <= {"0", "1"} else 0
    elif l[0] in "01" and l[1:].strip() in ids:
        cur[ids[l[1:].strip()]] = int(l[0])

heat = np.zeros((rows, cols), int)
for i in range(rows):
    for j in range(cols):
        for t in range(i + j, len(valid)):
            heat[i, j] += bool(valid[t - i - j] and weff[t][(i, j)])

assert heat.sum() == m * k * n, f"{heat.sum()} busy PE-cycles, expected m*k*n = {m * k * n}"
fig, ax = plt.subplots()
im = ax.imshow(heat, cmap="viridis")
for (i, j), v in np.ndenumerate(heat):
    ax.text(j, i, v, ha="center", va="center", color="w")
ax.set_xticks(range(cols))
ax.set_yticks(range(rows))
ax.set_xlabel("PE column")
ax.set_ylabel("PE row")
ax.set_title(f"Active cycles per PE, MATMUL m={m}, k={k}, n={n}")
fig.colorbar(im, label="active cycles")
os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
fig.savefig(png, dpi=150, bbox_inches="tight")
print(heat, "->", png)
