"""Builds a GEMM tuning table (jitnn-core/src/main/resources/tuning/<backend>-<device>.json) from a
sweep: one kernel benchmark file per variant, <dir>/<variant>.jsonl, the same calls in each, with
"auto" the built-in choice. Calls are grouped by shape class (Tuning.gemmClass: ceiling log2 of M,
N and K, and the transposes); a class gets an entry only when one variant beats auto's total for
the class by more than the margin, so every entry is a measured win, and only for classes of at
least 0.1 ms per step (below that, timing noise picks the winner). A call that the class's
variant makes slower than auto by more than the margin gets an exact-shape entry with its own best
variant.

usage: python tune.py <sweep dir> <device file name, e.g. cuda-nvidia-a100-80gb-pcie> [margin, default 0.03]
"""
import datetime
import glob
import json
import math
import os
import sys
from collections import defaultdict


MIN_CLASS_US = 100.0


def log2(x):
    return 0 if x <= 1 else math.ceil(math.log2(x))


def gemm_dims(r):
    a = r["iattr"]
    if r["op"] == "linear":
        m, n, k, ta, tb = a[0], a[1], a[2], 0, 1
    elif r["op"] == "matmul" and len(a) == 5:
        m, n, k, ta, tb = a
    else:
        return None  # batched products take their own kernel
    return m, n, k, ta, tb


def gemm_class(r):
    d = gemm_dims(r)
    if d is None:
        return None
    m, n, k, ta, tb = d
    return f"m{log2(m)}n{log2(n)}k{log2(k)}t{ta}{tb}"


def gemm_shape(r):
    m, n, k, ta, tb = gemm_dims(r)
    return f"{m}x{n}x{k}t{ta}{tb}"


def main(d, device, margin=0.03):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        runs[os.path.basename(f)[:-6]] = {json.dumps([r["op"], r["iattr"], r["in"]]): r for r in map(json.loads, open(f))}
    if "auto" not in runs:
        sys.exit("the sweep needs auto.jsonl, the built-in choice")
    keys = set.intersection(*(set(r) for r in runs.values()))
    totals = defaultdict(lambda: defaultdict(float))
    for k in keys:
        r = runs["auto"][k]
        c = gemm_class(r)
        if c is None:
            continue
        uses = sum(r["uses"].values())
        for v, rs in runs.items():
            totals[c][v] += rs[k]["device_us"] * uses
    table = {}
    before = after = 0.0
    for c, t in sorted(totals.items()):
        best = min(t, key=t.get)
        before += t["auto"]
        # classes under MIN_CLASS_US per step are a few microseconds per call, where timing noise
        # decides the winner: they keep the built-in choice
        if best != "auto" and t["auto"] >= MIN_CLASS_US and t[best] < t["auto"] * (1 - margin):
            table[c] = best
            after += t[best]
        else:
            after += t["auto"]
    # exact-shape overrides: a call the class's variant makes slower than auto by more than the
    # margin gets its own best variant (auto included), so no measured shape loses to the table
    overrides = {}
    for k in keys:
        r = runs["auto"][k]
        c = gemm_class(r)
        if c is None or c not in table:
            continue
        mine = runs[table[c]][k]["device_us"]
        if mine > r["device_us"] * (1 + margin):
            best = min(runs, key=lambda v: runs[v][k]["device_us"])
            overrides[gemm_shape(r)] = best
            uses = sum(r["uses"].values())
            after += (runs[best][k]["device_us"] - mine) * uses
    table.update(sorted(overrides.items()))
    out = {
        "device": device,
        "measured": datetime.date.today().isoformat(),
        "source": "scripts/bench/tune.py over " + ", ".join(sorted(runs)) + f"; {len(keys)} calls, margin {margin}",
        "gemm": table,
    }
    path = os.path.join("jitnn-core/src/main/resources/tuning", device + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"{len(totals)} shape classes, {len(table) - len(overrides)} class entries and {len(overrides)} exact-shape overrides -> {path}")
    print(f"GEMM time per step over the sweep: auto {before / 1e3:.2f} ms, with the table {after / 1e3:.2f} ms ({before / after:.3f}x)")
    for c, v in overrides.items():
        print(f"  override {c:16s} {v}")
    for c, v in table.items():
        if c in overrides:
            continue
        t = totals[c]
        print(f"  {c:16s} {v:14s} {t['auto'] / 1e3:8.2f} -> {t[v] / 1e3:8.2f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.03)
