"""Compares kernel variants from a sweep: one kernel benchmark file per variant (sweep/<variant>.jsonl),
the same calls in each. Prints each variant's time-weighted total, the total when every call takes
its fastest variant, and per call the winner and its gain over the reference variant.

usage: python sweep.py <dir> [reference variant, default auto]
"""
import glob
import json
import os
import sys
from collections import defaultdict


def main(d, ref="auto"):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        v = os.path.basename(f)[:-6]
        runs[v] = {json.dumps([r["op"], r["iattr"], r["in"]]): r for r in map(json.loads, open(f))}
    keys = set.intersection(*(set(r) for r in runs.values()))
    w = {k: sum(runs[ref][k]["uses"].values()) for k in keys}
    print(f"{len(keys)} calls in all {len(runs)} variants")
    for v, rs in sorted(runs.items(), key=lambda x: sum(x[1][k]["device_us"] * w[k] for k in keys)):
        t = sum(rs[k]["device_us"] * w[k] for k in keys)
        print(f"  {v:16s} {t / 1e3:10.2f} ms/step")
    best = {k: min(runs, key=lambda v: runs[v][k]["device_us"]) for k in keys}
    tb = sum(runs[best[k]][k]["device_us"] * w[k] for k in keys)
    tr = sum(runs[ref][k]["device_us"] * w[k] for k in keys)
    print(f"  {'best per call':16s} {tb / 1e3:10.2f} ms/step  ({tr / tb:.2f}x faster than {ref})")
    wins = defaultdict(float)
    for k in keys:
        wins[best[k]] += runs[ref][k]["device_us"] * w[k]
    print("time share won by each variant:", ", ".join(f"{v} {t / tr:.0%}" for v, t in sorted(wins.items(), key=lambda x: -x[1])))
    print(f"\n{'op':8s} {'shapes':38s} {'uses':>4s} {ref + ' us':>10s} {'best us':>9s} {'gain':>6s} {'TF/s':>6s} winner")
    for k in sorted(keys, key=lambda k: -runs[ref][k]["device_us"] * w[k])[:30]:
        r0, rb = runs[ref][k], runs[best[k]][k]
        shapes = " ".join("x".join(map(str, i["shape"])) for i in r0["in"])
        tf = rb["flops"] / (rb["device_us"] * 1e-6) / 1e12 if rb["device_us"] else 0
        print(f"{r0['op']:8s} {shapes[:38]:38s} {w[k]:4d} {r0['device_us']:10.1f} {rb['device_us']:9.1f} {r0['device_us'] / rb['device_us']:6.2f} {tf:6.1f} {best[k]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "auto")
