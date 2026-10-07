"""Summarises kernel benchmark results, and compares two result sets.

  python compare.py summary <jitnn run>.jsonl [more runs ...] [--baselines baselines.jsonl] [--host <run without profiler>.jsonl]
  python compare.py diff <before runs prefix or file> <after runs prefix or file>

Several runs of the same calls (separate JVMs) are merged by the median of their medians. The
summary prints, per family, the device time per workload step weighted by how often the workloads
use each call, its share of the total, the time-weighted bandwidth and compute efficiency, and the
ratio to PyTorch where a baseline exists; then the 15 calls costing most. The diff flags every call
more than 3% slower after than before, and prints each family's time-weighted change.
"""
import glob
import json
import statistics
import sys
from collections import defaultdict


def key(r):
    return json.dumps([r["op"], r["iattr"], r["fattr"], r["in"], r["out"]])


def load(paths):
    runs = defaultdict(list)
    for p in paths:
        for f in sorted(glob.glob(p)) or [p]:
            for line in open(f):
                r = json.loads(line)
                runs[key(r)].append(r)
    merged = {}
    for k, rs in runs.items():
        r = dict(rs[0])
        r["device_us"] = statistics.median(x["device_us"] for x in rs)
        r["runs"] = len(rs)
        r["accepted_all"] = all(x["accepted"] for x in rs)
        r["power_capped_any"] = any(x["power_capped"] for x in rs)
        dev_s = r["device_us"] * 1e-6
        r["bandwidth_eff"] = r["bytes"] / dev_s / 1721e9 if dev_s > 0 else 0
        merged[k] = r
    return merged


def weight(r):
    return sum(r["uses"].values())


def summary(args):
    host = {}
    if "--host" in args:
        i = args.index("--host")
        for line in open(args[i + 1]):
            r = json.loads(line)
            host[key(r)] = r["host_us"]
        args = args[:i] + args[i + 2:]
    base = {}
    if "--baselines" in args:
        i = args.index("--baselines")
        for line in open(args[i + 1]):
            b = json.loads(line)
            base[b["key"]] = b
        args = args[:i] + args[i + 2:]
    rows = load(args)
    fam = defaultdict(lambda: dict(t=0.0, bw=0.0, ce=0.0, tc=0.0, jt=0.0, pt=0.0, n=0, rejected=0))
    total = 0.0
    for k, r in rows.items():
        t = r["device_us"] * weight(r)
        f = fam[r["family"]]
        f["t"] += t
        f["bw"] += r["bandwidth_eff"] * t
        if r["flops"] > 0:
            f["ce"] += r["compute_eff"] * t
            f["tc"] += t
        if k in base:
            f["jt"] += r["device_us"] * weight(r)
            f["pt"] += base[k]["torch_us"] * weight(r)
        f["n"] += 1
        f["rejected"] += 0 if r["accepted_all"] else 1
        total += t
    print(f"{'family':14s} {'calls':>5s} {'time/step ms':>12s} {'share':>6s} {'bandwidth':>9s} {'compute':>8s} {'vs PyTorch':>10s} {'not accepted':>12s}")
    for name, f in sorted(fam.items(), key=lambda x: -x[1]["t"]):
        bw = f["bw"] / f["t"] if f["t"] else 0
        ce = f"{f['ce'] / f['tc']:.0%}" if f["tc"] else "-"
        ratio = f"{f['jt'] / f['pt']:.2f}x" if f["pt"] else "-"
        print(f"{name:14s} {f['n']:5d} {f['t'] / 1e3:12.2f} {f['t'] / total:6.1%} {bw:9.0%} {ce:>8s} {ratio:>10s} {f['rejected']:12d}")
    print(f"{'total':14s} {len(rows):5d} {total / 1e3:12.2f}")
    if host:
        ht = sum(host[k] * weight(r) for k, r in rows.items() if k in host)
        print(f"host-to-host time per step, calls run one at a time (profiler off): {ht / 1e3:.2f} ms; device time is {total / ht:.0%} of it")
    print("\nmost costly calls (device time x uses):")
    for k, r in sorted(rows.items(), key=lambda x: -x[1]["device_us"] * weight(x[1]))[:15]:
        ratio = f"{r['device_us'] / base[k]['torch_us']:.2f}x PyTorch" if k in base else ""
        shapes = " ".join("x".join(map(str, i["shape"])) for i in r["in"])
        print(f"  {r['op']:20s} {r['device_us']:9.1f} us x{weight(r):<3d} bw {r['bandwidth_eff']:4.0%} {('ce %3.0f%%' % (100 * r['compute_eff'])) if r['flops'] else '':8s} {ratio:16s} {shapes}")


def diff(before, after):
    b, a = load([before]), load([after])
    fam = defaultdict(lambda: [0.0, 0.0])
    worse = []
    for k, r in a.items():
        if k not in b:
            continue
        t0, t1 = b[k]["device_us"], r["device_us"]
        fam[r["family"]][0] += t0 * weight(r)
        fam[r["family"]][1] += t1 * weight(r)
        if t1 > 1.03 * t0:
            worse.append((t1 / t0, r))
    for name, (t0, t1) in sorted(fam.items(), key=lambda x: -x[1][0]):
        print(f"{name:14s} {t0 / 1e3:10.2f} -> {t1 / 1e3:10.2f} ms/step  {t1 / t0 - 1:+.1%}")
    print(f"\n{len(worse)} calls more than 3% slower:")
    for ratio, r in sorted(worse, key=lambda x: -x[0])[:30]:
        print(f"  {ratio:5.2f}x {r['op']:20s} {' '.join('x'.join(map(str, i['shape'])) for i in r['in'])}")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(sys.argv[2:])
    else:
        diff(sys.argv[2], sys.argv[3])
