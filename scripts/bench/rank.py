"""Step 2 of the kernel optimisation plan: the families ranked by measured device time, overall and
per workload, from the end-to-end model profiles (ModelProfileAccelTest), checked against the
replayed calls of the kernel benchmark.

usage: python rank.py <models profiled>.jsonl <models wall>.jsonl "<kernel runs glob>" [top N]

Prints: per workload, the step's wall time, its device time and the replayed calls' sum; the family
shares per workload; the families ranked over all workloads (and over inference and training
separately); and the indices of the N calls costing most, for the Nsight Systems pass.
"""
import glob
import json
import statistics
import sys
from collections import defaultdict


def jsonl(path):
    out = []
    for f in sorted(glob.glob(path)) or [path]:
        out += [json.loads(l) for l in open(f)]
    return out


def main(prof_path, wall_path, runs_glob, top=25):
    prof = {r["workload"]: r for r in jsonl(prof_path)}
    wall = {r["workload"]: r for r in jsonl(wall_path)}
    calls = defaultdict(list)
    for r in jsonl(runs_glob):
        calls[json.dumps([r["op"], r["iattr"], r["fattr"], r["in"], r["out"]])].append(r)
    replay = defaultdict(float)
    rows = []
    for k, rs in calls.items():
        r = dict(rs[0])
        r["device_us"] = statistics.median(x["device_us"] for x in rs)
        rows.append(r)
        for w, n in r["uses"].items():
            replay[w] += r["device_us"] * n / 1e3

    print(f"{'workload':24s} {'wall ms':>9s} {'device ms':>10s} {'replayed ms':>11s} {'replay/model':>12s}")
    for w, p in prof.items():
        wl = wall.get(w, {}).get("step_ms", float("nan"))
        print(f"{w:24s} {wl:9.2f} {p['device_ms']:10.2f} {replay[w]:11.2f} {replay[w] / p['device_ms']:12.2f}")

    def shares(names):
        fam = defaultdict(float)
        for w in names:
            for op, o in prof[w]["ops"].items():
                fam[o["family"]] += o["device_ms"]
        return fam

    print("\nfamily share per workload (end-to-end device time):")
    fams = sorted({o["family"] for p in prof.values() for o in p["ops"].values()})
    print(f"{'workload':24s} " + " ".join(f"{f[:11]:>11s}" for f in fams))
    for w in prof:
        s = shares([w])
        t = sum(s.values())
        print(f"{w:24s} " + " ".join(f"{s[f] / t:11.0%}" if s[f] else f"{'-':>11s}" for f in fams))

    for title, names in (("all workloads", list(prof)), ("inference", [w for w in prof if "infer" in w]), ("training", [w for w in prof if "train" in w])):
        s = shares(names)
        t = sum(s.values())
        print(f"\nranking, {title} ({t:.1f} ms of device time per round of steps):")
        for i, (f, v) in enumerate(sorted(s.items(), key=lambda x: -x[1]), 1):
            print(f"  {i:2d}. {f:14s} {v:9.2f} ms {v / t:6.1%}")

    print(f"\nthe {top} costliest calls (replayed device time x uses), indices for jitnn.kbench.only:")
    rows.sort(key=lambda r: -r["device_us"] * sum(r["uses"].values()))
    print(",".join(str(r["index"]) for r in rows[:top] if "index" in r) or "(no indices in these results: rerun with the index field)")
    for r in rows[:top]:
        shapes = " ".join("x".join(map(str, i["shape"])) for i in r["in"])
        print(f"  {r.get('index', '?'):>4} {r['op']:20s} {r['device_us'] * sum(r['uses'].values()) / 1e3:8.2f} ms  {shapes}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 25)
