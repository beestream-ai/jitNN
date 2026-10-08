"""The kernels behind the costliest calls, from an Nsight Systems trace of the kernel benchmark:
per CUDA kernel, launches, median duration, grid and block, registers per thread, shared memory,
and the theoretical occupancy on an A100 (65,536 registers, 2,048 threads, 32 blocks and 164 KiB of
shared memory per SM; registers allocated per warp in 256-register units).

usage: nsys stats --force-export=true --report cuda_gpu_trace --format csv -o <prefix> <trace>.nsys-rep
       python nsys_kernels.py <prefix>_cuda_gpu_trace.csv
"""
import csv
import statistics
import sys
from collections import defaultdict


def occupancy(block, regs, smem):
    warps_per_block = -(-block // 32)
    regs_per_warp = -(-max(regs, 1) * 32 // 256) * 256
    by_regs = 65536 // max(regs_per_warp * warps_per_block, 1)
    by_threads = 2048 // max(block, 1)
    by_smem = (164 * 1024) // smem if smem else 32
    blocks = max(min(by_regs, by_threads, by_smem, 32), 0)
    limit = min((by_regs, "registers"), (by_threads, "threads"), (by_smem, "shared memory"), (32, "blocks"))[1]
    return blocks * warps_per_block / 64, limit


def main(path):
    k = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            name = r.get("Name", "")
            if not r.get("GrdX"):
                continue
            k[name].append(r)
    print(f"{'kernel':52s} {'n':>5s} {'median us':>9s} {'grid':>16s} {'block':>10s} {'regs':>5s} {'smem B':>8s} {'occupancy':>9s} limit")
    rows = []
    for name, rs in k.items():
        d = statistics.median(float(r["Duration (ns)"]) for r in rs) / 1e3
        r = rs[0]
        block = int(r["BlkX"]) * int(r["BlkY"]) * int(r["BlkZ"])
        regs = int(r["Reg/Trd"])
        smem = int(float(r["StcSMem (MB)"]) * 1024 * 1024 + float(r["DymSMem (MB)"]) * 1024 * 1024) if "StcSMem (MB)" in r else int(r.get("StcSMem", 0)) + int(r.get("DymSMem", 0))
        occ, lim = occupancy(block, regs, smem)
        rows.append((d * len(rs), name, len(rs), d, f"{r['GrdX']}x{r['GrdY']}x{r['GrdZ']}", f"{r['BlkX']}x{r['BlkY']}x{r['BlkZ']}", regs, smem, occ, lim))
    for _, name, n, d, g, b, regs, smem, occ, lim in sorted(rows, reverse=True):
        print(f"{name[:52]:52s} {n:5d} {d:9.1f} {g:>16s} {b:>10s} {regs:5d} {smem:8d} {occ:9.0%} {lim}")


if __name__ == "__main__":
    main(sys.argv[1])
