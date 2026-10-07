#!/bin/bash
# Step 0: profiler access, ceilings, clocks under load.
export PATH=/usr/local/cuda-13.4/bin:$PATH
cd "$(dirname "$0")"
echo "== driver profiling restriction"
grep -i "RmProfilingAdminOnly" /proc/driver/nvidia/params 2>/dev/null || echo "param not readable"
nvcc -O3 -arch=sm_80 -o ceil ceil.cu 2>&1 | grep -v warning | head -5
echo "== clocks during the run (100 ms samples) -> clocks.csv"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv,noheader -lms 100 > clocks.csv &
SMI=$!
sleep 1
echo "== ceilings (CUDA C)"
for r in 1 2 3; do ./ceil; done
echo "== cuBLAS peaks (PyTorch)"
${PYTHON:-python} gemm_peak.py 2>&1 | grep -v Warn
echo "== sustained FMA load 30 s (clock behaviour)"
end=$((SECONDS+30)); while [ $SECONDS -lt $end ]; do ./ceil > /dev/null; done
kill $SMI
echo "== clock summary"
awk -F', ' '{sm=$2+0; if (min==""||sm<min) min=sm; if (sm>max) max=sm; n++; s+=sm; t=$4+0; if (t>tm) tm=t; p=$5+0; if (p>pm) pm=p} END {printf "samples %d  SM MHz min %d max %d mean %.0f  max temp %d C  max power %.0f W\n", n, min, max, s/n, tm, pm}' clocks.csv
awk -F', ' '$2+0 < 1350 && $2+0 > 500' clocks.csv | head -5
cut -d, -f6 clocks.csv | sort | uniq -c | sort -rn | head -5
echo "== ncu counter access (non-root)"
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed -k copy4 -c 1 ./ceil 2>&1 | grep -E "ERR|PERM|throughput|==PROF==|Error" | head -8
