"""Max error of jitNN's saved inference outputs (<name>.jitnn.f32) against gsweep.py's FP32 references
(<shape>.ref.npy), relative to the reference's largest magnitude: python jerr.py <work dir> <jitnn out dir>..."""
import glob
import os
import sys

import numpy as np

work = sys.argv[1]
for d in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(d, "*.jitnn.f32"))):
        n = os.path.basename(f)[:-len(".jitnn.f32")]
        ref = os.path.join(work, n.rsplit("_", 1)[0] + ".ref.npy")
        if not os.path.exists(ref):
            print(f"[jerr] {os.path.basename(d)} {n} no reference")
            continue
        r = np.load(ref).ravel()
        y = np.fromfile(f, dtype="<f4").astype(np.float64)
        if y.size != r.size:
            print(f"[jerr] {os.path.basename(d)} {n} size {y.size} vs {r.size}")
            continue
        print(f"[jerr] {os.path.basename(d):10s} {n:28s} err {np.max(np.abs(y - r)) / max(np.max(np.abs(r)), 1e-30):.1e}")
