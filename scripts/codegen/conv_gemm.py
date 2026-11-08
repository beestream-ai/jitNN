# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates jitNN's implicit-GEMM convolution kernels (NCHW, square kernels, symmetric stride,
padding and dilation, groups) on 64x64 register-blocked tiles: 16x16 threads, 4x4 outputs per
thread strided by 16, BK = 16. Prints three Java methods:

  conv2dForward     out[n, co, oh, ow]   GEMM over (n, oh, ow) x co, K = (ci, kh, kw); bias/ReLU epilogue
  conv2dForwardSplitK                    the same over K slice s of {@code chunk}, partials at s * m * cout in
                                         output layout, no epilogue (ConvKernels.sumSplitK adds them)
  conv2dBackwardData dx[n, ci, ih, iw]   GEMM over (n, ih, iw) x ci, K = (co, kh, kw), gathering dy
  conv2dBackwardWeight dw[co, ci, kh, kw] GEMM over co x (ci, kh, kw), K = (n, oh, ow), split-K partials

Each kernel also comes as a "Pd" variant: local-memory rows padded by one float (no bank
conflicts on the tile stores) and the next k tile loaded into registers while the current one is
multiplied.

Groups are folded into the x grid dimension (groupIdx = g * tilesN + column tile); split-K chunks
into the y dimension (groupIdy = s * tilesM + row tile).
"""
BM = BN = 64
BK = 16
T = 256


def body(name, sig, prologue, kloop, a_fast_m, a_load, b_fast_n, b_load, store, pd=False):
    """pd: local-memory rows one float longer than the tile (no bank conflicts when consecutive
    threads store consecutive k), and the next k tile is loaded into registers while the current
    one is multiplied (as scripts/codegen/gemm.py pad dbuf)."""
    SA = BM + (1 if pd else 0)
    SB = BN + (1 if pd else 0)
    QA, QB = BM * BK // T, BN * BK // T
    o = []
    w = o.append
    w(f"    public static void {name}(KernelContext kc, {sig}) {{")
    w(f"        float[] as = kc.allocateFloatLocalArray({BK * SA});")
    w(f"        float[] bs = kc.allocateFloatLocalArray({BK * SB});")
    w("        int tx = kc.localIdx;")
    w("        int ty = kc.localIdy;")
    w("        int tid = ty * 16 + tx;")
    for line in prologue:
        w("        " + line)
    for i in range(4):
        for j in range(4):
            w(f"        float c{i}_{j} = 0.0f;")

    def a_index(ind):
        if a_fast_m:
            w(f"{ind}int kk = idx / {BM};")
            w(f"{ind}int mm = idx - kk * {BM};")
        else:
            w(f"{ind}int mm = idx / {BK};")
            w(f"{ind}int kk = idx - mm * {BK};")

    def b_index(ind):
        if b_fast_n:
            w(f"{ind}int kb = idx / {BN};")
            w(f"{ind}int nn = idx - kb * {BN};")
        else:
            w(f"{ind}int nn = idx / {BK};")
            w(f"{ind}int kb = idx - nn * {BK};")

    def a_elem(ind, idx, k0, dst):
        w(f"{ind}int idx = {idx};")
        a_index(ind)
        w(f"{ind}int gm = row0 + mm;")
        w(f"{ind}int gk = {k0} + kk;")
        w(f"{ind}float va = 0.0f;")
        for line in a_load:
            w(ind + line)
        w(f"{ind}{dst} = va;")

    def b_elem(ind, idx, k0, dst):
        w(f"{ind}int idx = {idx};")
        b_index(ind)
        w(f"{ind}int gn = col0 + nn;")
        w(f"{ind}int gkb = {k0} + kb;")
        w(f"{ind}float vb = 0.0f;")
        for line in b_load:
            w(ind + line)
        w(f"{ind}{dst} = vb;")

    if pd:
        for q in range(QA):
            w(f"        float ra{q} = 0.0f;")
        for q in range(QB):
            w(f"        float rb{q} = 0.0f;")
        w(f"        int kFirst = {kloop[0]};")
        w(f"        int tiles = ({kloop[1]} - kFirst + {BK - 1}) / {BK};")

        def prefetch(ind, k0):
            for q in range(QA):
                w(f"{ind}{{")
                a_elem(ind + "    ", f"tid + {T * q}", k0, f"ra{q}")
                w(f"{ind}}}")
            for q in range(QB):
                w(f"{ind}{{")
                b_elem(ind + "    ", f"tid + {T * q}", k0, f"rb{q}")
                w(f"{ind}}}")
        w("        if (tiles > 0) {")
        prefetch("            ", "kFirst")
        w("        }")
        w("        for (int tile = 0; tile < tiles; tile++) {")
        for q in range(QA):
            w("            {")
            w(f"                int idx = tid + {T * q};")
            a_index("                ")
            w(f"                as[kk * {SA} + mm] = ra{q};")
            w("            }")
        for q in range(QB):
            w("            {")
            w(f"                int idx = tid + {T * q};")
            b_index("                ")
            w(f"                bs[kb * {SB} + nn] = rb{q};")
            w("            }")
        w("            kc.localBarrier();")
        w("            if (tile + 1 < tiles) {")
        w(f"                int k0 = kFirst + (tile + 1) * {BK};")
        prefetch("                ", "k0")
        w("            }")
    else:
        w(f"        for (int k0 = {kloop[0]}; k0 < {kloop[1]}; k0 += {BK}) {{")
        w(f"            for (int q = 0; q < {QA}; q++) {{")
        a_elem("                ", f"tid + {T} * q", "k0", f"as[kk * {SA} + mm]")
        w("            }")
        w(f"            for (int q = 0; q < {QB}; q++) {{")
        b_elem("                ", f"tid + {T} * q", "k0", f"bs[kb * {SB} + nn]")
        w("            }")
        w("            kc.localBarrier();")
    w(f"            for (int p = 0; p < {BK}; p++) {{")
    for i in range(4):
        w(f"                float a{i} = as[p * {SA} + ty + {16 * i}];")
    for j in range(4):
        w(f"                float b{j} = bs[p * {SB} + tx + {16 * j}];")
    for i in range(4):
        for j in range(4):
            w(f"                c{i}_{j} += a{i} * b{j};")
    w("            }")
    w("            kc.localBarrier();")
    w("        }")
    for i in range(4):
        for j in range(4):
            w(f"        {store}({store_args}, row0 + ty + {16 * i}, col0 + tx + {16 * j}, c{i}_{j});")
    w("    }")
    return "\n".join(o)


def both(name, *spec):
    """The kernel as before, and its padded, prefetching variant named name + "Pd"."""
    return body(name, *spec), body(name + "Pd", *spec, pd=True)


# ---------------------------------------------------------------- forward
store_args = "out, bias, cin, cout, oh, ow, groups, m, g, epilogue"
fwd, fwd_pd = both(
    "conv2dForward",
    "FloatArray x, FloatArray w, FloatArray out, FloatArray bias, int cin, int h, int wd, int cout, int oh, int ow, int ks, int stride, int pad, int dil, int groups, int m, int epilogue, int tilesN",
    [
        "int g = kc.groupIdx / tilesN;",
        "int col0 = (kc.groupIdx - g * tilesN) * 64;",
        "int row0 = kc.groupIdy * 64;",
        "int cg = cin / groups;",
        "int cog = cout / groups;",
        "int kg = cg * ks * ks;",
    ],
    ("0", "kg"),
    True,
    [
        "if ((gm < m) & (gk < kg)) {",
        "    int owi = gm % ow;",
        "    int t = gm / ow;",
        "    int ohi = t % oh;",
        "    int n = t / oh;",
        "    int kw = gk % ks;",
        "    int t2 = gk / ks;",
        "    int kh = t2 % ks;",
        "    int ci = t2 / ks;",
        "    int ih = ohi * stride - pad + kh * dil;",
        "    int iw = owi * stride - pad + kw * dil;",
        "    if ((ih >= 0) & (ih < h) & (iw >= 0) & (iw < wd)) {",
        "        va = x.get(((n * cin + g * cg + ci) * h + ih) * wd + iw);",
        "    }",
        "}",
    ],
    False,
    [
        "if ((gkb < kg) & (gn < cog)) {",
        "    vb = w.get((g * cog + gn) * kg + gkb);",
        "}",
    ],
    "storeForward",
)

# ---------------------------------------------------------------- forward, split-K
store_args = "part, cout, oh, ow, groups, m, g, s"
fwd_split, fwd_split_pd = both(
    "conv2dForwardSplitK",
    "FloatArray x, FloatArray w, FloatArray part, int cin, int h, int wd, int cout, int oh, int ow, int ks, int stride, int pad, int dil, int groups, int m, int tilesN, int tilesM, int chunk",
    [
        "int g = kc.groupIdx / tilesN;",
        "int col0 = (kc.groupIdx - g * tilesN) * 64;",
        "int s = kc.groupIdy / tilesM;",
        "int row0 = (kc.groupIdy - s * tilesM) * 64;",
        "int cg = cin / groups;",
        "int cog = cout / groups;",
        "int kg = cg * ks * ks;",
        "int kBegin = s * chunk;",
        "int kEnd = kBegin + chunk < kg ? kBegin + chunk : kg;",
    ],
    ("kBegin", "kEnd"),
    True,
    [
        "if ((gm < m) & (gk < kEnd)) {",
        "    int owi = gm % ow;",
        "    int t = gm / ow;",
        "    int ohi = t % oh;",
        "    int n = t / oh;",
        "    int kw = gk % ks;",
        "    int t2 = gk / ks;",
        "    int kh = t2 % ks;",
        "    int ci = t2 / ks;",
        "    int ih = ohi * stride - pad + kh * dil;",
        "    int iw = owi * stride - pad + kw * dil;",
        "    if ((ih >= 0) & (ih < h) & (iw >= 0) & (iw < wd)) {",
        "        va = x.get(((n * cin + g * cg + ci) * h + ih) * wd + iw);",
        "    }",
        "}",
    ],
    False,
    [
        "if ((gkb < kEnd) & (gn < cog)) {",
        "    vb = w.get((g * cog + gn) * kg + gkb);",
        "}",
    ],
    "storeForwardPartial",
)

# ---------------------------------------------------------------- backward data
store_args = "dx, cin, h, wd, groups, m, g"
bwd_data, bwd_data_pd = both(
    "conv2dBackwardData",
    "FloatArray dy, FloatArray w, FloatArray dx, int cin, int h, int wd, int cout, int oh, int ow, int ks, int stride, int pad, int dil, int groups, int m, int tilesN",
    [
        "int g = kc.groupIdx / tilesN;",
        "int col0 = (kc.groupIdx - g * tilesN) * 64;",
        "int row0 = kc.groupIdy * 64;",
        "int cg = cin / groups;",
        "int cog = cout / groups;",
        "int kk2 = ks * ks;",
        "int kred = cog * kk2;",
    ],
    ("0", "kred"),
    True,
    [
        "if ((gm < m) & (gk < kred)) {",
        "    int iwi = gm % wd;",
        "    int t = gm / wd;",
        "    int ihi = t % h;",
        "    int n = t / h;",
        "    int kw = gk % ks;",
        "    int t2 = gk / ks;",
        "    int kh = t2 % ks;",
        "    int co = t2 / ks;",
        "    int ohn = ihi + pad - kh * dil;",
        "    int own = iwi + pad - kw * dil;",
        "    int ohq = ohn / stride;",
        "    int owq = own / stride;",
        "    if ((ohn >= 0) & (own >= 0) & (ohq * stride == ohn) & (owq * stride == own) & (ohq < oh) & (owq < ow)) {",
        "        va = dy.get(((n * cout + g * cog + co) * oh + ohq) * ow + owq);",
        "    }",
        "}",
    ],
    False,
    [
        "if ((gkb < kred) & (gn < cg)) {",
        "    int kw = gkb % ks;",
        "    int t2 = gkb / ks;",
        "    int kh = t2 % ks;",
        "    int co = t2 / ks;",
        "    vb = w.get(((g * cog + co) * cg + gn) * kk2 + kh * ks + kw);",
        "}",
    ],
    "storeBackwardData",
)

# ---------------------------------------------------------------- backward weight (split-K)
store_args = "partial, cout, groups, g, s, kgOut, cogOut"
bwd_weight, bwd_weight_pd = both(
    "conv2dBackwardWeight",
    "FloatArray dy, FloatArray x, FloatArray partial, int cin, int h, int wd, int cout, int oh, int ow, int ks, int stride, int pad, int dil, int groups, int positions, int tilesN, int tilesM, int chunk",
    [
        "int g = kc.groupIdx / tilesN;",
        "int col0 = (kc.groupIdx - g * tilesN) * 64;",
        "int s = kc.groupIdy / tilesM;",
        "int row0 = (kc.groupIdy - s * tilesM) * 64;",
        "int cg = cin / groups;",
        "int cog = cout / groups;",
        "int kgOut = cg * ks * ks;",
        "int cogOut = cog;",
        "int kStart = s * chunk;",
        "int kEnd = kStart + chunk < positions ? kStart + chunk : positions;",
    ],
    ("kStart", "kEnd"),
    False,
    [
        "if ((gm < cog) & (gk < kEnd)) {",
        "    int owi = gk % ow;",
        "    int t = gk / ow;",
        "    int ohi = t % oh;",
        "    int n = t / oh;",
        "    va = dy.get(((n * cout + g * cog + gm) * oh + ohi) * ow + owi);",
        "}",
    ],
    False,
    [
        "if ((gkb < kEnd) & (gn < kgOut)) {",
        "    int owi = gkb % ow;",
        "    int t = gkb / ow;",
        "    int ohi = t % oh;",
        "    int n = t / oh;",
        "    int kw = gn % ks;",
        "    int t2 = gn / ks;",
        "    int kh = t2 % ks;",
        "    int ci = t2 / ks;",
        "    int ih = ohi * stride - pad + kh * dil;",
        "    int iw = owi * stride - pad + kw * dil;",
        "    if ((ih >= 0) & (ih < h) & (iw >= 0) & (iw < wd)) {",
        "        vb = x.get(((n * cin + g * cg + ci) * h + ih) * wd + iw);",
        "    }",
        "}",
    ],
    "storeBackwardWeight",
)

# ---------------------------------------------------------------- backward data, strided (one launch, s x s phases)
store_args = "dx, cin, h, wd, groups, mp, g, ph, pw, hp, wp, stride"
bwd_data_phase = body(
    "conv2dBackwardDataPhase",
    "FloatArray dy, FloatArray w, FloatArray dx, int cin, int h, int wd, int cout, int oh, int ow, int ks, int stride, int pad, int groups, int n, int tilesN, int tilesM",
    [
        "int g = kc.groupIdx / tilesN;",
        "int col0 = (kc.groupIdx - g * tilesN) * 64;",
        "int phase = kc.groupIdy / tilesM;",
        "int row0 = (kc.groupIdy - phase * tilesM) * 64;",
        "int ph = phase / stride;",
        "int pw = phase - ph * stride;",
        "int hp = (h - ph + stride - 1) / stride;",
        "int wp = (wd - pw + stride - 1) / stride;",
        "int mp = n * hp * wp;",
        "int kh0 = (ph + pad) % stride;",
        "int kw0 = (pw + pad) % stride;",
        "int nkh = (ks - kh0 + stride - 1) / stride;",
        "int nkw = (ks - kw0 + stride - 1) / stride;",
        "int taps = nkh * nkw;",
        "int cg = cin / groups;",
        "int cog = cout / groups;",
        "int kk2 = ks * ks;",
        "int kred = cog * taps;",
    ],
    ("0", "kred"),
    True,
    [
        "if ((gm < mp) & (gk < kred)) {",
        "    int jwi = gm % wp;",
        "    int t = gm / wp;",
        "    int jhi = t % hp;",
        "    int ni = t / hp;",
        "    int co = gk / taps;",
        "    int r = gk - co * taps;",
        "    int jh = r / nkw;",
        "    int jw = r - jh * nkw;",
        "    int ohq = (ph + jhi * stride + pad - kh0 - jh * stride) / stride;",
        "    int owq = (pw + jwi * stride + pad - kw0 - jw * stride) / stride;",
        "    if ((ohq >= 0) & (ohq < oh) & (owq >= 0) & (owq < ow)) {",
        "        va = dy.get(((ni * cout + g * cog + co) * oh + ohq) * ow + owq);",
        "    }",
        "}",
    ],
    False,
    [
        "if ((gkb < kred) & (gn < cg)) {",
        "    int co = gkb / taps;",
        "    int r = gkb - co * taps;",
        "    int jh = r / nkw;",
        "    int jw = r - jh * nkw;",
        "    vb = w.get(((g * cog + co) * cg + gn) * kk2 + (kh0 + jh * stride) * ks + kw0 + jw * stride);",
        "}",
    ],
    "storeBackwardDataPhase",
    pd=True,
)

phase_helper = '''
    private static void storeBackwardDataPhase(FloatArray dx, int cin, int h, int wd, int groups, int mp, int g, int ph, int pw, int hp, int wp, int stride, int gm, int gn, float v) {
        int cg = cin / groups;
        if ((gm < mp) & (gn < cg)) {
            int jwi = gm % wp;
            int t = gm / wp;
            int jhi = t % hp;
            int n = t / hp;
            dx.set(((n * cin + g * cg + gn) * h + ph + jhi * stride) * wd + pw + jwi * stride, v);
        }
    }
'''

helpers = '''
    private static void storeForward(FloatArray out, FloatArray bias, int cin, int cout, int oh, int ow, int groups, int m, int g, int epilogue, int gm, int gn, float v) {
        int cog = cout / groups;
        if ((gm < m) & (gn < cog)) {
            int owi = gm % ow;
            int t = gm / ow;
            int ohi = t % oh;
            int n = t / oh;
            int co = g * cog + gn;
            float r = v;
            if ((epilogue & 1) != 0) {
                r = r + bias.get(co);
            }
            if ((epilogue & 2) != 0) {
                r = r > 0.0f ? r : 0.0f;
            }
            out.set(((n * cout + co) * oh + ohi) * ow + owi, r);
        }
    }

    private static void storeForwardPartial(FloatArray part, int cout, int oh, int ow, int groups, int m, int g, int s, int gm, int gn, float v) {
        int cog = cout / groups;
        if ((gm < m) & (gn < cog)) {
            int owi = gm % ow;
            int t = gm / ow;
            int ohi = t % oh;
            int n = t / oh;
            part.set(s * m * cout + ((n * cout + g * cog + gn) * oh + ohi) * ow + owi, v);
        }
    }

    /**
     * Sums the {@code slices} partial outputs of {@link #conv2dForwardSplitK} in slice order
     * (deterministic) and applies the bias/ReLU epilogue. {@code total} = N * cout * OH * OW,
     * {@code plane} = OH * OW.
     */
    public static void sumSplitK(KernelContext kc, FloatArray part, FloatArray out, FloatArray bias, int slices, int total, int cout, int plane, int epilogue) {
        int i = kc.globalIdx;
        if (i < total) {
            float r = 0.0f;
            for (int z = 0; z < slices; z++) {
                r += part.get(z * total + i);
            }
            if ((epilogue & 1) != 0) {
                r = r + bias.get((i / plane) % cout);
            }
            if ((epilogue & 2) != 0) {
                r = r > 0.0f ? r : 0.0f;
            }
            out.set(i, r);
        }
    }

    private static void storeBackwardData(FloatArray dx, int cin, int h, int wd, int groups, int m, int g, int gm, int gn, float v) {
        int cg = cin / groups;
        if ((gm < m) & (gn < cg)) {
            int iwi = gm % wd;
            int t = gm / wd;
            int ihi = t % h;
            int n = t / h;
            dx.set(((n * cin + g * cg + gn) * h + ihi) * wd + iwi, v);
        }
    }

    private static void storeBackwardWeight(FloatArray partial, int cout, int groups, int g, int s, int kgOut, int cogOut, int gm, int gn, float v) {
        if ((gm < cogOut) & (gn < kgOut)) {
            partial.set(s * cout * kgOut + (g * cogOut + gm) * kgOut + gn, v);
        }
    }
'''

print("    // Generated by scripts/codegen/conv_gemm.py; edit the generator, not this code.")
print(fwd)
print()
print(fwd_split)
print()
print(bwd_data)
print()
print(bwd_weight)
print(helpers)
print("    // Padded local tiles and register prefetch of the next k tile; generated by scripts/codegen/conv_gemm.py.")
print(fwd_pd)
print()
print(fwd_split_pd)
print()
print(bwd_data_pd)
print()
print(bwd_weight_pd)
print()
print("    /**")
print("     * Gradient of the input of a strided convolution (dilation 1), generated by scripts/codegen/conv_gemm.py: the")
print("     * input positions split into stride x stride phases (ih = ph + stride i, iw = pw + stride j); in phase (ph, pw)")
print("     * only the taps kh = (ph + pad) mod stride + stride jh, kw likewise, reach the input, so each phase is a dense")
print("     * implicit GEMM over (n, i, j) x ci with K = (co, jh, jw), instead of gathering every tap and masking the")
print("     * misaligned ones. Phases with no taps write zeros. Launch tilesN * groups x stride^2 * tilesM groups of 16x16")
print("     * threads, tilesM for phase (0, 0), the largest.")
print("     */")
print(bwd_data_phase)
print(phase_helper)
