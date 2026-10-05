"""Generates jitNN's implicit-GEMM convolution kernels (NCHW, square kernels, symmetric stride,
padding and dilation, groups) on 64x64 register-blocked tiles: 16x16 threads, 4x4 outputs per
thread strided by 16, BK = 16. Prints three Java methods:

  conv2dForward     out[n, co, oh, ow]   GEMM over (n, oh, ow) x co, K = (ci, kh, kw); bias/ReLU epilogue
  conv2dForwardSplitK                    the same over K slice s of {@code chunk}, partials at s * m * cout in
                                         output layout, no epilogue (ConvKernels.sumSplitK adds them)
  conv2dBackwardData dx[n, ci, ih, iw]   GEMM over (n, ih, iw) x ci, K = (co, kh, kw), gathering dy
  conv2dBackwardWeight dw[co, ci, kh, kw] GEMM over co x (ci, kh, kw), K = (n, oh, ow), split-K partials

Groups are folded into the x grid dimension (groupIdx = g * tilesN + column tile); split-K chunks
into the y dimension (groupIdy = s * tilesM + row tile).
"""
BM = BN = 64
BK = 16
T = 256


def body(name, sig, prologue, kloop, a_fast_m, a_load, b_fast_n, b_load, store):
    o = []
    w = o.append
    w(f"    public static void {name}(KernelContext kc, {sig}) {{")
    w(f"        float[] as = kc.allocateFloatLocalArray({BK * BM});")
    w(f"        float[] bs = kc.allocateFloatLocalArray({BK * BN});")
    w("        int tx = kc.localIdx;")
    w("        int ty = kc.localIdy;")
    w("        int tid = ty * 16 + tx;")
    for line in prologue:
        w("        " + line)
    for i in range(4):
        for j in range(4):
            w(f"        float c{i}_{j} = 0.0f;")
    w(f"        for (int k0 = {kloop[0]}; k0 < {kloop[1]}; k0 += {BK}) {{")
    w(f"            for (int q = 0; q < {BM * BK // T}; q++) {{")
    w(f"                int idx = tid + {T} * q;")
    if a_fast_m:
        w(f"                int kk = idx / {BM};")
        w(f"                int mm = idx - kk * {BM};")
    else:
        w(f"                int mm = idx / {BK};")
        w(f"                int kk = idx - mm * {BK};")
    w("                int gm = row0 + mm;")
    w("                int gk = k0 + kk;")
    w("                float va = 0.0f;")
    for line in a_load:
        w("                " + line)
    w(f"                as[kk * {BM} + mm] = va;")
    w("            }")
    w(f"            for (int q = 0; q < {BN * BK // T}; q++) {{")
    w(f"                int idx = tid + {T} * q;")
    if b_fast_n:
        w(f"                int kb = idx / {BN};")
        w(f"                int nn = idx - kb * {BN};")
    else:
        w(f"                int nn = idx / {BK};")
        w(f"                int kb = idx - nn * {BK};")
    w("                int gn = col0 + nn;")
    w("                int gkb = k0 + kb;")
    w("                float vb = 0.0f;")
    for line in b_load:
        w("                " + line)
    w(f"                bs[kb * {BN} + nn] = vb;")
    w("            }")
    w("            kc.localBarrier();")
    w(f"            for (int p = 0; p < {BK}; p++) {{")
    for i in range(4):
        w(f"                float a{i} = as[p * {BM} + ty + {16 * i}];")
    for j in range(4):
        w(f"                float b{j} = bs[p * {BN} + tx + {16 * j}];")
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


# ---------------------------------------------------------------- forward
store_args = "out, bias, cin, cout, oh, ow, groups, m, g, epilogue"
fwd = body(
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
fwd_split = body(
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
bwd_data = body(
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
bwd_weight = body(
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
