package org.beehive.jitnn.ops;

import java.lang.foreign.MemorySegment;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.cpu.Cpu;
import org.beehive.jitnn.cpu.Elementwise;
import org.beehive.jitnn.cpu.Gemm;
import org.beehive.jitnn.cpu.Gemm.Matrix;
import org.beehive.jitnn.engine.Storage;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The CPU backend's operators on {@link Gemm}: each takes the decoded attributes of its node and
 * returns false when it does not handle the operands (not FP32), so the node runs its reference.
 */
final class CpuKernels {

    private CpuKernels() {
    }

    /** The FP32 contents of {@code t}, or null when it is not stored as FP32. */
    static MemorySegment f32(Tensor t) {
        return t.dtype() == DType.F32 && t.storage().host() instanceof FloatArray a ? a.getSegment() : null;
    }

    /**
     * {@code op} packed as {@code layout}: cached when {@code t} is persistent (a weight), for as long
     * as its version stays the same; else packed into this thread's scratch {@code slot}, or into a
     * new array when {@code slot} is negative.
     */
    private static float[] packed(Tensor t, String layout, int slot, int rows, int k, int w, Matrix op) {
        Storage s = t.storage();
        if (s.isPersistent()) {
            return Gemm.cached(s, layout, s.version(), () -> pack(op, rows, k, w, new float[Gemm.packedSize(rows, k, w)]));
        }
        int size = Gemm.packedSize(rows, k, w);
        return pack(op, rows, k, w, slot < 0 ? new float[size] : Gemm.scratch(slot, size));
    }

    private static float[] pack(Matrix op, int rows, int k, int w, float[] dst) {
        return w == Gemm.MR ? Gemm.packA(op, rows, k, dst) : Gemm.packB(op, rows, k, dst);
    }

    /** y [m, n] = act(x [m, k] W^T + b), W [n, k]; act 0 none, 1 ReLU, 2 GELU. */
    static boolean linear(Tensor x, Tensor w, Tensor b, Tensor y, int m, int n, int k, int act) {
        MemorySegment xs = f32(x);
        MemorySegment ws = f32(w);
        MemorySegment ys = f32(y);
        MemorySegment bs = b == null ? null : f32(b);
        if (xs == null || ws == null || ys == null || b != null && bs == null || act > 2) {
            return false;
        }
        float[] ap = packed(x, "linearA" + m + "x" + k, 0, m, k, Gemm.MR, new Matrix(xs, 0, k, 1));
        float[] bp = packed(w, "linearB" + n + "x" + k, 1, n, k, Gemm.NR, new Matrix(ws, 0, k, 1));
        int epilogue = (b != null ? Gemm.BIAS_COL : 0) | (act == 1 ? Gemm.RELU : 0) | (act == 2 ? Gemm.GELU : 0);
        Gemm.compute(ap, bp, m, n, k, ys, 0, n, bs, 0, epilogue);
        return true;
    }

    /**
     * c [batch, m, n] = op(a) op(b) for row-major a ([m, k], or [k, m] transposed) and b ([k, n], or
     * [n, k] transposed); a stride of 0 shares that operand across the batch.
     */
    static boolean matmul(Tensor a, Tensor b, Tensor c, int m, int n, int k, boolean ta, boolean tb, int batch, int strideA, int strideB) {
        MemorySegment as = f32(a);
        MemorySegment bs = f32(b);
        MemorySegment cs = f32(c);
        if (as == null || bs == null || cs == null) {
            return false;
        }
        long axs = ta ? 1 : k;
        long aqs = ta ? m : 1;
        long bxs = tb ? k : 1;
        long bqs = tb ? 1 : n;
        if (batch == 1 || strideA == 0 && strideB == 0) {
            float[] ap = packed(a, "matmulA" + m + "x" + k + ta, 0, m, k, Gemm.MR, new Matrix(as, 0, axs, aqs));
            float[] bp = packed(b, "matmulB" + n + "x" + k + tb, 1, n, k, Gemm.NR, new Matrix(bs, 0, bxs, bqs));
            for (int z = 0; z < batch; z++) {
                Gemm.compute(ap, bp, m, n, k, cs, (long) z * m * n, n, null, 0, 0);
            }
            return true;
        }
        Cpu.Range one = (from, to) -> {
            for (int z = from; z < to; z++) {
                Gemm.multiply(new Matrix(as, (long) z * strideA, axs, aqs), new Matrix(bs, (long) z * strideB, bxs, bqs), m, n, k, cs, (long) z * m * n, n, null, 0, 0);
            }
        };
        if (batch >= Cpu.threads() || 2L * m * n * k < 1 << 22) {
            // many small products: one per thread at a time (each runs single-threaded inside)
            Cpu.parallelFor(batch, 1, one);
        } else {
            one.run(0, batch);
        }
        return true;
    }

    /**
     * out = act(conv2d(x, w) + bias) in NCHW: per image and group, the weight rows (packed once) times
     * the input's im2col columns, packed straight from the image (a 1 x 1, stride 1 convolution reads
     * the image as the matrix itself).
     */
    static boolean conv2d(Tensor x, Tensor w, Tensor bias, Tensor out, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups,
            boolean relu) {
        MemorySegment xs = f32(x);
        MemorySegment wsg = f32(w);
        MemorySegment os = f32(out);
        MemorySegment bs = bias == null ? null : f32(bias);
        if (xs == null || wsg == null || os == null || bias != null && bs == null) {
            return false;
        }
        int cg = cin / groups;
        int cog = cout / groups;
        int k = cg * r * r;
        int pixels = oh * ow;
        boolean pointwise = r == 1 && stride == 1 && pad == 0;
        int epilogue = (bs != null ? Gemm.BIAS_ROW : 0) | (relu ? Gemm.RELU : 0);
        float[][] weights = new float[groups][];
        for (int g = 0; g < groups; g++) {
            weights[g] = packed(w, "convA" + g + "/" + groups + ":" + cog + "x" + k, groups == 1 ? 0 : -1, cog, k, Gemm.MR, new Matrix(wsg, (long) g * cog * k, k, 1));
        }
        int images = n * groups;
        int panelsM = (cog + Gemm.MR - 1) / Gemm.MR;
        if (images == 1) {
            // one image: packing and the GEMM each run on every thread
            float[] col = Gemm.scratch(1, Gemm.packedSize(pixels, k, Gemm.NR));
            if (pointwise) {
                Gemm.packB(new Matrix(xs, 0, 1, (long) h * wd), pixels, k, col);
            } else {
                Gemm.packIm2col(xs, 0, cg, h, wd, r, stride, pad, dil, oh, ow, col);
            }
            Gemm.compute(weights[0], col, cog, pixels, k, os, 0, pixels, bs, 0, epilogue);
            return true;
        }
        // several images (or groups): one per work item, the output channels split in two when there
        // are fewer images than threads; each item packs its own columns (a split packs them again,
        // so no more than two) and runs its GEMM on its thread
        int split = Math.max(1, Math.min(Math.min(panelsM, 2), (Cpu.threads() + images - 1) / images));
        Cpu.parallelFor(images * split, 1, (from, to) -> {
            for (int item = from; item < to; item++) {
                int img = item / split / groups;
                int g = item / split % groups;
                int part = item % split;
                int p0 = part * panelsM / split;
                int p1 = (part + 1) * panelsM / split;
                if (p0 == p1) {
                    continue;
                }
                float[] col = Gemm.scratch(1, Gemm.packedSize(pixels, k, Gemm.NR));
                long xOff = ((long) img * cin + (long) g * cg) * h * wd;
                if (pointwise) {
                    Gemm.packB(new Matrix(xs, xOff, 1, (long) h * wd), pixels, k, col);
                } else {
                    Gemm.packIm2col(xs, xOff, cg, h, wd, r, stride, pad, dil, oh, ow, col);
                }
                int row0 = p0 * Gemm.MR;
                int rows = Math.min(cog, p1 * Gemm.MR) - row0;
                long cOff = ((long) img * cout + (long) g * cog + row0) * pixels;
                Gemm.compute(weights[g], p0 * k * Gemm.MR, col, rows, pixels, k, os, cOff, pixels, bs, (long) g * cog + row0, epilogue);
            }
        });
        return true;
    }

    private static final java.lang.foreign.ValueLayout.OfFloat F = java.lang.foreign.ValueLayout.JAVA_FLOAT;
    private static final java.lang.foreign.ValueLayout.OfInt I = java.lang.foreign.ValueLayout.JAVA_INT;

    /** out = op(a, b) with broadcasting, when both operands broadcast in the repeating form (see {@link Binary#repeat}). */
    static boolean binary(int code, Tensor a, Tensor b, Tensor out, float alpha, float beta) {
        MemorySegment as = f32(a);
        MemorySegment bs = f32(b);
        MemorySegment os = f32(out);
        int[] ra = Binary.repeat(a.shape(), out.shape());
        int[] rb = Binary.repeat(b.shape(), out.shape());
        if (as == null || bs == null || os == null || ra == null || rb == null) {
            return false;
        }
        Elementwise.binary(code, as, ra[0], ra[1], bs, rb[0], rb[1], os, out.numel(), alpha, beta);
        return true;
    }

    /** out = op(x) element by element. */
    static boolean unary(int code, Tensor x, Tensor out, float alpha, float beta) {
        MemorySegment xs = f32(x);
        MemorySegment os = f32(out);
        if (xs == null || os == null || x.numel() != out.numel()) {
            return false;
        }
        Elementwise.unary(code, xs, os, out.numel(), alpha, beta);
        return true;
    }

    /** y = softmax (or log-softmax) of each row of x [rows, cols]. */
    static boolean softmax(Tensor x, Tensor y, int rows, int cols, boolean log) {
        MemorySegment xs = f32(x);
        MemorySegment ys = f32(y);
        if (xs == null || ys == null) {
            return false;
        }
        Cpu.parallelFor(rows, Math.max(1, Elementwise_GRAIN / Math.max(1, cols)), (from, to) -> {
            for (int r = from; r < to; r++) {
                long off = (long) r * cols;
                MemorySegment.copy(xs, off * 4, ys, off * 4, cols * 4L);
                float lse = Elementwise.softmaxRow(ys, off, cols);
                if (log) {
                    for (int c = 0; c < cols; c++) {
                        ys.setAtIndex(F, off + c, xs.getAtIndex(F, off + c) - lse);
                    }
                }
            }
        });
        return true;
    }

    private static final int Elementwise_GRAIN = 1 << 15;

    /** LayerNorm or RMSNorm of each row (x plus the residual when given), with the statistics saved for the gradient. */
    static boolean rowNorm(Tensor x, Tensor residual, Tensor w, Tensor b, Tensor y, Tensor mean, Tensor rstd, int rows, int cols, boolean layer, float eps) {
        MemorySegment xs = f32(x);
        MemorySegment rs = residual == null ? null : f32(residual);
        MemorySegment ws = f32(w);
        MemorySegment bs = b == null ? null : f32(b);
        MemorySegment ys = f32(y);
        MemorySegment ms = f32(mean);
        MemorySegment ss = f32(rstd);
        if (xs == null || ws == null || ys == null || ms == null || ss == null || residual != null && rs == null || b != null && bs == null) {
            return false;
        }
        Cpu.parallelFor(rows, Math.max(1, Elementwise_GRAIN / Math.max(1, cols)), (from, to) -> {
            for (int r = from; r < to; r++) {
                long off = (long) r * cols;
                MemorySegment src = xs;
                if (rs != null) {
                    // x + residual, staged in y, which the normalisation then overwrites
                    Elementwise.binary(org.beehive.jitnn.kernels.ElementwiseKernels.ADD, xs.asSlice(off * 4, cols * 4L), 1, cols, rs.asSlice(off * 4, cols * 4L), 1, cols,
                            ys.asSlice(off * 4, cols * 4L), cols, 0, 0);
                    src = ys;
                }
                float mu = layer ? Elementwise.sum(src, off, cols) / cols : 0;
                float is = (float) (1 / Math.sqrt(Elementwise.sumSquares(src, off, cols, mu) / cols + eps));
                Elementwise.normalise(src, off, mu, is, ws, bs, ys, off, cols);
                ms.setAtIndex(F, r, mu);
                ss.setAtIndex(F, r, is);
            }
        });
        return true;
    }

    /** Max (with the arg-max index) or average pooling of [planes, h, w] into [planes, oh, ow]. */
    static boolean pool2d(Tensor x, Tensor out, Tensor arg, int planes, int h, int w, int oh, int ow, int k, int stride, int pad, boolean max) {
        MemorySegment xs = f32(x);
        MemorySegment os = f32(out);
        MemorySegment as = arg == null ? null : arg.storage().host() instanceof IntArray ia ? ia.getSegment() : null;
        if (xs == null || os == null || max && as == null) {
            return false;
        }
        float inv = 1.0f / (k * k);
        Cpu.parallelFor(planes, Math.max(1, Elementwise_GRAIN / Math.max(1, oh * ow * k * k)), (from, to) -> {
            for (int p = from; p < to; p++) {
                long base = (long) p * h * w;
                for (int i = 0; i < oh; i++) {
                    for (int j = 0; j < ow; j++) {
                        float best = Float.NEGATIVE_INFINITY;
                        long bestAt = base;
                        float sum = 0;
                        for (int kh = 0; kh < k; kh++) {
                            int ih = i * stride - pad + kh;
                            if (ih < 0 || ih >= h) {
                                continue;
                            }
                            for (int kw = 0; kw < k; kw++) {
                                int iw = j * stride - pad + kw;
                                if (iw < 0 || iw >= w) {
                                    continue;
                                }
                                long at = base + (long) ih * w + iw;
                                float v = xs.getAtIndex(F, at);
                                sum += v;
                                if (v > best) {
                                    best = v;
                                    bestAt = at;
                                }
                            }
                        }
                        long o = ((long) p * oh + i) * ow + j;
                        if (max) {
                            os.setAtIndex(F, o, best);
                            as.setAtIndex(I, o, (int) bestAt);
                        } else {
                            os.setAtIndex(F, o, sum * inv);
                        }
                    }
                }
            }
        });
        return true;
    }

    /** Sum (times scale), max, min or product over the middle axis of x [outer, len, inner]. */
    static boolean reduce(Tensor x, Tensor out, int op, int outer, int len, int inner, float scale) {
        MemorySegment xs = f32(x);
        MemorySegment os = f32(out);
        if (xs == null || os == null) {
            return false;
        }
        boolean sum = op == org.beehive.jitnn.kernels.ReduceKernels.SUM;
        if (inner == 1 && sum) {
            Cpu.parallelFor(outer, Math.max(1, Elementwise_GRAIN / Math.max(1, len)), (from, to) -> {
                for (int o = from; o < to; o++) {
                    long at = (long) o * len;
                    float total;
                    if (len < 64) {
                        // short rows (a global average pool's 7 x 7): a plain loop; the vector sum's
                        // few lanes cost more than they save, and the loop is at full speed sooner
                        total = 0;
                        for (int q = 0; q < len; q++) {
                            total += xs.getAtIndex(F, at + q);
                        }
                    } else {
                        total = Elementwise.sum(xs, at, len);
                    }
                    os.setAtIndex(F, o, total * scale);
                }
            });
            return true;
        }
        int columns = outer * inner;
        Cpu.parallelFor(columns, Math.max(1, Elementwise_GRAIN / Math.max(1, len)), (from, to) -> {
            for (int c = from; c < to; c++) {
                int o = c / inner;
                int j = c - o * inner;
                long at = (long) o * len * inner + j;
                float acc = op == org.beehive.jitnn.kernels.ReduceKernels.MAX ? Float.NEGATIVE_INFINITY : op == org.beehive.jitnn.kernels.ReduceKernels.MIN ? Float.POSITIVE_INFINITY
                        : op == org.beehive.jitnn.kernels.ReduceKernels.PROD ? 1 : 0;
                for (int q = 0; q < len; q++) {
                    float v = xs.getAtIndex(F, at + (long) q * inner);
                    acc = op == org.beehive.jitnn.kernels.ReduceKernels.MAX ? Math.max(acc, v) : op == org.beehive.jitnn.kernels.ReduceKernels.MIN ? Math.min(acc, v)
                            : op == org.beehive.jitnn.kernels.ReduceKernels.PROD ? acc * v : acc + v;
                }
                os.setAtIndex(F, c, sum ? acc * scale : acc);
            }
        });
        return true;
    }

    /** out [count, dim] = rows of weight [vocab, dim] picked by index. */
    static boolean embedding(Tensor index, Tensor weight, Tensor out, int count, int vocab, int dim) {
        MemorySegment ws = f32(weight);
        MemorySegment os = f32(out);
        MemorySegment is = index.storage().host() instanceof IntArray ia ? ia.getSegment() : null;
        if (ws == null || os == null || is == null) {
            return false;
        }
        Cpu.parallelFor(count, Math.max(1, Elementwise_GRAIN / Math.max(1, dim)), (from, to) -> {
            for (int t = from; t < to; t++) {
                int row = is.getAtIndex(I, t);
                if (row < 0 || row >= vocab) {
                    throw new IndexOutOfBoundsException("embedding index " + row + " for " + vocab + " rows");
                }
                MemorySegment.copy(ws, (long) row * dim * 4, os, (long) t * dim * 4, dim * 4L);
            }
        });
        return true;
    }

    /** Query rows per attention work item: their scores, [rows, S], stay in this thread's scratch. */
    private static final int ATTENTION_ROWS = 64;
    private static final ThreadLocal<float[]> SCORES = ThreadLocal.withInitial(() -> new float[0]);

    /**
     * o = softmax(q k^T scale + mask) v per problem (batch and head) and block of query rows, as two
     * GEMMs around a vector softmax, with the log-sum-exp of each row; layouts through {@code meta}
     * ({@link FlashAttention#meta}).
     */
    static boolean attention(int[] a, float scale, Tensor q, Tensor k, Tensor v, Tensor mask, Tensor o, Tensor lse) {
        MemorySegment qs = f32(q);
        MemorySegment ks = f32(k);
        MemorySegment vs = f32(v);
        MemorySegment ms = mask == null ? null : f32(mask);
        MemorySegment os = f32(o);
        MemorySegment ls = f32(lse);
        if (qs == null || ks == null || vs == null || os == null || ls == null || mask != null && ms == null) {
            return false;
        }
        int z = a[0], t = a[1], s = a[2], d = a[3], heads = a[5];
        boolean causal = a[4] != 0;
        int[] meta = FlashAttention.meta(a);
        int blocks = (t + ATTENTION_ROWS - 1) / ATTENTION_ROWS;
        Cpu.parallelFor(z * blocks, 1, (from, to) -> {
            for (int item = from; item < to; item++) {
                int zi = item / blocks;
                int r0 = (item - zi * blocks) * ATTENTION_ROWS;
                int rows = Math.min(ATTENTION_ROWS, t - r0);
                float[] sc = SCORES.get();
                if (sc.length < rows * s) {
                    sc = new float[rows * s];
                    SCORES.set(sc);
                }
                MemorySegment scores = MemorySegment.ofArray(sc);
                long qRow = meta[5 + 3];
                long kRow = meta[9 + 3];
                long vRow = meta[13 + 3];
                long oRow = meta[17 + 3];
                long q0 = FlashAttention.at(meta, 0, zi, r0);
                long k0 = FlashAttention.at(meta, 1, zi, 0);
                long v0 = FlashAttention.at(meta, 2, zi, 0);
                long o0 = FlashAttention.at(meta, 3, zi, r0);
                // scores = q k^T
                Gemm.multiply(new Matrix(qs, q0, qRow, 1), new Matrix(ks, k0, kRow, 1), rows, s, d, scores, 0, s, null, 0, 0);
                int b = zi / heads;
                int h = zi - b * heads;
                for (int i = 0; i < rows; i++) {
                    long off = (long) i * s;
                    int qi = r0 + i;
                    Elementwise.scale(scores, off, s, scale);
                    if (ms != null) {
                        long mrow = (long) b * a[6] + (long) h * a[7] + (long) qi * a[8];
                        for (int j = 0; j < s; j++) {
                            scores.setAtIndex(F, off + j, scores.getAtIndex(F, off + j) + ms.getAtIndex(F, mrow + (long) j * a[9]));
                        }
                    }
                    if (causal) {
                        for (int j = qi + 1; j < s; j++) {
                            scores.setAtIndex(F, off + j, Float.NEGATIVE_INFINITY);
                        }
                    }
                    ls.setAtIndex(F, (long) zi * t + qi, Elementwise.softmaxRow(scores, off, s));
                }
                // o = p v
                Gemm.multiply(new Matrix(scores, 0, s, 1), new Matrix(vs, v0, 1, vRow), rows, d, s, os, o0, oRow, null, 0, 0);
            }
        });
        return true;
    }
}
