package ai.beestream.jitnn.ops;

import java.lang.foreign.MemorySegment;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.cpu.Blas;
import ai.beestream.jitnn.cpu.Bnns;
import ai.beestream.jitnn.cpu.Cpu;
import ai.beestream.jitnn.cpu.Dnnl;
import ai.beestream.jitnn.cpu.Elementwise;
import ai.beestream.jitnn.cpu.Gemm;
import ai.beestream.jitnn.cpu.Gemm.Matrix;
import ai.beestream.jitnn.cpu.Race;
import ai.beestream.jitnn.engine.Storage;

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
        int epilogue = (b != null ? Gemm.BIAS_COL : 0) | (act == 1 ? Gemm.RELU : 0) | (act == 2 ? Gemm.GELU : 0);
        Blas.multiply("linear " + m + "x" + n + "x" + k, new Matrix(xs, 0, k, 1), new Matrix(ws, 0, k, 1), m, n, k, ys, 0, n, bs, 0, epilogue, () -> {
            float[] ap = packed(x, "linearA" + m + "x" + k, 0, m, k, Gemm.MR, new Matrix(xs, 0, k, 1));
            float[] bp = packed(w, "linearB" + n + "x" + k, 1, n, k, Gemm.NR, new Matrix(ws, 0, k, 1));
            Gemm.compute(ap, bp, m, n, k, ys, 0, n, bs, 0, epilogue);
        });
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
        if (batch == 1) {
            Blas.multiply("matmul " + m + "x" + n + "x" + k + (ta ? "t" : "n") + (tb ? "t" : "n"), new Matrix(as, 0, axs, aqs), new Matrix(bs, 0, bxs, bqs), m, n, k, cs, 0, n, null, 0, 0,
                    () -> {
                        float[] ap = packed(a, "matmulA" + m + "x" + k + ta, 0, m, k, Gemm.MR, new Matrix(as, 0, axs, aqs));
                        float[] bp = packed(b, "matmulB" + n + "x" + k + tb, 1, n, k, Gemm.NR, new Matrix(bs, 0, bxs, bqs));
                        Gemm.compute(ap, bp, m, n, k, cs, 0, n, null, 0, 0);
                    });
            return true;
        }
        if (strideA == 0 && strideB == 0) {
            float[] ap = packed(a, "matmulA" + m + "x" + k + ta, 0, m, k, Gemm.MR, new Matrix(as, 0, axs, aqs));
            float[] bp = packed(b, "matmulB" + n + "x" + k + tb, 1, n, k, Gemm.NR, new Matrix(bs, 0, bxs, bqs));
            for (int z = 0; z < batch; z++) {
                Gemm.compute(ap, bp, m, n, k, cs, (long) z * m * n, n, null, 0, 0);
            }
            return true;
        }
        Cpu.Range one = (from, to) -> {
            for (int z = from; z < to; z++) {
                Matrix az = new Matrix(as, (long) z * strideA, axs, aqs);
                Matrix bz = new Matrix(bs, (long) z * strideB, bxs, bqs);
                long cOff = (long) z * m * n;
                // inside a parallel loop the library is not used and its race not run
                Blas.multiply("matmul " + m + "x" + n + "x" + k + (ta ? "t" : "n") + (tb ? "t" : "n") + " batched", az, bz, m, n, k, cs, cOff, n, null, 0, 0,
                        () -> Gemm.multiply(az, bz, m, n, k, cs, cOff, n, null, 0, 0));
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
        // the BLAS library, per image: the image's im2col matrix (the image itself for a 1 x 1,
        // stride 1 convolution) times the weights, the GEMM on all the library's threads
        Runnable blas = groups != 1 || !Blas.usable() ? null : () -> {
            MemorySegment col = pointwise ? null : Blas.scratch(0, (long) k * pixels);
            for (int img = 0; img < n; img++) {
                long xOff = (long) img * cin * h * wd;
                Matrix columns;
                if (pointwise) {
                    columns = new Matrix(xs, xOff, 1, (long) h * wd);
                } else {
                    Gemm.im2col(xs, xOff, cin, h, wd, r, stride, pad, dil, oh, ow, col, 0, pixels);
                    columns = new Matrix(col, 0, 1, pixels);
                }
                long cOff = (long) img * cout * pixels;
                Blas.sgemm(new Matrix(wsg, 0, k, 1), columns, cout, pixels, k, os, cOff, pixels, 0f);
                Gemm.epilogue(os, cOff, pixels, cout, pixels, bs, 0, epilogue);
            }
        };
        Storage weight = w.storage();
        Object owner = weight.isPersistent() ? weight : null;
        Runnable onednn = !Dnnl.available() ? null
                : () -> Dnnl.conv2d(xs, wsg, owner, weight.version(), bs, os, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, relu);
        // BNNS packs the weight and the bias into its filter: both versions name the filter's contents
        long versions = weight.version() * 1_000_003L + (bias != null ? bias.storage().version() : 0);
        Object owners = owner != null && (bias == null || bias.storage().isPersistent()) ? owner : null;
        Runnable bnns = !Bnns.available() ? null : () -> Bnns.conv2d(xs, wsg, owners, versions, bs, os, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, relu);
        Race.run("conv2d " + n + "x" + cin + "x" + h + "x" + wd + " " + cout + " r" + r + " s" + stride + " p" + pad + " d" + dil + " g" + groups + (relu ? " relu" : ""),
                () -> conv2dKernel(xs, wsg, os, bs, w, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups, cg, cog, k, pixels, pointwise, epilogue), blas, onednn, bnns);
        return true;
    }

    private static void conv2dKernel(MemorySegment xs, MemorySegment wsg, MemorySegment os, MemorySegment bs, Tensor w, int n, int cin, int h, int wd, int cout, int oh, int ow, int r,
            int stride, int pad, int dil, int groups, int cg, int cog, int k, int pixels, boolean pointwise, int epilogue) {
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
            return;
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
                    Elementwise.binary(ai.beestream.jitnn.kernels.ElementwiseKernels.ADD, xs.asSlice(off * 4, cols * 4L), 1, cols, rs.asSlice(off * 4, cols * 4L), 1, cols,
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
        boolean sum = op == ai.beestream.jitnn.kernels.ReduceKernels.SUM;
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
                float acc = op == ai.beestream.jitnn.kernels.ReduceKernels.MAX ? Float.NEGATIVE_INFINITY : op == ai.beestream.jitnn.kernels.ReduceKernels.MIN ? Float.POSITIVE_INFINITY
                        : op == ai.beestream.jitnn.kernels.ReduceKernels.PROD ? 1 : 0;
                for (int q = 0; q < len; q++) {
                    float v = xs.getAtIndex(F, at + (long) q * inner);
                    acc = op == ai.beestream.jitnn.kernels.ReduceKernels.MAX ? Math.max(acc, v) : op == ai.beestream.jitnn.kernels.ReduceKernels.MIN ? Math.min(acc, v)
                            : op == ai.beestream.jitnn.kernels.ReduceKernels.PROD ? acc * v : acc + v;
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
        int z = a[0], t = a[1], s = a[2], d = a[3];
        int[] meta = FlashAttention.meta(a);
        long qRow = meta[5 + 3];
        long kRow = meta[9 + 3];
        long vRow = meta[13 + 3];
        long oRow = meta[17 + 3];
        int blocks = (t + ATTENTION_ROWS - 1) / ATTENTION_ROWS;
        Runnable kernel = () -> Cpu.parallelFor(z * blocks, 1, (from, to) -> {
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
                long q0 = FlashAttention.at(meta, 0, zi, r0);
                long k0 = FlashAttention.at(meta, 1, zi, 0);
                long v0 = FlashAttention.at(meta, 2, zi, 0);
                long o0 = FlashAttention.at(meta, 3, zi, r0);
                // scores = q k^T
                Gemm.multiply(new Matrix(qs, q0, qRow, 1), new Matrix(ks, k0, kRow, 1), rows, s, d, scores, 0, s, null, 0, 0);
                attentionRows(a, scale, ms, ls, scores, 0, zi, r0, rows);
                // o = p v
                Gemm.multiply(new Matrix(scores, 0, s, 1), new Matrix(vs, v0, 1, vRow), rows, d, s, os, o0, oRow, null, 0, 0);
            }
        });
        // the BLAS library: each head's two products on all the library's threads, the softmax rows
        // of every head in parallel between them (the heads' products are too small to split)
        Runnable blas = !Blas.usable() || !qs.isNative() || !ks.isNative() || !vs.isNative() || !os.isNative() ? null : () -> {
            MemorySegment scores = Blas.scratch(0, (long) z * t * s);
            for (int zi = 0; zi < z; zi++) {
                Blas.sgemm(new Matrix(qs, FlashAttention.at(meta, 0, zi, 0), qRow, 1), new Matrix(ks, FlashAttention.at(meta, 1, zi, 0), kRow, 1), t, s, d, scores, (long) zi * t * s, s,
                        0f);
            }
            Cpu.parallelFor(z * t, Math.max(1, 4096 / Math.max(1, s)), (from, to) -> {
                for (int row = from; row < to; row++) {
                    attentionRows(a, scale, ms, ls, scores, (long) row * s, row / t, row % t, 1);
                }
            });
            for (int zi = 0; zi < z; zi++) {
                Blas.sgemm(new Matrix(scores, (long) zi * t * s, s, 1), new Matrix(vs, FlashAttention.at(meta, 2, zi, 0), 1, vRow), t, d, s, os, FlashAttention.at(meta, 3, zi, 0), oRow,
                        0f);
            }
        };
        Race.run("attention " + z + "x" + t + "x" + s + "x" + d + (a[4] != 0 ? " causal" : "") + (ms != null ? " mask" : ""), kernel, blas, null);
        return true;
    }

    /**
     * Scale, mask and causal limit, then softmax in place, of {@code rows} rows of attention scores
     * (query rows {@code r0 ..} of head {@code zi}, the first at {@code off} of {@code scores}, {@code
     * s} apart), saving each row's log-sum-exp.
     */
    private static void attentionRows(int[] a, float scale, MemorySegment ms, MemorySegment ls, MemorySegment scores, long off0, int zi, int r0, int rows) {
        int t = a[1], s = a[2], heads = a[5];
        boolean causal = a[4] != 0;
        int b = zi / heads;
        int h = zi - b * heads;
        for (int i = 0; i < rows; i++) {
            long off = off0 + (long) i * s;
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
    }

    /**
     * BatchNorm of x [n, c, hw] per channel: in training, the batch statistics (saved as mean and
     * invstd, the running statistics updated in place); in inference the running ones; then
     * y = act((x - mean) invstd gamma + beta). Statistics sum per-row vector sums in double.
     */
    static boolean batchNorm(Tensor x, Tensor gamma, Tensor beta, Tensor runMean, Tensor runVar, Tensor y, Tensor mean, Tensor invstd, int n, int c, int hw, boolean training,
            boolean relu, float eps, float momentum) {
        MemorySegment xs = f32(x);
        MemorySegment gs = f32(gamma);
        MemorySegment bs = f32(beta);
        MemorySegment rm = f32(runMean);
        MemorySegment rv = f32(runVar);
        MemorySegment ys = f32(y);
        MemorySegment ms = mean == null ? null : f32(mean);
        MemorySegment is = invstd == null ? null : f32(invstd);
        if (xs == null || gs == null || bs == null || rm == null || rv == null || ys == null || training && (ms == null || is == null)) {
            return false;
        }
        long count = (long) n * hw;
        Cpu.parallelFor(c, 1, (from, to) -> {
            for (int ch = from; ch < to; ch++) {
                float mu;
                float inv;
                if (training) {
                    double sum = 0;
                    for (int b = 0; b < n; b++) {
                        sum += Elementwise.sum(xs, ((long) b * c + ch) * hw, hw);
                    }
                    mu = (float) (sum / count);
                    double sq = 0;
                    for (int b = 0; b < n; b++) {
                        sq += Elementwise.sumSquares(xs, ((long) b * c + ch) * hw, hw, mu);
                    }
                    double var = sq / count;
                    inv = (float) (1 / Math.sqrt(var + eps));
                    ms.setAtIndex(F, ch, mu);
                    is.setAtIndex(F, ch, inv);
                    double unbiased = count > 1 ? sq / (count - 1) : var;
                    rm.setAtIndex(F, ch, (float) ((1 - momentum) * rm.getAtIndex(F, ch) + momentum * mu));
                    rv.setAtIndex(F, ch, (float) ((1 - momentum) * rv.getAtIndex(F, ch) + momentum * unbiased));
                } else {
                    mu = rm.getAtIndex(F, ch);
                    inv = (float) (1 / Math.sqrt(rv.getAtIndex(F, ch) + eps));
                }
                float scale = inv * gs.getAtIndex(F, ch);
                float shift = bs.getAtIndex(F, ch) - mu * scale;
                for (int b = 0; b < n; b++) {
                    long off = ((long) b * c + ch) * hw;
                    Elementwise.affineRow(xs, off, scale, shift, relu, ys, off, hw);
                }
            }
        });
        return true;
    }

    /** BatchNorm's gradient per channel from the saved mean and invstd: dx, dgamma and dbeta. */
    static boolean batchNormBackward(Tensor dy, Tensor x, Tensor mean, Tensor invstd, Tensor gamma, Tensor dx, Tensor dgamma, Tensor dbeta, int n, int c, int hw) {
        MemorySegment dys = f32(dy);
        MemorySegment xs = f32(x);
        MemorySegment ms = f32(mean);
        MemorySegment is = f32(invstd);
        MemorySegment gs = f32(gamma);
        MemorySegment dxs = f32(dx);
        MemorySegment dgs = f32(dgamma);
        MemorySegment dbs = f32(dbeta);
        if (dys == null || xs == null || ms == null || is == null || gs == null || dxs == null || dgs == null || dbs == null) {
            return false;
        }
        double count = (double) n * hw;
        Cpu.parallelFor(c, 1, (from, to) -> {
            for (int ch = from; ch < to; ch++) {
                float mu = ms.getAtIndex(F, ch);
                float inv = is.getAtIndex(F, ch);
                double sa = 0;
                double sb = 0;
                for (int b = 0; b < n; b++) {
                    long off = ((long) b * c + ch) * hw;
                    sa += Elementwise.sum(dys, off, hw);
                    sb += Elementwise.sumCentredProduct(dys, off, xs, off, mu, hw);
                }
                sb *= inv;
                dbs.setAtIndex(F, ch, (float) sa);
                dgs.setAtIndex(F, ch, (float) sb);
                // dx = g inv (dy - sa / count - (x - mu) inv sb / count)
                float gi = gs.getAtIndex(F, ch) * inv;
                float bx = (float) (-gi * inv * sb / count);
                float c0 = (float) (-gi * sa / count) - bx * mu;
                for (int b = 0; b < n; b++) {
                    long off = ((long) b * c + ch) * hw;
                    Elementwise.combineRow(dys, off, gi, xs, off, bx, c0, dxs, off, hw);
                }
            }
        });
        return true;
    }

    private static final ThreadLocal<float[]> COLUMNS = ThreadLocal.withInitial(() -> new float[0]);

    /** Images per BLAS GEMM of a convolution's gradient: about 4096 pixels deep, at most 16 Mi floats per operand. */
    private static int imagesPerGemm(int n, int pixels, int rows) {
        long byDepth = Math.max(1, 4096 / Math.max(1, pixels));
        long bySize = Math.max(1, (1L << 24) / Math.max(1L, (long) rows * pixels));
        return (int) Math.max(1, Math.min(n, Math.min(byDepth, bySize)));
    }

    /** dx = the input gradient of conv2d: the CPU kernel or oneDNN, whichever is faster for the shape. */
    static boolean conv2dBackwardData(Tensor dy, Tensor w, Tensor dx, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups) {
        MemorySegment dys = f32(dy);
        MemorySegment ws = f32(w);
        MemorySegment dxs = f32(dx);
        if (dys == null || ws == null || dxs == null) {
            return false;
        }
        int k = cin * r * r;
        int pixels = oh * ow;
        boolean pointwise = r == 1 && stride == 1 && pad == 0;
        // the BLAS library, over a chunk of images at once: the columns' gradient W^T dy, then col2im
        // (a 1 x 1, stride 1 convolution writes W^T dy straight into dx, one image at a time)
        Runnable blas = groups != 1 || !Blas.usable() ? null : () -> {
            int chunk = pointwise ? 1 : imagesPerGemm(n, pixels, Math.max(k, cout));
            Matrix wT = new Matrix(ws, 0, 1, k);
            for (int img0 = 0; img0 < n; img0 += chunk) {
                int images = Math.min(chunk, n - img0);
                long width = (long) images * pixels;
                Matrix grads;
                if (images == 1) {
                    grads = new Matrix(dys, (long) img0 * cout * pixels, 1, pixels);
                } else {
                    MemorySegment g = Blas.scratch(0, cout * width);
                    Gemm.gatherImages(dys, (long) img0 * cout * pixels, (long) cout * pixels, images, cout, pixels, g);
                    grads = new Matrix(g, 0, 1, width);
                }
                if (pointwise) {
                    Blas.sgemm(wT, grads, cin, pixels, cout, dxs, (long) img0 * cin * pixels, pixels, 0f);
                    continue;
                }
                MemorySegment dcol = Blas.scratch(1, k * width);
                Blas.sgemm(wT, grads, k, (int) width, cout, dcol, 0, width, 0f);
                for (int i = 0; i < images; i++) {
                    Gemm.col2im(dcol, (long) i * pixels, width, cin, h, wd, r, stride, pad, dil, oh, ow, dxs, (long) (img0 + i) * cin * h * wd);
                }
            }
        };
        Storage weight = w.storage();
        Object owner = weight.isPersistent() ? weight : null;
        Race.run("conv2d_bwd_data " + n + "x" + cin + "x" + h + "x" + wd + " " + cout + " r" + r + " s" + stride + " p" + pad + " d" + dil + " g" + groups,
                () -> conv2dBackwardDataKernel(dy, w, dx, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups), blas,
                !Dnnl.available() ? null : () -> Dnnl.conv2dBackwardData(dys, ws, owner, weight.version(), dxs, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups),
                !Bnns.available() ? null : () -> Bnns.conv2dBackwardData(dys, ws, owner, weight.version(), dxs, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups));
        return true;
    }

    /** dw = the weight gradient of conv2d: the CPU kernel or oneDNN, whichever is faster for the shape. */
    static boolean conv2dBackwardWeight(Tensor dy, Tensor x, Tensor dw, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups) {
        MemorySegment dys = f32(dy);
        MemorySegment xs = f32(x);
        MemorySegment dws = f32(dw);
        if (dys == null || xs == null || dws == null) {
            return false;
        }
        int k = cin * r * r;
        int pixels = oh * ow;
        boolean pointwise = r == 1 && stride == 1 && pad == 0;
        // the BLAS library, over a chunk of images at once: dy (the images side by side) times their
        // im2col matrices transposed, accumulated into dw from the second chunk on
        Runnable blas = groups != 1 || !Blas.usable() ? null : () -> {
            int chunk = imagesPerGemm(n, pixels, Math.max(k, cout));
            for (int img0 = 0; img0 < n; img0 += chunk) {
                int images = Math.min(chunk, n - img0);
                long width = (long) images * pixels;
                Matrix grads;
                Matrix columns;
                if (images == 1) {
                    grads = new Matrix(dys, (long) img0 * cout * pixels, width, 1);
                } else {
                    MemorySegment g = Blas.scratch(0, cout * width);
                    Gemm.gatherImages(dys, (long) img0 * cout * pixels, (long) cout * pixels, images, cout, pixels, g);
                    grads = new Matrix(g, 0, width, 1);
                }
                if (pointwise && images == 1) {
                    columns = new Matrix(xs, (long) img0 * cin * pixels, pixels, 1);
                } else {
                    MemorySegment col = Blas.scratch(1, k * width);
                    for (int i = 0; i < images; i++) {
                        Gemm.im2col(xs, (long) (img0 + i) * cin * h * wd, cin, h, wd, r, stride, pad, dil, oh, ow, col, (long) i * pixels, width);
                    }
                    columns = new Matrix(col, 0, width, 1);
                }
                Blas.sgemm(grads, columns, cout, k, (int) width, dws, 0, k, img0 == 0 ? 0f : 1f);
            }
        };
        Race.run("conv2d_bwd_weight " + n + "x" + cin + "x" + h + "x" + wd + " " + cout + " r" + r + " s" + stride + " p" + pad + " d" + dil + " g" + groups,
                () -> conv2dBackwardWeightKernel(dy, x, dw, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups), blas,
                !Dnnl.available() ? null : () -> Dnnl.conv2dBackwardWeights(dys, xs, dws, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups),
                !Bnns.available() ? null : () -> Bnns.conv2dBackwardWeights(dys, xs, dws, n, cin, h, wd, cout, oh, ow, r, stride, pad, dil, groups));
        return true;
    }

    /**
     * The input gradient of a convolution: per image and group, dcol = W^T dy (a GEMM of depth
     * cout / groups), then col2im adds the columns back onto the input's pixels.
     */
    private static boolean conv2dBackwardDataKernel(Tensor dy, Tensor w, Tensor dx, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups) {
        MemorySegment dys = f32(dy);
        MemorySegment ws = f32(w);
        MemorySegment dxs = f32(dx);
        if (dys == null || ws == null || dxs == null) {
            return false;
        }
        int cg = cin / groups;
        int cog = cout / groups;
        int k = cg * r * r;
        int pixels = oh * ow;
        float[][] weightsT = new float[groups][];
        for (int g = 0; g < groups; g++) {
            weightsT[g] = Gemm.packA(new Matrix(ws, (long) g * cog * k, 1, k), k, cog, new float[Gemm.packedSize(k, cog, Gemm.MR)]);
        }
        int items = n * groups;
        Cpu.Range body = (from, to) -> {
            for (int item = from; item < to; item++) {
                int img = item / groups;
                int g = item % groups;
                float[] bp = Gemm.packB(new Matrix(dys, ((long) img * cout + (long) g * cog) * pixels, 1, pixels), pixels, cog, Gemm.scratch(1, Gemm.packedSize(pixels, cog, Gemm.NR)));
                float[] cols = COLUMNS.get();
                if (cols.length < k * pixels) {
                    cols = new float[k * pixels];
                    COLUMNS.set(cols);
                }
                MemorySegment cs = MemorySegment.ofArray(cols);
                Gemm.compute(weightsT[g], bp, k, pixels, cog, cs, 0, pixels, null, 0, 0);
                long base = ((long) img * cin + (long) g * cg) * h * wd;
                float[] dcol = cols;
                Cpu.parallelFor(cg, 1, (c0, c1) -> {
                    for (int ci = c0; ci < c1; ci++) {
                        long plane = base + (long) ci * h * wd;
                        dxs.asSlice(plane * 4, (long) h * wd * 4).fill((byte) 0);
                        for (int kh = 0; kh < r; kh++) {
                            for (int kw = 0; kw < r; kw++) {
                                int q = (ci * r + kh) * r + kw;
                                for (int y = 0; y < oh; y++) {
                                    int ih = y * stride - pad + kh * dil;
                                    if (ih < 0 || ih >= h) {
                                        continue;
                                    }
                                    long rowAt = plane + (long) ih * wd;
                                    int src = q * pixels + y * ow;
                                    for (int xo = 0; xo < ow; xo++) {
                                        int iw = xo * stride - pad + kw * dil;
                                        if (iw >= 0 && iw < wd) {
                                            dxs.setAtIndex(F, rowAt + iw, dxs.getAtIndex(F, rowAt + iw) + dcol[src + xo]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
        };
        if (items * 2 >= Cpu.threads()) {
            Cpu.parallelFor(items, 1, body);
        } else {
            body.run(0, items);
        }
        return true;
    }

    /**
     * The weight gradient of a convolution: dw[g] = sum over images of dy (cout / groups rows) times
     * the transposed im2col matrix, images grouped so each GEMM is about 8192 pixels deep,
     * accumulating into dw.
     */
    private static boolean conv2dBackwardWeightKernel(Tensor dy, Tensor x, Tensor dw, int n, int cin, int h, int wd, int cout, int oh, int ow, int r, int stride, int pad, int dil, int groups) {
        MemorySegment dys = f32(dy);
        MemorySegment xs = f32(x);
        MemorySegment dws = f32(dw);
        if (dys == null || xs == null || dws == null) {
            return false;
        }
        int cg = cin / groups;
        int cog = cout / groups;
        int k = cg * r * r;
        int pixels = oh * ow;
        // images per GEMM: about 8192 pixels deep, at most 8 Mi floats per packed operand
        int perChunk = Math.max(1, Math.min(n, (int) Math.min(Math.max(1, 8192 / Math.max(1, pixels)), (1L << 23) / Math.max(1L, (long) Math.max(k, cog) * pixels))));
        for (int g = 0; g < groups; g++) {
            for (int img0 = 0; img0 < n; img0 += perChunk) {
                int images = Math.min(perChunk, n - img0);
                int depth = images * pixels;
                float[] ap = Gemm.packRowsOverImages(dys, (long) g * cog * pixels, (long) cout * pixels, img0, images, cog, pixels, Gemm.scratch(0, Gemm.packedSize(cog, depth, Gemm.MR)));
                float[] bp = Gemm.packIm2colT(xs, (long) g * cg * h * wd, (long) cin * h * wd, img0, images, cg, h, wd, r, stride, pad, dil, oh, ow,
                        Gemm.scratch(1, Gemm.packedSize(k, depth, Gemm.NR)));
                Gemm.compute(ap, bp, cog, k, depth, dws, (long) g * cog * k, k, null, 0, img0 == 0 ? 0 : Gemm.ACCUMULATE);
            }
        }
        return true;
    }

    /** Mean cross-entropy of logits x [rows, cols] against labels: one log-sum-exp per row, in parallel. */
    static boolean crossEntropy(Tensor x, Tensor labels, Tensor out, int rows, int cols) {
        MemorySegment xs = f32(x);
        MemorySegment os = f32(out);
        MemorySegment ls = labels.storage().host() instanceof IntArray ia ? ia.getSegment() : null;
        if (xs == null || os == null || ls == null) {
            return false;
        }
        double[] loss = new double[rows];
        Cpu.parallelFor(rows, Math.max(1, Elementwise_GRAIN / Math.max(1, cols)), (from, to) -> {
            float[] row = new float[cols];
            MemorySegment rs = MemorySegment.ofArray(row);
            for (int rr = from; rr < to; rr++) {
                int label = ls.getAtIndex(I, rr);
                if (label < 0 || label >= cols) {
                    throw new IllegalArgumentException("label " + label + " out of range [0, " + cols + ")");
                }
                MemorySegment.copy(xs, (long) rr * cols * 4, rs, 0, cols * 4L);
                float lse = Elementwise.softmaxRow(rs, 0, cols);
                loss[rr] = lse - xs.getAtIndex(F, (long) rr * cols + label);
            }
        });
        double total = 0;
        for (double v : loss) {
            total += v;
        }
        os.setAtIndex(F, 0, (float) (total / rows));
        return true;
    }

    /** dx = (softmax(x) - onehot(label)) * g / rows, per row in parallel. */
    static boolean crossEntropyBackward(Tensor x, Tensor labels, Tensor g, Tensor dx, int rows, int cols) {
        MemorySegment xs = f32(x);
        MemorySegment gs = f32(g);
        MemorySegment ds = f32(dx);
        MemorySegment ls = labels.storage().host() instanceof IntArray ia ? ia.getSegment() : null;
        if (xs == null || gs == null || ds == null || ls == null) {
            return false;
        }
        float scale = gs.getAtIndex(F, 0) / rows;
        Cpu.parallelFor(rows, Math.max(1, Elementwise_GRAIN / Math.max(1, cols)), (from, to) -> {
            for (int rr = from; rr < to; rr++) {
                long off = (long) rr * cols;
                MemorySegment.copy(xs, off * 4, ds, off * 4, cols * 4L);
                Elementwise.softmaxRow(ds, off, cols);
                Elementwise.scale(ds, off, cols, scale);
                long at = off + ls.getAtIndex(I, rr);
                ds.setAtIndex(F, at, ds.getAtIndex(F, at) - scale);
            }
        });
        return true;
    }

    /** dw [vocab, dim]: row v is the sum of the gradient rows of the tokens that read row v (offsets, positions: CSR by row). */
    static boolean embeddingBackward(Tensor dy, Tensor offsets, Tensor positions, Tensor dw, int vocab, int dim) {
        MemorySegment dys = f32(dy);
        MemorySegment dws = f32(dw);
        MemorySegment os = offsets.storage().host() instanceof IntArray a ? a.getSegment() : null;
        MemorySegment ps = positions.storage().host() instanceof IntArray b ? b.getSegment() : null;
        if (dys == null || dws == null || os == null || ps == null) {
            return false;
        }
        Cpu.parallelFor(vocab, Math.max(1, Elementwise_GRAIN / Math.max(1, dim)), (from, to) -> {
            for (int v = from; v < to; v++) {
                long row = (long) v * dim;
                int k0 = os.getAtIndex(I, v);
                int k1 = os.getAtIndex(I, v + 1);
                if (k0 == k1) {
                    dws.asSlice(row * 4, dim * 4L).fill((byte) 0);
                    continue;
                }
                MemorySegment.copy(dys, (long) ps.getAtIndex(I, k0) * dim * 4, dws, row * 4, dim * 4L);
                for (int k = k0 + 1; k < k1; k++) {
                    Elementwise.addRow(dys, (long) ps.getAtIndex(I, k) * dim, dws, row, dim);
                }
            }
        });
        return true;
    }

    /** LayerNorm (or RMSNorm) gradient: dx per row; dw (and db) summed over rows in per-chunk partials. */
    static boolean rowNormBackward(Tensor dy, Tensor x, Tensor w, Tensor mean, Tensor rstd, Tensor dx, Tensor dw, Tensor db, int rows, int cols, boolean layer) {
        MemorySegment dys = f32(dy);
        MemorySegment xs = f32(x);
        MemorySegment ws = f32(w);
        MemorySegment ms = f32(mean);
        MemorySegment ss = f32(rstd);
        MemorySegment dxs = f32(dx);
        MemorySegment dws = f32(dw);
        MemorySegment dbs = db == null ? null : f32(db);
        if (dys == null || xs == null || ws == null || ms == null || ss == null || dxs == null || dws == null || db != null && dbs == null) {
            return false;
        }
        int chunks = Math.max(1, Math.min(Cpu.threads(), rows));
        float[][] pw = new float[chunks][cols];
        float[][] pb = new float[chunks][cols];
        Cpu.parallelFor(chunks, 1, (c0, c1) -> {
            float[] g = new float[cols];
            for (int ch = c0; ch < c1; ch++) {
                int r0 = (int) ((long) rows * ch / chunks);
                int r1 = (int) ((long) rows * (ch + 1) / chunks);
                float[] aw = pw[ch];
                float[] ab = pb[ch];
                for (int rr = r0; rr < r1; rr++) {
                    long off = (long) rr * cols;
                    float mu = ms.getAtIndex(F, rr);
                    float is = ss.getAtIndex(F, rr);
                    double sa = 0;
                    double sb = 0;
                    for (int c = 0; c < cols; c++) {
                        float d = dys.getAtIndex(F, off + c);
                        float xhat = (xs.getAtIndex(F, off + c) - mu) * is;
                        float gc = d * ws.getAtIndex(F, c);
                        g[c] = gc;
                        sa += gc;
                        sb += gc * xhat;
                        aw[c] += d * xhat;
                        ab[c] += d;
                    }
                    float meanG = layer ? (float) (sa / cols) : 0;
                    float meanGX = (float) (sb / cols);
                    for (int c = 0; c < cols; c++) {
                        float xhat = (xs.getAtIndex(F, off + c) - mu) * is;
                        dxs.setAtIndex(F, off + c, is * (g[c] - meanG - xhat * meanGX));
                    }
                }
            }
        });
        for (int c = 0; c < cols; c++) {
            double sw = 0;
            double sbias = 0;
            for (int ch = 0; ch < chunks; ch++) {
                sw += pw[ch][c];
                sbias += pb[ch][c];
            }
            dws.setAtIndex(F, c, (float) sw);
            if (dbs != null) {
                dbs.setAtIndex(F, c, (float) sbias);
            }
        }
        return true;
    }

    private static final ThreadLocal<float[][]> ATTENTION_BACKWARD = ThreadLocal.withInitial(() -> new float[2][0]);

    /**
     * The attention gradient per problem (batch and head), as GEMMs: P from the saved log-sum-exp,
     * dV = P^T dO, dP = dO V^T, dS = P (dP - rowsum(dO o O)) scale, dQ = dS K, dK = dS^T Q.
     */
    static boolean attentionBackward(int[] a, float scale, Tensor q, Tensor k, Tensor v, Tensor mask, Tensor o, Tensor lse, Tensor dout, Tensor dq, Tensor dk, Tensor dv) {
        MemorySegment qs = f32(q);
        MemorySegment ks = f32(k);
        MemorySegment vs = f32(v);
        MemorySegment ms = mask == null ? null : f32(mask);
        MemorySegment os = f32(o);
        MemorySegment ls = f32(lse);
        MemorySegment gos = f32(dout);
        MemorySegment dqs = f32(dq);
        MemorySegment dks = f32(dk);
        MemorySegment dvs = f32(dv);
        if (qs == null || ks == null || vs == null || os == null || ls == null || gos == null || dqs == null || dks == null || dvs == null || mask != null && ms == null) {
            return false;
        }
        int z = a[0], t = a[1], s = a[2], d = a[3], heads = a[5];
        boolean causal = a[4] != 0;
        int[] meta = FlashAttention.meta(a);
        long qRow = meta[8], kRow = meta[12], vRow = meta[16], oRow = meta[20];
        Cpu.parallelFor(z, 1, (from, to) -> {
            float[][] buf = ATTENTION_BACKWARD.get();
            if (buf[0].length < t * s) {
                buf[0] = new float[t * s];
                buf[1] = new float[t * s];
            }
            MemorySegment p = MemorySegment.ofArray(buf[0]);
            MemorySegment dp = MemorySegment.ofArray(buf[1]);
            for (int zi = from; zi < to; zi++) {
                long q0 = FlashAttention.at(meta, 0, zi, 0);
                long k0 = FlashAttention.at(meta, 1, zi, 0);
                long v0 = FlashAttention.at(meta, 2, zi, 0);
                long o0 = FlashAttention.at(meta, 3, zi, 0);
                int b = zi / heads;
                int h = zi - b * heads;
                // P = exp(scores * scale + mask - lse)
                Gemm.multiply(new Matrix(qs, q0, qRow, 1), new Matrix(ks, k0, kRow, 1), t, s, d, p, 0, s, null, 0, 0);
                for (int i = 0; i < t; i++) {
                    long off = (long) i * s;
                    float l = ls.getAtIndex(F, (long) zi * t + i);
                    for (int j = 0; j < s; j++) {
                        float sc = p.getAtIndex(F, off + j) * scale;
                        if (ms != null) {
                            sc += ms.getAtIndex(F, (long) b * a[6] + (long) h * a[7] + (long) i * a[8] + (long) j * a[9]);
                        }
                        p.setAtIndex(F, off + j, causal && j > i ? 0 : (float) Math.exp(sc - l));
                    }
                }
                // dV = P^T dO
                Gemm.multiply(new Matrix(p, 0, 1, s), new Matrix(gos, o0, 1, oRow), s, d, t, dvs, v0, vRow, null, 0, 0);
                // dP = dO V^T
                Gemm.multiply(new Matrix(gos, o0, oRow, 1), new Matrix(vs, v0, vRow, 1), t, s, d, dp, 0, s, null, 0, 0);
                // dS = P (dP - D) scale, D = rowsum(dO o O); kept in dp
                for (int i = 0; i < t; i++) {
                    float dd = 0;
                    for (int c = 0; c < d; c++) {
                        dd += gos.getAtIndex(F, o0 + i * oRow + c) * os.getAtIndex(F, o0 + i * oRow + c);
                    }
                    long off = (long) i * s;
                    for (int j = 0; j < s; j++) {
                        dp.setAtIndex(F, off + j, p.getAtIndex(F, off + j) * (dp.getAtIndex(F, off + j) - dd) * scale);
                    }
                }
                // dQ = dS K, dK = dS^T Q
                Gemm.multiply(new Matrix(dp, 0, s, 1), new Matrix(ks, k0, 1, kRow), t, d, s, dqs, q0, qRow, null, 0, 0);
                Gemm.multiply(new Matrix(dp, 0, 1, s), new Matrix(qs, q0, 1, qRow), s, d, t, dks, k0, kRow, null, 0, 0);
            }
        });
        return true;
    }
}
