package ai.beestream.jitnn.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.WeakHashMap;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * FP32 matrix multiplication on the CPU, C [m, n] = A [m, k] B [k, n] with an epilogue (bias per row
 * or per column, ReLU, GELU), in the BLIS scheme: A is packed into panels of {@link #MR} rows and B
 * into panels of {@link #NR} columns, each holding the full depth k; C is cut into tiles of whole
 * panels that run in parallel, and each tile walks k in blocks of {@code jitnn.cpu.gemm.kc} so the A
 * block stays in L2 and one B panel in L1 while a register-tiled micro-kernel computes MR x NR
 * outputs with vector FMAs. Packed operands that do not change (weights) are cached.
 */
public final class Gemm {

    static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;
    static final int L = S.length();
    /** Bytes in one vector. */
    private static final long V = 4L * L;
    /** Rows of A per micro-tile. */
    public static final int MR = 6;
    /** Columns of B per micro-tile: two vectors. */
    public static final int NR = 2 * L;

    private static final int KC = Integer.getInteger("jitnn.cpu.gemm.kc", 256);
    private static final int MC_PANELS = Integer.getInteger("jitnn.cpu.gemm.mcPanels", 16);
    private static final int NC_PANELS = Math.max(1, Integer.getInteger("jitnn.cpu.gemm.nc", 256) / NR);
    /** Multiply-adds a tile of C holds at least, so small products are not cut finer than pays. */
    private static final long TILE_MIN_MACS = Long.getLong("jitnn.cpu.gemm.tileMacs", 1L << 20);
    /** Elements one parallel chunk packs at least. */
    private static final int PACK_GRAIN = 1 << 15;

    private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    private static final ValueLayout.OfFloat F = ValueLayout.JAVA_FLOAT;

    /** Epilogue: add {@code bias[i]} to row i. */
    public static final int BIAS_ROW = 1;
    /** Epilogue: add {@code bias[j]} to column j. */
    public static final int BIAS_COL = 2;
    /** Epilogue: max(x, 0), after the bias. */
    public static final int RELU = 4;
    /** Epilogue: exact GELU, x Phi(x), after the bias. */
    public static final int GELU = 8;
    /** C += A B: add the product to C's contents instead of overwriting them (before any other epilogue). */
    public static final int ACCUMULATE = 16;

    private Gemm() {
    }

    /**
     * A matrix operand of {@code rows} x {@code depth} values in {@code seg}: the value at row x and
     * depth q is the float at index {@code off + x * xs + q * qs}. A row-major [m, k] A is (k, 1); a
     * row-major [k, n] B, read by columns, is (1, n); transposed operands swap the strides.
     */
    public record Matrix(MemorySegment seg, long off, long xs, long qs) {
    }

    private static final boolean STATS = Boolean.getBoolean("jitnn.cpu.stats");
    private static final java.util.concurrent.atomic.LongAdder PACK_NANOS = new java.util.concurrent.atomic.LongAdder();
    private static final java.util.concurrent.atomic.LongAdder COMPUTE_NANOS = new java.util.concurrent.atomic.LongAdder();

    /** A start time for {@link #packed} or {@link #computed}: only with {@code jitnn.cpu.stats}, and only outside parallel loops (wall time). */
    private static long timing() {
        return STATS && !Cpu.inside() ? System.nanoTime() : 0;
    }

    private static void packed(long t0) {
        if (t0 != 0) {
            PACK_NANOS.add(System.nanoTime() - t0);
        }
    }

    private static void computed(long t0) {
        if (t0 != 0) {
            COMPUTE_NANOS.add(System.nanoTime() - t0);
        }
    }

    /**
     * With {@code jitnn.cpu.stats}: {@code [packing ns, multiply ns]} of the GEMMs called outside
     * parallel loops since the last call, which resets them.
     */
    public static long[] drainTimes() {
        return new long[] { PACK_NANOS.sumThenReset(), COMPUTE_NANOS.sumThenReset() };
    }

    /** Panels of {@code w} rows covering {@code rows}. */
    static int panels(int rows, int w) {
        return (rows + w - 1) / w;
    }

    /** Floats a packed operand of {@code rows} x {@code k} in panels of {@code w} takes. */
    public static int packedSize(int rows, int k, int w) {
        return panels(rows, w) * Math.max(k, 1) * w;
    }

    /** Packs A (m x k) into panels of {@link #MR} rows, in {@code dst} (at least {@link #packedSize}). */
    public static float[] packA(Matrix a, int m, int k, float[] dst) {
        long t0 = timing();
        pack(a, m, k, MR, dst);
        packed(t0);
        return dst;
    }

    /** Packs B (n columns x k) into panels of {@link #NR} columns, in {@code dst}. */
    public static float[] packB(Matrix b, int n, int k, float[] dst) {
        long t0 = timing();
        pack(b, n, k, NR, dst);
        packed(t0);
        return dst;
    }

    private static void pack(Matrix src, int rows, int k, int w, float[] dst) {
        int panels = panels(rows, w);
        int grain = Math.max(1, PACK_GRAIN / Math.max(1, k * w));
        Cpu.parallelFor(panels, grain, (from, to) -> {
            for (int p = from; p < to; p++) {
                packPanel(src, rows, k, w, dst, p);
            }
        });
    }

    private static void packPanel(Matrix src, int rows, int k, int w, float[] dst, int p) {
        int x0 = p * w;
        int valid = Math.min(w, rows - x0);
        int base = p * k * w;
        MemorySegment seg = src.seg();
        if (src.xs() == 1) {
            // a panel row is contiguous in the source: one bulk copy per depth
            for (int q = 0; q < k; q++) {
                int d = base + q * w;
                MemorySegment.copy(seg, F, (src.off() + x0 + q * src.qs()) * 4, dst, d, valid);
                for (int c = valid; c < w; c++) {
                    dst[d + c] = 0;
                }
            }
            return;
        }
        for (int c = 0; c < w; c++) {
            if (c >= valid) {
                for (int q = 0; q < k; q++) {
                    dst[base + q * w + c] = 0;
                }
                continue;
            }
            long at = src.off() + (long) (x0 + c) * src.xs();
            long qs = src.qs();
            for (int q = 0; q < k; q++) {
                dst[base + q * w + c] = seg.getAtIndex(F, at + q * qs);
            }
        }
    }

    /**
     * Packs the columns of a convolution's im2col matrix for one image and one group: column j is
     * output pixel (j / ow, j % ow), depth q is (ci, kh, kw) of {@code cg} input channels starting at
     * index {@code off} of {@code x} ([cg, h, w] planes), zero outside the image.
     */
    public static float[] packIm2col(MemorySegment x, long off, int cg, int h, int w, int r, int stride, int pad, int dil, int oh, int ow, float[] dst) {
        long t0 = timing();
        int n = oh * ow;
        int k = cg * r * r;
        int panels = panels(n, NR);
        int grain = Math.max(1, PACK_GRAIN / Math.max(1, k * NR));
        Cpu.parallelFor(panels, grain, (from, to) -> {
            int[] ihs = new int[NR];
            int[] iws = new int[NR];
            for (int p = from; p < to; p++) {
                int j0 = p * NR;
                int valid = Math.min(NR, n - j0);
                for (int c = 0; c < NR; c++) {
                    int j = j0 + Math.min(c, valid - 1);
                    ihs[c] = (j / ow) * stride - pad;
                    iws[c] = (j % ow) * stride - pad;
                }
                int base = p * k * NR;
                if (stride == 1) {
                    packIm2colStride1(x, off, cg, h, w, r, pad, dil, ow, dst, base, j0, valid);
                    continue;
                }
                int q = 0;
                for (int ci = 0; ci < cg; ci++) {
                    long plane = off + (long) ci * h * w;
                    for (int kh = 0; kh < r; kh++) {
                        for (int kw = 0; kw < r; kw++, q++) {
                            int d = base + q * NR;
                            int dh = kh * dil;
                            int dw = kw * dil;
                            for (int c = 0; c < NR; c++) {
                                int ih = ihs[c] + dh;
                                int iw = iws[c] + dw;
                                dst[d + c] = c < valid && ih >= 0 && ih < h && iw >= 0 && iw < w ? x.getAtIndex(F, plane + (long) ih * w + iw) : 0;
                            }
                        }
                    }
                }
            }
        });
        packed(t0);
        return dst;
    }

    /**
     * One im2col panel of a stride-1 convolution: the panel's pixels fall into runs along output
     * rows, and in each run the input row is read contiguously, so the in-image part is one bulk copy.
     */
    private static void packIm2colStride1(MemorySegment x, long off, int cg, int h, int w, int r, int pad, int dil, int ow, float[] dst, int base, int j0, int valid) {
        int q = 0;
        for (int ci = 0; ci < cg; ci++) {
            long plane = off + (long) ci * h * w;
            for (int kh = 0; kh < r; kh++) {
                for (int kw = 0; kw < r; kw++, q++) {
                    int d = base + q * NR;
                    int c = 0;
                    while (c < valid) {
                        int j = j0 + c;
                        int oh = j / ow;
                        int ox = j - oh * ow;
                        int len = Math.min(valid - c, ow - ox);
                        int ih = oh - pad + kh * dil;
                        int iw0 = ox - pad + kw * dil;
                        if (ih < 0 || ih >= h) {
                            java.util.Arrays.fill(dst, d + c, d + c + len, 0);
                        } else {
                            // columns c .. c + len read input columns iw0 .. iw0 + len - 1 of row ih
                            int lo = Math.max(0, -iw0);
                            int hi = Math.min(len, w - iw0);
                            if (lo > 0) {
                                java.util.Arrays.fill(dst, d + c, d + c + Math.min(lo, len), 0);
                            }
                            if (hi > lo) {
                                MemorySegment.copy(x, F, (plane + (long) ih * w + iw0 + lo) * 4, dst, d + c + lo, hi - lo);
                            }
                            if (hi < len) {
                                java.util.Arrays.fill(dst, d + c + Math.min(len, Math.max(hi, lo)), d + c + len, 0);
                            }
                        }
                        c += len;
                    }
                    for (; c < NR; c++) {
                        dst[d + c] = 0;
                    }
                }
            }
        }
    }

    /**
     * The im2col matrix of one image and group, unpacked and row-major: row q is (ci, kh, kw) of
     * {@code cg} input channels starting at index {@code off} of {@code x} ([cg, h, w] planes), column
     * j output pixel (j / ow, j % ow), zero outside the image; for a native BLAS. Element (q, j) goes
     * to index {@code dstOff + q * ld + j} of {@code dst} (several images side by side share rows).
     * Parallel over rows.
     */
    public static void im2col(MemorySegment x, long off, int cg, int h, int w, int r, int stride, int pad, int dil, int oh, int ow, MemorySegment dst, long dstOff, long ld) {
        int pixels = oh * ow;
        int rows = cg * r * r;
        Cpu.parallelFor(rows, Math.max(1, PACK_GRAIN / Math.max(1, pixels)), (from, to) -> {
            for (int q = from; q < to; q++) {
                int ci = q / (r * r);
                int kh = q / r % r;
                int kw = q % r;
                long plane = off + (long) ci * h * w;
                long row = dstOff + q * ld;
                for (int oy = 0; oy < oh; oy++) {
                    int iy = oy * stride - pad + kh * dil;
                    long at = row + (long) oy * ow;
                    if (iy < 0 || iy >= h) {
                        dst.asSlice(at * 4, (long) ow * 4).fill((byte) 0);
                        continue;
                    }
                    long src = plane + (long) iy * w;
                    int x0 = kw * dil - pad;
                    if (stride == 1) {
                        // columns ox with 0 <= ox + x0 < w are one contiguous run
                        int lo = Math.max(0, -x0);
                        int hi = Math.min(ow, w - x0);
                        if (hi <= lo) {
                            dst.asSlice(at * 4, (long) ow * 4).fill((byte) 0);
                            continue;
                        }
                        if (lo > 0) {
                            dst.asSlice(at * 4, (long) lo * 4).fill((byte) 0);
                        }
                        MemorySegment.copy(x, (src + lo + x0) * 4, dst, (at + lo) * 4, (long) (hi - lo) * 4);
                        if (hi < ow) {
                            dst.asSlice((at + hi) * 4, (long) (ow - hi) * 4).fill((byte) 0);
                        }
                        continue;
                    }
                    for (int ox = 0; ox < ow; ox++) {
                        int ix = ox * stride + x0;
                        dst.setAtIndex(F, at + ox, ix >= 0 && ix < w ? x.getAtIndex(F, src + ix) : 0f);
                    }
                }
            }
        });
    }

    /**
     * The inverse of {@link #im2col}: sets the {@code cg} planes ([h, w]) at index {@code off} of
     * {@code dx} to the sum of the im2col entries that read each input pixel, from the columns at
     * {@code colOff} (leading dimension {@code ld}) of {@code col}. Parallel over channels.
     */
    public static void col2im(MemorySegment col, long colOff, long ld, int cg, int h, int w, int r, int stride, int pad, int dil, int oh, int ow, MemorySegment dx, long off) {
        Cpu.parallelFor(cg, 1, (c0, c1) -> {
            for (int ci = c0; ci < c1; ci++) {
                long plane = off + (long) ci * h * w;
                dx.asSlice(plane * 4, (long) h * w * 4).fill((byte) 0);
                for (int kh = 0; kh < r; kh++) {
                    for (int kw = 0; kw < r; kw++) {
                        long row = colOff + (long) ((ci * r + kh) * r + kw) * ld;
                        for (int oy = 0; oy < oh; oy++) {
                            int iy = oy * stride - pad + kh * dil;
                            if (iy < 0 || iy >= h) {
                                continue;
                            }
                            long at = plane + (long) iy * w;
                            long src = row + (long) oy * ow;
                            int x0 = kw * dil - pad;
                            int lo = stride == 1 ? Math.max(0, -x0) : 0;
                            int hi = stride == 1 ? Math.min(ow, w - x0) : ow;
                            if (stride == 1) {
                                int ox = lo;
                                int vEnd = lo + S.loopBound(Math.max(0, hi - lo));
                                for (; ox < vEnd; ox += L) {
                                    long d = (at + ox + x0) * 4;
                                    FloatVector.fromMemorySegment(S, dx, d, ORDER).add(FloatVector.fromMemorySegment(S, col, (src + ox) * 4, ORDER)).intoMemorySegment(dx, d, ORDER);
                                }
                                for (; ox < hi; ox++) {
                                    dx.setAtIndex(F, at + ox + x0, dx.getAtIndex(F, at + ox + x0) + col.getAtIndex(F, src + ox));
                                }
                                continue;
                            }
                            for (int ox = 0; ox < ow; ox++) {
                                int ix = ox * stride + x0;
                                if (ix >= 0 && ix < w) {
                                    dx.setAtIndex(F, at + ix, dx.getAtIndex(F, at + ix) + col.getAtIndex(F, src + ox));
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /**
     * Gathers {@code rows} rows of {@code pixels} floats from each of {@code images} images (image i's
     * row c at index {@code off + i * imageStride + c * pixels} of {@code src}) into {@code dst},
     * row-major [rows, images * pixels]: the images side by side, for one GEMM over all of them.
     */
    public static void gatherImages(MemorySegment src, long off, long imageStride, int images, int rows, int pixels, MemorySegment dst) {
        Cpu.parallelFor(rows, Math.max(1, PACK_GRAIN / Math.max(1, images * pixels)), (from, to) -> {
            for (int c = from; c < to; c++) {
                for (int i = 0; i < images; i++) {
                    MemorySegment.copy(src, (off + i * imageStride + (long) c * pixels) * 4, dst, ((long) c * images * pixels + (long) i * pixels) * 4, (long) pixels * 4);
                }
            }
        });
    }

    /** Depth positions (pixels) one packing work item covers. */
    private static final int DEPTH_BLOCK = 512;

    /**
     * Packs the transposed im2col matrices of {@code images} images, from image {@code img0}, as B
     * for a weight gradient: column q is (ci, kh, kw) of {@code cg} input channels starting at index
     * {@code off} of image 0 (images {@code imageStride} floats apart), depth d is output pixel
     * {@code d % (oh * ow)} of image {@code img0 + d / (oh * ow)}; zero outside the image. Work items
     * are (panel, block of depth); each writes its panel rows contiguously.
     */
    public static float[] packIm2colT(MemorySegment x, long off, long imageStride, int img0, int images, int cg, int h, int w, int r, int stride, int pad, int dil, int oh, int ow,
            float[] dst) {
        long t0 = timing();
        int k = cg * r * r;
        int pixels = oh * ow;
        int depth = images * pixels;
        int panels = panels(k, NR);
        int blocks = (depth + DEPTH_BLOCK - 1) / DEPTH_BLOCK;
        Cpu.parallelFor(panels * blocks, 1, (from, to) -> {
            long[] plane = new long[NR];
            int[] dh = new int[NR];
            int[] dw = new int[NR];
            boolean[] live = new boolean[NR];
            for (int item = from; item < to; item++) {
                int p = item / blocks;
                int d0 = (item - p * blocks) * DEPTH_BLOCK;
                int d1 = Math.min(depth, d0 + DEPTH_BLOCK);
                for (int c = 0; c < NR; c++) {
                    int q = p * NR + c;
                    live[c] = q < k;
                    if (live[c]) {
                        int ci = q / (r * r);
                        plane[c] = off + (long) ci * h * w;
                        dh[c] = (q / r) % r * dil - pad;
                        dw[c] = q % r * dil - pad;
                    }
                }
                int base = p * depth * NR;
                for (int d = d0; d < d1; d++) {
                    int img = d / pixels;
                    int pix = d - img * pixels;
                    int y = pix / ow;
                    int xo = pix - y * ow;
                    long imageAt = (long) (img0 + img) * imageStride;
                    int at = base + d * NR;
                    for (int c = 0; c < NR; c++) {
                        float v = 0;
                        if (live[c]) {
                            int ih = y * stride + dh[c];
                            int iw = xo * stride + dw[c];
                            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                v = x.getAtIndex(F, imageAt + plane[c] + (long) ih * w + iw);
                            }
                        }
                        dst[at + c] = v;
                    }
                }
            }
        });
        packed(t0);
        return dst;
    }

    /**
     * Packs, as A, rows {@code 0 .. rows} of the output gradients of {@code images} images from
     * {@code img0}: row co, depth d = pixel {@code d % pixels} of image {@code img0 + d / pixels}, the
     * value at {@code off + image * imageStride + co * pixels + pixel}. Work items are (panel, block
     * of depth).
     */
    public static float[] packRowsOverImages(MemorySegment g, long off, long imageStride, int img0, int images, int rows, int pixels, float[] dst) {
        long t0 = timing();
        int depth = images * pixels;
        int panels = panels(rows, MR);
        int blocks = (depth + DEPTH_BLOCK - 1) / DEPTH_BLOCK;
        Cpu.parallelFor(panels * blocks, 1, (from, to) -> {
            for (int item = from; item < to; item++) {
                int p = item / blocks;
                int d0 = (item - p * blocks) * DEPTH_BLOCK;
                int d1 = Math.min(depth, d0 + DEPTH_BLOCK);
                int base = p * depth * MR;
                for (int rr = 0; rr < MR; rr++) {
                    int co = p * MR + rr;
                    if (co >= rows) {
                        for (int d = d0; d < d1; d++) {
                            dst[base + d * MR + rr] = 0;
                        }
                        continue;
                    }
                    int d = d0;
                    while (d < d1) {
                        int img = d / pixels;
                        int pix = d - img * pixels;
                        int len = Math.min(d1 - d, pixels - pix);
                        long src = off + (long) (img0 + img) * imageStride + (long) co * pixels + pix;
                        for (int t = 0; t < len; t++) {
                            dst[base + (d + t) * MR + rr] = g.getAtIndex(F, src + t);
                        }
                        d += len;
                    }
                }
            }
        });
        packed(t0);
        return dst;
    }

    /** One cached packing of an operand that does not change: the source's version it was made from. */
    private record Packed(long version, float[] data) {
    }

    private static final Map<Object, Map<String, Packed>> CACHE = new WeakHashMap<>();

    /**
     * The packing {@code layout} of {@code owner} (a weight's storage) made at {@code version}, or
     * {@code make}'s result, kept until the owner is collected or its version changes.
     */
    public static float[] cached(Object owner, String layout, long version, java.util.function.Supplier<float[]> make) {
        synchronized (CACHE) {
            Packed p = CACHE.computeIfAbsent(owner, o -> new java.util.HashMap<>()).get(layout);
            if (p != null && p.version == version) {
                return p.data;
            }
        }
        float[] data = make.get();
        synchronized (CACHE) {
            CACHE.computeIfAbsent(owner, o -> new java.util.HashMap<>()).put(layout, new Packed(version, data));
        }
        return data;
    }

    private static final ThreadLocal<float[][]> SCRATCH = ThreadLocal.withInitial(() -> new float[2][0]);

    /** A scratch array of at least {@code size} floats for this thread, slot 0 or 1 (contents undefined). */
    public static float[] scratch(int slot, int size) {
        float[][] s = SCRATCH.get();
        if (s[slot].length < size) {
            s[slot] = new float[size];
        }
        return s[slot];
    }

    /**
     * C = A B (+ epilogue) for packed A ({@link #packA}) and B ({@link #packB}) of depth k: C's
     * element (i, j) is the float at index {@code cOff + i * ldc + j} of {@code c}; the bias (with
     * {@link #BIAS_ROW} or {@link #BIAS_COL}) is read at {@code biasOff + i} or {@code biasOff + j}.
     */
    public static void compute(float[] ap, float[] bp, int m, int n, int k, MemorySegment c, long cOff, long ldc, MemorySegment bias, long biasOff, int epilogue) {
        compute(ap, 0, bp, m, n, k, c, cOff, ldc, bias, biasOff, epilogue);
    }

    /**
     * {@link #compute} for the m rows of A whose packed panels start at index {@code apOff} of
     * {@code ap} (a multiple of {@code k * MR}: a slice of a larger packed A).
     */
    public static void compute(float[] ap, int apOff, float[] bp, int m, int n, int k, MemorySegment c, long cOff, long ldc, MemorySegment bias, long biasOff, int epilogue) {
        long t0 = timing();
        int pm = panels(m, MR);
        int pn = panels(n, NR);
        int mcp = Math.min(MC_PANELS, pm);
        int ncp = Math.min(NC_PANELS, pn);
        long macs = (long) m * n * Math.max(k, 1);
        int want = (int) Math.max(1, Math.min(2L * Cpu.threads(), macs / TILE_MIN_MACS));
        while (panels(pm, mcp) * panels(pn, ncp) < want && (mcp > 1 || ncp > 1)) {
            if (mcp >= ncp * 2 && mcp > 1 || ncp == 1) {
                mcp = (mcp + 1) / 2;
            } else {
                ncp = (ncp + 1) / 2;
            }
        }
        int tm = panels(pm, mcp);
        int tn = panels(pn, ncp);
        int fm = mcp;
        int fn = ncp;
        long work = 2L * m * n * Math.max(k, 1);
        int grain = work < (1 << 16) ? tm * tn : 1;
        Cpu.parallelFor(tm * tn, grain, (from, to) -> {
            float[] edge = new float[MR * NR];
            for (int t = from; t < to; t++) {
                int ti = t / tn;
                int tj = t % tn;
                tile(ap, apOff, bp, m, n, k, c, cOff, ldc, bias, biasOff, epilogue, ti * fm, Math.min(pm, (ti + 1) * fm), tj * fn, Math.min(pn, (tj + 1) * fn), edge);
            }
        });
        computed(t0);
    }

    private static void tile(float[] ap, int apOff, float[] bp, int m, int n, int k, MemorySegment c, long cOff, long ldc, MemorySegment bias, long biasOff, int epilogue, int ip0, int ip1,
            int jp0, int jp1, float[] edge) {
        boolean accumulate = (epilogue & ACCUMULATE) != 0;
        if (k == 0 && !accumulate) {
            for (int i = ip0 * MR; i < Math.min(m, ip1 * MR); i++) {
                long row = cOff + i * ldc;
                for (int j = jp0 * NR; j < Math.min(n, jp1 * NR); j++) {
                    c.setAtIndex(F, row + j, 0);
                }
            }
        }
        for (int pc = 0; pc < k; pc += KC) {
            int kc = Math.min(KC, k - pc);
            boolean first = pc == 0 && !accumulate;
            for (int jp = jp0; jp < jp1; jp++) {
                int bo = (jp * k + pc) * NR;
                for (int ip = ip0; ip < ip1; ip++) {
                    kernel(ap, apOff + (ip * k + pc) * MR, bp, bo, kc, c, cOff, ldc, ip * MR, jp * NR, m, n, first, edge);
                }
            }
        }
        if ((epilogue & ~ACCUMULATE) != 0) {
            epilogue(c, cOff, ldc, bias, biasOff, epilogue, ip0 * MR, Math.min(m, ip1 * MR), jp0 * NR, Math.min(n, jp1 * NR));
        }
    }

    /** MR x NR outputs at (i0, j0) over kc depths: stored on the first block, accumulated after. */
    static void kernel(float[] ap, int ao, float[] bp, int bo, int kc, MemorySegment c, long cOff, long ldc, int i0, int j0, int m, int n, boolean first, float[] edge) {
        FloatVector c00 = FloatVector.zero(S), c01 = FloatVector.zero(S);
        FloatVector c10 = FloatVector.zero(S), c11 = FloatVector.zero(S);
        FloatVector c20 = FloatVector.zero(S), c21 = FloatVector.zero(S);
        FloatVector c30 = FloatVector.zero(S), c31 = FloatVector.zero(S);
        FloatVector c40 = FloatVector.zero(S), c41 = FloatVector.zero(S);
        FloatVector c50 = FloatVector.zero(S), c51 = FloatVector.zero(S);
        for (int q = 0; q < kc; q++) {
            FloatVector b0 = FloatVector.fromArray(S, bp, bo);
            FloatVector b1 = FloatVector.fromArray(S, bp, bo + L);
            FloatVector a = FloatVector.broadcast(S, ap[ao]);
            c00 = a.fma(b0, c00);
            c01 = a.fma(b1, c01);
            a = FloatVector.broadcast(S, ap[ao + 1]);
            c10 = a.fma(b0, c10);
            c11 = a.fma(b1, c11);
            a = FloatVector.broadcast(S, ap[ao + 2]);
            c20 = a.fma(b0, c20);
            c21 = a.fma(b1, c21);
            a = FloatVector.broadcast(S, ap[ao + 3]);
            c30 = a.fma(b0, c30);
            c31 = a.fma(b1, c31);
            a = FloatVector.broadcast(S, ap[ao + 4]);
            c40 = a.fma(b0, c40);
            c41 = a.fma(b1, c41);
            a = FloatVector.broadcast(S, ap[ao + 5]);
            c50 = a.fma(b0, c50);
            c51 = a.fma(b1, c51);
            ao += MR;
            bo += NR;
        }
        if (i0 + MR <= m && j0 + NR <= n) {
            // written out: a helper taking the vectors is not inlined into this large method, and
            // the vectors it receives are boxed (8 times slower)
            long at = (cOff + i0 * ldc + j0) * 4;
            long row = ldc * 4;
            if (first) {
                c00.intoMemorySegment(c, at, ORDER);
                c01.intoMemorySegment(c, at + V, ORDER);
                c10.intoMemorySegment(c, at + row, ORDER);
                c11.intoMemorySegment(c, at + row + V, ORDER);
                c20.intoMemorySegment(c, at + 2 * row, ORDER);
                c21.intoMemorySegment(c, at + 2 * row + V, ORDER);
                c30.intoMemorySegment(c, at + 3 * row, ORDER);
                c31.intoMemorySegment(c, at + 3 * row + V, ORDER);
                c40.intoMemorySegment(c, at + 4 * row, ORDER);
                c41.intoMemorySegment(c, at + 4 * row + V, ORDER);
                c50.intoMemorySegment(c, at + 5 * row, ORDER);
                c51.intoMemorySegment(c, at + 5 * row + V, ORDER);
            } else {
                c00.add(FloatVector.fromMemorySegment(S, c, at, ORDER)).intoMemorySegment(c, at, ORDER);
                c01.add(FloatVector.fromMemorySegment(S, c, at + V, ORDER)).intoMemorySegment(c, at + V, ORDER);
                c10.add(FloatVector.fromMemorySegment(S, c, at + row, ORDER)).intoMemorySegment(c, at + row, ORDER);
                c11.add(FloatVector.fromMemorySegment(S, c, at + row + V, ORDER)).intoMemorySegment(c, at + row + V, ORDER);
                c20.add(FloatVector.fromMemorySegment(S, c, at + 2 * row, ORDER)).intoMemorySegment(c, at + 2 * row, ORDER);
                c21.add(FloatVector.fromMemorySegment(S, c, at + 2 * row + V, ORDER)).intoMemorySegment(c, at + 2 * row + V, ORDER);
                c30.add(FloatVector.fromMemorySegment(S, c, at + 3 * row, ORDER)).intoMemorySegment(c, at + 3 * row, ORDER);
                c31.add(FloatVector.fromMemorySegment(S, c, at + 3 * row + V, ORDER)).intoMemorySegment(c, at + 3 * row + V, ORDER);
                c40.add(FloatVector.fromMemorySegment(S, c, at + 4 * row, ORDER)).intoMemorySegment(c, at + 4 * row, ORDER);
                c41.add(FloatVector.fromMemorySegment(S, c, at + 4 * row + V, ORDER)).intoMemorySegment(c, at + 4 * row + V, ORDER);
                c50.add(FloatVector.fromMemorySegment(S, c, at + 5 * row, ORDER)).intoMemorySegment(c, at + 5 * row, ORDER);
                c51.add(FloatVector.fromMemorySegment(S, c, at + 5 * row + V, ORDER)).intoMemorySegment(c, at + 5 * row + V, ORDER);
            }
            return;
        }
        c00.intoArray(edge, 0);
        c01.intoArray(edge, L);
        c10.intoArray(edge, NR);
        c11.intoArray(edge, NR + L);
        c20.intoArray(edge, 2 * NR);
        c21.intoArray(edge, 2 * NR + L);
        c30.intoArray(edge, 3 * NR);
        c31.intoArray(edge, 3 * NR + L);
        c40.intoArray(edge, 4 * NR);
        c41.intoArray(edge, 4 * NR + L);
        c50.intoArray(edge, 5 * NR);
        c51.intoArray(edge, 5 * NR + L);
        int rows = Math.min(MR, m - i0);
        int cols = Math.min(NR, n - j0);
        for (int r = 0; r < rows; r++) {
            long base = cOff + (i0 + r) * ldc + j0;
            for (int j = 0; j < cols; j++) {
                float v = edge[r * NR + j];
                c.setAtIndex(F, base + j, first ? v : c.getAtIndex(F, base + j) + v);
            }
        }
    }

    /**
     * Applies {@code epilogue} (bias, ReLU, GELU: not {@link #ACCUMULATE}) to the m x n matrix C at
     * {@code cOff} with leading dimension {@code ldc}, in parallel over rows: after a product
     * another library computed.
     */
    public static void epilogue(MemorySegment c, long cOff, long ldc, int m, int n, MemorySegment bias, long biasOff, int epilogue) {
        if ((epilogue & ~ACCUMULATE) == 0) {
            return;
        }
        Cpu.parallelFor(m, Math.max(1, (1 << 15) / Math.max(1, n)), (from, to) -> epilogue(c, cOff, ldc, bias, biasOff, epilogue, from, to, 0, n));
    }

    private static final float INV_SQRT2 = (float) (1 / Math.sqrt(2));

    private static void epilogue(MemorySegment c, long cOff, long ldc, MemorySegment bias, long biasOff, int epilogue, int i0, int i1, int j0, int j1) {
        boolean rowBias = (epilogue & BIAS_ROW) != 0;
        boolean colBias = (epilogue & BIAS_COL) != 0;
        boolean relu = (epilogue & RELU) != 0;
        boolean gelu = (epilogue & GELU) != 0;
        int vEnd = j0 + S.loopBound(j1 - j0);
        for (int i = i0; i < i1; i++) {
            long row = cOff + i * ldc;
            float rb = rowBias ? bias.getAtIndex(F, biasOff + i) : 0;
            FloatVector vrb = FloatVector.broadcast(S, rb);
            int j = j0;
            for (; j < vEnd; j += L) {
                FloatVector v = FloatVector.fromMemorySegment(S, c, (row + j) * 4, ORDER);
                if (rowBias) {
                    v = v.add(vrb);
                }
                if (colBias) {
                    v = v.add(FloatVector.fromMemorySegment(S, bias, (biasOff + j) * 4, ORDER));
                }
                if (relu) {
                    v = v.lanewise(VectorOperators.MAX, 0f);
                }
                if (gelu) {
                    v = Elementwise.gelu(v);
                }
                v.intoMemorySegment(c, (row + j) * 4, ORDER);
            }
            for (; j < j1; j++) {
                float v = c.getAtIndex(F, row + j) + rb + (colBias ? bias.getAtIndex(F, biasOff + j) : 0);
                if (relu) {
                    v = Math.max(v, 0);
                }
                if (gelu) {
                    v = 0.5f * v * (1 + ai.beestream.jitnn.kernels.ElementwiseKernels.erf(v * INV_SQRT2));
                }
                c.setAtIndex(F, row + j, v);
            }
        }
    }

    /** C = op(A) op(B) (+ epilogue) from unpacked operands: packs both into this thread's scratch, then {@link #compute}. */
    public static void multiply(Matrix a, Matrix b, int m, int n, int k, MemorySegment c, long cOff, long ldc, MemorySegment bias, long biasOff, int epilogue) {
        float[] ap = packA(a, m, k, scratch(0, packedSize(m, k, MR)));
        float[] bp = packB(b, n, k, scratch(1, packedSize(n, k, NR)));
        compute(ap, bp, m, n, k, c, cOff, ldc, bias, biasOff, epilogue);
    }
}
