package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Recurrent steps of LSTM and GRU layers, PyTorch's gate layout and equations. The input
 * projection xg = x W_ih^T + b_ih of every timestep is computed beforehand by one GEMM; each
 * kernel here is one timestep t, one thread per (batch b, hidden unit u), and adds the recurrent
 * term h_{t-1} W_hh^T + b_hh itself. Sequences are [T, B, ...] (time first); h_{-1} = h0 and
 * c_{-1} = c0.
 *
 * <p>The backward kernels run from t = T - 1 down to 0 and carry the recurrent gradient through
 * the gate gradients of step t + 1; the {@code *Initial} kernels give the gradients of h0 (and
 * c0). The tiled path computes each step's recurrent product with {@link #backProduct} as
 * BACK_SPLIT partial sums, which the {@code *Cell} kernels add in a fixed order. Everything is
 * deterministic: each thread owns its outputs.
 *
 * <p>{@code lengths[b]} masks padded sequences: steps t >= lengths[b] output 0 (PyTorch's padded
 * output), keep the cell state, and pass no gradient.
 */
public final class RnnKernels {

    private RnnKernels() {
    }

    /**
     * LSTM step t: gates i, f, g, o (rows q H .. q H + H - 1 of W_hh for gate q), c = f c_{t-1} +
     * i g, h = o tanh(c). Writes y[t] = h, cs[t] = c and the activations gates[t] = [i, f, g, o].
     */
    public static void lstmForward(KernelContext kc, FloatArray xg, FloatArray whh, FloatArray bhh, FloatArray h0, FloatArray c0, FloatArray y, FloatArray cs, FloatArray gates,
            IntArray lengths, int t, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            int g4 = 4 * h;
            int xrow = (t * b + bi) * g4;
            int prev = ((t - 1) * b + bi) * h;
            if (t >= lengths.get(bi)) {
                int out = (t * b + bi) * h + u;
                cs.set(out, cs.get(prev + u));
                y.set(out, 0.0f);
                gates.set(xrow + u, 0.0f);
                gates.set(xrow + h + u, 0.0f);
                gates.set(xrow + 2 * h + u, 0.0f);
                gates.set(xrow + 3 * h + u, 0.0f);
            } else {
                float a0 = xg.get(xrow + u) + bhh.get(u);
                float a1 = xg.get(xrow + h + u) + bhh.get(h + u);
                float a2 = xg.get(xrow + 2 * h + u) + bhh.get(2 * h + u);
                float a3 = xg.get(xrow + 3 * h + u) + bhh.get(3 * h + u);
                int w0 = u * h;
                int w1 = (h + u) * h;
                int w2 = (2 * h + u) * h;
                int w3 = (3 * h + u) * h;
                for (int k = 0; k < h; k++) {
                    float hp;
                    if (t == 0) {
                        hp = h0.get(bi * h + k);
                    } else {
                        hp = y.get(prev + k);
                    }
                    a0 += hp * whh.get(w0 + k);
                    a1 += hp * whh.get(w1 + k);
                    a2 += hp * whh.get(w2 + k);
                    a3 += hp * whh.get(w3 + k);
                }
                float i = ElementwiseKernels.sigmoid(a0);
                float f = ElementwiseKernels.sigmoid(a1);
                float g = ElementwiseKernels.tanh(a2);
                float o = ElementwiseKernels.sigmoid(a3);
                float cp;
                if (t == 0) {
                    cp = c0.get(bi * h + u);
                } else {
                    cp = cs.get(prev + u);
                }
                float c = f * cp + i * g;
                int out = (t * b + bi) * h + u;
                cs.set(out, c);
                y.set(out, o * ElementwiseKernels.tanh(c));
                gates.set(xrow + u, i);
                gates.set(xrow + h + u, f);
                gates.set(xrow + 2 * h + u, g);
                gates.set(xrow + 3 * h + u, o);
            }
        }
    }

    /**
     * LSTM backward step t: dh = dy[t] + dGates[t + 1] W_hh, dc = dh o (1 - tanh^2 c) + dc[t + 1]
     * f[t + 1]; writes dc[t] and the pre-activation gate gradients dGates[t].
     */
    public static void lstmBackward(KernelContext kc, FloatArray dy, FloatArray whh, FloatArray gates, FloatArray cs, FloatArray c0, FloatArray dGates, FloatArray dc,
            IntArray lengths, int t, int steps, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            int g4 = 4 * h;
            int cell = (t * b + bi) * h + u;
            int row = (t * b + bi) * g4;
            if (t >= lengths.get(bi)) {
                dc.set(cell, 0.0f);
                dGates.set(row + u, 0.0f);
                dGates.set(row + h + u, 0.0f);
                dGates.set(row + 2 * h + u, 0.0f);
                dGates.set(row + 3 * h + u, 0.0f);
            } else {
                float dh = dy.get(cell);
                float dcv = 0.0f;
                if (t < steps - 1) {
                    int next = ((t + 1) * b + bi) * g4;
                    for (int j = 0; j < g4; j++) {
                        dh += dGates.get(next + j) * whh.get(j * h + u);
                    }
                    dcv = dc.get(cell + b * h) * gates.get(next + h + u);
                }
                float i = gates.get(row + u);
                float f = gates.get(row + h + u);
                float g = gates.get(row + 2 * h + u);
                float o = gates.get(row + 3 * h + u);
                float c = cs.get(cell);
                float tc = ElementwiseKernels.tanh(c);
                dcv += dh * o * (1.0f - tc * tc);
                dc.set(cell, dcv);
                float cp;
                if (t == 0) {
                    cp = c0.get(bi * h + u);
                } else {
                    cp = cs.get(cell - b * h);
                }
                dGates.set(row + u, dcv * g * i * (1.0f - i));
                dGates.set(row + h + u, dcv * cp * f * (1.0f - f));
                dGates.set(row + 2 * h + u, dcv * i * (1.0f - g * g));
                dGates.set(row + 3 * h + u, dh * tc * o * (1.0f - o));
            }
        }
    }

    /** Gradients of h0 and c0: dh0 = dGates[0] W_hh, dc0 = dc[0] f[0]. */
    public static void lstmInitial(KernelContext kc, FloatArray whh, FloatArray gates, FloatArray dGates, FloatArray dc, FloatArray dh0, FloatArray dc0, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            int g4 = 4 * h;
            float acc = 0.0f;
            for (int j = 0; j < g4; j++) {
                acc += dGates.get(bi * g4 + j) * whh.get(j * h + u);
            }
            dh0.set(idx, acc);
            dc0.set(idx, dc.get(idx) * gates.get(bi * g4 + h + u));
        }
    }

    /**
     * GRU step t: r, z from (xg + b_hh + h W_hh^T) in rows 0 and H, hn = b_hn + h W_hn^T,
     * n = tanh(xg_n + r hn), h = (1 - z) n + z h_{t-1}. Writes y[t] = h, gates[t] = [r, z, n] and
     * hn[t].
     */
    public static void gruForward(KernelContext kc, FloatArray xg, FloatArray whh, FloatArray bhh, FloatArray h0, FloatArray y, FloatArray gates, FloatArray hn,
            IntArray lengths, int t, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            int g3 = 3 * h;
            int xrow = (t * b + bi) * g3;
            int prev = ((t - 1) * b + bi) * h;
            if (t >= lengths.get(bi)) {
                int out = (t * b + bi) * h + u;
                y.set(out, 0.0f);
                hn.set(out, 0.0f);
                gates.set(xrow + u, 0.0f);
                gates.set(xrow + h + u, 0.0f);
                gates.set(xrow + 2 * h + u, 0.0f);
            } else {
                float ar = xg.get(xrow + u) + bhh.get(u);
                float az = xg.get(xrow + h + u) + bhh.get(h + u);
                float an = bhh.get(2 * h + u);
                int w0 = u * h;
                int w1 = (h + u) * h;
                int w2 = (2 * h + u) * h;
                float hpu = 0.0f;
                for (int k = 0; k < h; k++) {
                    float hp;
                    if (t == 0) {
                        hp = h0.get(bi * h + k);
                    } else {
                        hp = y.get(prev + k);
                    }
                    ar += hp * whh.get(w0 + k);
                    az += hp * whh.get(w1 + k);
                    an += hp * whh.get(w2 + k);
                    if (k == u) {
                        hpu = hp;
                    }
                }
                float r = ElementwiseKernels.sigmoid(ar);
                float z = ElementwiseKernels.sigmoid(az);
                float n = ElementwiseKernels.tanh(xg.get(xrow + 2 * h + u) + r * an);
                int out = (t * b + bi) * h + u;
                y.set(out, (1.0f - z) * n + z * hpu);
                gates.set(xrow + u, r);
                gates.set(xrow + h + u, z);
                gates.set(xrow + 2 * h + u, n);
                hn.set(out, an);
            }
        }
    }

    /**
     * GRU backward step t: dh = dy[t] + dHg[t + 1] W_hh + dh[t + 1] z[t + 1]; writes dh[t], the
     * input-side gate gradients dXg[t] = [dar, daz, dan] and the recurrent ones dHg[t] =
     * [dar, daz, dan r].
     */
    public static void gruBackward(KernelContext kc, FloatArray dy, FloatArray whh, FloatArray y, FloatArray h0, FloatArray gates, FloatArray hn, FloatArray dXg, FloatArray dHg,
            FloatArray dhs, IntArray lengths, int t, int steps, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            int g3 = 3 * h;
            int cell = (t * b + bi) * h + u;
            int row = (t * b + bi) * g3;
            if (t >= lengths.get(bi)) {
                dhs.set(cell, 0.0f);
                dXg.set(row + u, 0.0f);
                dXg.set(row + h + u, 0.0f);
                dXg.set(row + 2 * h + u, 0.0f);
                dHg.set(row + u, 0.0f);
                dHg.set(row + h + u, 0.0f);
                dHg.set(row + 2 * h + u, 0.0f);
            } else {
                float dh = dy.get(cell);
                if (t < steps - 1) {
                    int next = ((t + 1) * b + bi) * g3;
                    for (int j = 0; j < g3; j++) {
                        dh += dHg.get(next + j) * whh.get(j * h + u);
                    }
                    dh += dhs.get(cell + b * h) * gates.get(next + h + u);
                }
                dhs.set(cell, dh);
                float hp;
                if (t == 0) {
                    hp = h0.get(bi * h + u);
                } else {
                    hp = y.get(cell - b * h);
                }
                float r = gates.get(row + u);
                float z = gates.get(row + h + u);
                float n = gates.get(row + 2 * h + u);
                float dan = dh * (1.0f - z) * (1.0f - n * n);
                float dar = dan * hn.get(cell) * r * (1.0f - r);
                float daz = dh * (hp - n) * z * (1.0f - z);
                dXg.set(row + u, dar);
                dXg.set(row + h + u, daz);
                dXg.set(row + 2 * h + u, dan);
                dHg.set(row + u, dar);
                dHg.set(row + h + u, daz);
                dHg.set(row + 2 * h + u, dan * r);
            }
        }
    }

    /** Gradient of h0: dHg[0] W_hh + dh[0] z[0]. */
    public static void gruInitial(KernelContext kc, FloatArray whh, FloatArray gates, FloatArray dHg, FloatArray dhs, FloatArray dh0, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            int g3 = 3 * h;
            float acc = dhs.get(idx) * gates.get(bi * g3 + h + u);
            for (int j = 0; j < g3; j++) {
                acc += dHg.get(bi * g3 + j) * whh.get(j * h + u);
            }
            dh0.set(idx, acc);
        }
    }

    /** Batch rows per work-group of the tiled step kernels. */
    public static final int STEP_ROWS = 32;
    /** Hidden units per work-group (each with all of its gates). */
    public static final int STEP_UNITS = 8;
    /** Hidden-state columns staged in local memory per chunk. */
    public static final int STEP_K = 32;

    /**
     * LSTM step t as a tiled product: a work-group of STEP_ROWS x STEP_UNITS threads computes the
     * four gates of STEP_UNITS hidden units for STEP_ROWS sequences, staging STEP_K-wide chunks of
     * h_{t-1} (padded rows, no bank conflicts) and of the units' rows of W_hh in local memory with
     * coalesced loads, so each weight is read once per work-group instead of once per sequence. The
     * cell update follows in the same kernel; semantics as {@link #lstmForward}. Launch
     * ceil(b / STEP_ROWS) * ceil(h / STEP_UNITS) groups of STEP_ROWS * STEP_UNITS threads.
     */
    public static void lstmStep(KernelContext kc, FloatArray xg, FloatArray whh, FloatArray bhh, FloatArray h0, FloatArray c0, FloatArray y, FloatArray cs, FloatArray gates,
            IntArray lengths, int t, int b, int h) {
        float[] hs = kc.allocateFloatLocalArray(STEP_ROWS * (STEP_K + 1));
        float[] ws = kc.allocateFloatLocalArray(4 * STEP_UNITS * STEP_K);
        int tid = kc.localIdx;
        int tb = tid % STEP_ROWS;
        int tu = tid / STEP_ROWS;
        int unitTiles = (h + STEP_UNITS - 1) / STEP_UNITS;
        int b0 = (kc.groupIdx / unitTiles) * STEP_ROWS;
        int u0 = (kc.groupIdx % unitTiles) * STEP_UNITS;
        int threads = STEP_ROWS * STEP_UNITS;
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        for (int k0 = 0; k0 < h; k0 += STEP_K) {
            for (int idx = tid; idx < STEP_ROWS * STEP_K; idx += threads) {
                int rb = idx / STEP_K;
                int kk = idx - rb * STEP_K;
                int bi = b0 + rb;
                int kx = k0 + kk;
                float v = 0.0f;
                if ((bi < b) & (kx < h)) {
                    if (t == 0) {
                        v = h0.get(bi * h + kx);
                    } else {
                        v = y.get(((t - 1) * b + bi) * h + kx);
                    }
                }
                hs[rb * (STEP_K + 1) + kk] = v;
            }
            for (int idx = tid; idx < 4 * STEP_UNITS * STEP_K; idx += threads) {
                int r = idx / STEP_K;
                int kk = idx - r * STEP_K;
                int gate = r / STEP_UNITS;
                int uu = r - gate * STEP_UNITS;
                int u = u0 + uu;
                int kx = k0 + kk;
                float v = 0.0f;
                if ((u < h) & (kx < h)) {
                    v = whh.get((gate * h + u) * h + kx);
                }
                ws[r * STEP_K + kk] = v;
            }
            kc.localBarrier();
            for (int kk = 0; kk < STEP_K; kk++) {
                float hv = hs[tb * (STEP_K + 1) + kk];
                a0 += hv * ws[tu * STEP_K + kk];
                a1 += hv * ws[(STEP_UNITS + tu) * STEP_K + kk];
                a2 += hv * ws[(2 * STEP_UNITS + tu) * STEP_K + kk];
                a3 += hv * ws[(3 * STEP_UNITS + tu) * STEP_K + kk];
            }
            kc.localBarrier();
        }
        int bi = b0 + tb;
        int u = u0 + tu;
        if ((bi < b) & (u < h)) {
            int g4 = 4 * h;
            int xrow = (t * b + bi) * g4;
            int prev = ((t - 1) * b + bi) * h;
            int out = (t * b + bi) * h + u;
            if (t >= lengths.get(bi)) {
                cs.set(out, cs.get(prev + u));
                y.set(out, 0.0f);
                gates.set(xrow + u, 0.0f);
                gates.set(xrow + h + u, 0.0f);
                gates.set(xrow + 2 * h + u, 0.0f);
                gates.set(xrow + 3 * h + u, 0.0f);
            } else {
                float i = ElementwiseKernels.sigmoid(a0 + xg.get(xrow + u) + bhh.get(u));
                float f = ElementwiseKernels.sigmoid(a1 + xg.get(xrow + h + u) + bhh.get(h + u));
                float g = ElementwiseKernels.tanh(a2 + xg.get(xrow + 2 * h + u) + bhh.get(2 * h + u));
                float o = ElementwiseKernels.sigmoid(a3 + xg.get(xrow + 3 * h + u) + bhh.get(3 * h + u));
                float cp;
                if (t == 0) {
                    cp = c0.get(bi * h + u);
                } else {
                    cp = cs.get(prev + u);
                }
                float c = f * cp + i * g;
                cs.set(out, c);
                y.set(out, o * ElementwiseKernels.tanh(c));
                gates.set(xrow + u, i);
                gates.set(xrow + h + u, f);
                gates.set(xrow + 2 * h + u, g);
                gates.set(xrow + 3 * h + u, o);
            }
        }
    }

    /**
     * GRU step t as a tiled product, as {@link #lstmStep} with three gates; semantics as
     * {@link #gruForward}.
     */
    public static void gruStep(KernelContext kc, FloatArray xg, FloatArray whh, FloatArray bhh, FloatArray h0, FloatArray y, FloatArray gates, FloatArray hn,
            IntArray lengths, int t, int b, int h) {
        float[] hs = kc.allocateFloatLocalArray(STEP_ROWS * (STEP_K + 1));
        float[] ws = kc.allocateFloatLocalArray(3 * STEP_UNITS * STEP_K);
        int tid = kc.localIdx;
        int tb = tid % STEP_ROWS;
        int tu = tid / STEP_ROWS;
        int unitTiles = (h + STEP_UNITS - 1) / STEP_UNITS;
        int b0 = (kc.groupIdx / unitTiles) * STEP_ROWS;
        int u0 = (kc.groupIdx % unitTiles) * STEP_UNITS;
        int threads = STEP_ROWS * STEP_UNITS;
        float ar = 0.0f;
        float az = 0.0f;
        float an = 0.0f;
        for (int k0 = 0; k0 < h; k0 += STEP_K) {
            for (int idx = tid; idx < STEP_ROWS * STEP_K; idx += threads) {
                int rb = idx / STEP_K;
                int kk = idx - rb * STEP_K;
                int bi = b0 + rb;
                int kx = k0 + kk;
                float v = 0.0f;
                if ((bi < b) & (kx < h)) {
                    if (t == 0) {
                        v = h0.get(bi * h + kx);
                    } else {
                        v = y.get(((t - 1) * b + bi) * h + kx);
                    }
                }
                hs[rb * (STEP_K + 1) + kk] = v;
            }
            for (int idx = tid; idx < 3 * STEP_UNITS * STEP_K; idx += threads) {
                int r = idx / STEP_K;
                int kk = idx - r * STEP_K;
                int gate = r / STEP_UNITS;
                int uu = r - gate * STEP_UNITS;
                int u = u0 + uu;
                int kx = k0 + kk;
                float v = 0.0f;
                if ((u < h) & (kx < h)) {
                    v = whh.get((gate * h + u) * h + kx);
                }
                ws[r * STEP_K + kk] = v;
            }
            kc.localBarrier();
            for (int kk = 0; kk < STEP_K; kk++) {
                float hv = hs[tb * (STEP_K + 1) + kk];
                ar += hv * ws[tu * STEP_K + kk];
                az += hv * ws[(STEP_UNITS + tu) * STEP_K + kk];
                an += hv * ws[(2 * STEP_UNITS + tu) * STEP_K + kk];
            }
            kc.localBarrier();
        }
        int bi = b0 + tb;
        int u = u0 + tu;
        if ((bi < b) & (u < h)) {
            int g3 = 3 * h;
            int xrow = (t * b + bi) * g3;
            int out = (t * b + bi) * h + u;
            if (t >= lengths.get(bi)) {
                y.set(out, 0.0f);
                hn.set(out, 0.0f);
                gates.set(xrow + u, 0.0f);
                gates.set(xrow + h + u, 0.0f);
                gates.set(xrow + 2 * h + u, 0.0f);
            } else {
                float hpu;
                if (t == 0) {
                    hpu = h0.get(bi * h + u);
                } else {
                    hpu = y.get(((t - 1) * b + bi) * h + u);
                }
                float r = ElementwiseKernels.sigmoid(ar + xg.get(xrow + u) + bhh.get(u));
                float z = ElementwiseKernels.sigmoid(az + xg.get(xrow + h + u) + bhh.get(h + u));
                float hnv = an + bhh.get(2 * h + u);
                float n = ElementwiseKernels.tanh(xg.get(xrow + 2 * h + u) + r * hnv);
                y.set(out, (1.0f - z) * n + z * hpu);
                gates.set(xrow + u, r);
                gates.set(xrow + h + u, z);
                gates.set(xrow + 2 * h + u, n);
                hn.set(out, hnv);
            }
        }
    }

    /** Hidden units per work-group of {@link #backProduct}: two per thread. */
    public static final int BACK_UNITS = 2 * STEP_UNITS;
    /** Gate-gradient columns staged in local memory per chunk of {@link #backProduct}. */
    public static final int BACK_K = 64;
    /** Slices of the gate-gradient columns that {@link #backProduct} computes in separate work-groups. */
    public static final int BACK_SPLIT = 4;

    /**
     * The recurrent product of the backward steps, split along its inner dimension: for slice s of
     * the gw = G h gate-gradient columns, part[s] = grad[t + 1][:, slice] W_hh[slice, :] ([b, h]
     * each), with grad [T, b, gw] and W_hh [gw, h]; t = -1 gives the product for h0. Each
     * work-group computes STEP_ROWS sequences x BACK_UNITS units of one slice, staging BACK_K-wide
     * chunks of the gradients and of W_hh in local memory and loading the next chunk while it
     * computes the current one. Launch BACK_SPLIT * ceil(b / STEP_ROWS) * ceil(h / BACK_UNITS)
     * groups of STEP_ROWS * STEP_UNITS threads.
     */
    public static void backProduct(KernelContext kc, FloatArray grad, FloatArray whh, FloatArray part, int t, int b, int h, int gw) {
        float[] hs = kc.allocateFloatLocalArray(STEP_ROWS * (BACK_K + 1));
        float[] ws = kc.allocateFloatLocalArray(BACK_K * BACK_UNITS);
        int tid = kc.localIdx;
        int tb = tid % STEP_ROWS;
        int tu = tid / STEP_ROWS;
        int unitTiles = (h + BACK_UNITS - 1) / BACK_UNITS;
        int tiles = ((b + STEP_ROWS - 1) / STEP_ROWS) * unitTiles;
        int slice = kc.groupIdx / tiles;
        int tile = kc.groupIdx - slice * tiles;
        int b0 = (tile / unitTiles) * STEP_ROWS;
        int u0 = (tile % unitTiles) * BACK_UNITS;
        int width = ((gw + BACK_SPLIT - 1) / BACK_SPLIT + BACK_K - 1) / BACK_K * BACK_K;
        int kbeg = slice * width;
        int kend = kbeg + width;
        if (kend > gw) {
            kend = gw;
        }
        int next = (t + 1) * b * gw;
        // each thread loads STEP_ROWS * BACK_K / threads = 8 gradients and BACK_K * BACK_UNITS / threads = 4
        // weights per chunk, all issued together
        int threads = STEP_ROWS * STEP_UNITS;
        int hk = tid % BACK_K;
        int hstep = threads / BACK_K;
        int hb = b0 + tid / BACK_K;
        int wu = tid % BACK_UNITS;
        int wk = tid / BACK_UNITS;
        int wstep = threads / BACK_UNITS;
        int wn = u0 + wu;
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f, p4 = 0.0f, p5 = 0.0f, p6 = 0.0f, p7 = 0.0f;
        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
        if (kbeg < kend) {
            p0 = 0.0f;
            if ((hb < b) & (kbeg + hk < kend)) {
                p0 = grad.get(next + (hb) * gw + kbeg + hk);
            }
            p1 = 0.0f;
            if ((hb + 1 * hstep < b) & (kbeg + hk < kend)) {
                p1 = grad.get(next + (hb + 1 * hstep) * gw + kbeg + hk);
            }
            p2 = 0.0f;
            if ((hb + 2 * hstep < b) & (kbeg + hk < kend)) {
                p2 = grad.get(next + (hb + 2 * hstep) * gw + kbeg + hk);
            }
            p3 = 0.0f;
            if ((hb + 3 * hstep < b) & (kbeg + hk < kend)) {
                p3 = grad.get(next + (hb + 3 * hstep) * gw + kbeg + hk);
            }
            p4 = 0.0f;
            if ((hb + 4 * hstep < b) & (kbeg + hk < kend)) {
                p4 = grad.get(next + (hb + 4 * hstep) * gw + kbeg + hk);
            }
            p5 = 0.0f;
            if ((hb + 5 * hstep < b) & (kbeg + hk < kend)) {
                p5 = grad.get(next + (hb + 5 * hstep) * gw + kbeg + hk);
            }
            p6 = 0.0f;
            if ((hb + 6 * hstep < b) & (kbeg + hk < kend)) {
                p6 = grad.get(next + (hb + 6 * hstep) * gw + kbeg + hk);
            }
            p7 = 0.0f;
            if ((hb + 7 * hstep < b) & (kbeg + hk < kend)) {
                p7 = grad.get(next + (hb + 7 * hstep) * gw + kbeg + hk);
            }
            q0 = 0.0f;
            if ((wn < h) & (kbeg + wk < kend)) {
                q0 = whh.get((kbeg + wk) * h + wn);
            }
            q1 = 0.0f;
            if ((wn < h) & (kbeg + wk + 1 * wstep < kend)) {
                q1 = whh.get((kbeg + wk + 1 * wstep) * h + wn);
            }
            q2 = 0.0f;
            if ((wn < h) & (kbeg + wk + 2 * wstep < kend)) {
                q2 = whh.get((kbeg + wk + 2 * wstep) * h + wn);
            }
            q3 = 0.0f;
            if ((wn < h) & (kbeg + wk + 3 * wstep < kend)) {
                q3 = whh.get((kbeg + wk + 3 * wstep) * h + wn);
            }
        }
        float acc = 0.0f;
        float acc2 = 0.0f;
        for (int k0 = kbeg; k0 < kend; k0 += BACK_K) {
            hs[(tid / BACK_K) * (BACK_K + 1) + hk] = p0;
            hs[(tid / BACK_K + 1 * hstep) * (BACK_K + 1) + hk] = p1;
            hs[(tid / BACK_K + 2 * hstep) * (BACK_K + 1) + hk] = p2;
            hs[(tid / BACK_K + 3 * hstep) * (BACK_K + 1) + hk] = p3;
            hs[(tid / BACK_K + 4 * hstep) * (BACK_K + 1) + hk] = p4;
            hs[(tid / BACK_K + 5 * hstep) * (BACK_K + 1) + hk] = p5;
            hs[(tid / BACK_K + 6 * hstep) * (BACK_K + 1) + hk] = p6;
            hs[(tid / BACK_K + 7 * hstep) * (BACK_K + 1) + hk] = p7;
            ws[(wk) * BACK_UNITS + wu] = q0;
            ws[(wk + 1 * wstep) * BACK_UNITS + wu] = q1;
            ws[(wk + 2 * wstep) * BACK_UNITS + wu] = q2;
            ws[(wk + 3 * wstep) * BACK_UNITS + wu] = q3;
            kc.localBarrier();
            int k1 = k0 + BACK_K;
            if (k1 < kend) {
                p0 = 0.0f;
                if ((hb < b) & (k1 + hk < kend)) {
                    p0 = grad.get(next + (hb) * gw + k1 + hk);
                }
                p1 = 0.0f;
                if ((hb + 1 * hstep < b) & (k1 + hk < kend)) {
                    p1 = grad.get(next + (hb + 1 * hstep) * gw + k1 + hk);
                }
                p2 = 0.0f;
                if ((hb + 2 * hstep < b) & (k1 + hk < kend)) {
                    p2 = grad.get(next + (hb + 2 * hstep) * gw + k1 + hk);
                }
                p3 = 0.0f;
                if ((hb + 3 * hstep < b) & (k1 + hk < kend)) {
                    p3 = grad.get(next + (hb + 3 * hstep) * gw + k1 + hk);
                }
                p4 = 0.0f;
                if ((hb + 4 * hstep < b) & (k1 + hk < kend)) {
                    p4 = grad.get(next + (hb + 4 * hstep) * gw + k1 + hk);
                }
                p5 = 0.0f;
                if ((hb + 5 * hstep < b) & (k1 + hk < kend)) {
                    p5 = grad.get(next + (hb + 5 * hstep) * gw + k1 + hk);
                }
                p6 = 0.0f;
                if ((hb + 6 * hstep < b) & (k1 + hk < kend)) {
                    p6 = grad.get(next + (hb + 6 * hstep) * gw + k1 + hk);
                }
                p7 = 0.0f;
                if ((hb + 7 * hstep < b) & (k1 + hk < kend)) {
                    p7 = grad.get(next + (hb + 7 * hstep) * gw + k1 + hk);
                }
                q0 = 0.0f;
                if ((wn < h) & (k1 + wk < kend)) {
                    q0 = whh.get((k1 + wk) * h + wn);
                }
                q1 = 0.0f;
                if ((wn < h) & (k1 + wk + 1 * wstep < kend)) {
                    q1 = whh.get((k1 + wk + 1 * wstep) * h + wn);
                }
                q2 = 0.0f;
                if ((wn < h) & (k1 + wk + 2 * wstep < kend)) {
                    q2 = whh.get((k1 + wk + 2 * wstep) * h + wn);
                }
                q3 = 0.0f;
                if ((wn < h) & (k1 + wk + 3 * wstep < kend)) {
                    q3 = whh.get((k1 + wk + 3 * wstep) * h + wn);
                }
            }
            for (int kk = 0; kk < BACK_K; kk++) {
                float gv = hs[tb * (BACK_K + 1) + kk];
                acc += gv * ws[kk * BACK_UNITS + tu];
                acc2 += gv * ws[kk * BACK_UNITS + STEP_UNITS + tu];
            }
            kc.localBarrier();
        }
        int bi = b0 + tb;
        int u = u0 + tu;
        if (bi < b) {
            int out = (slice * b + bi) * h;
            if (u < h) {
                part.set(out + u, acc);
            }
            if (u + STEP_UNITS < h) {
                part.set(out + u + STEP_UNITS, acc2);
            }
        }
    }

    /**
     * LSTM backward step t after {@link #backProduct}: as {@link #lstmBackward} with the recurrent
     * product dGates[t + 1] W_hh read as the sum of the BACK_SPLIT partial products. One thread
     * per (b, u).
     */
    public static void lstmBackCell(KernelContext kc, FloatArray dy, FloatArray part, FloatArray gates, FloatArray cs, FloatArray c0, FloatArray dGates, FloatArray dc,
            IntArray lengths, int t, int steps, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            int g4 = 4 * h;
            int cell = (t * b + bi) * h + u;
            int row = (t * b + bi) * g4;
            if (t >= lengths.get(bi)) {
                dc.set(cell, 0.0f);
                dGates.set(row + u, 0.0f);
                dGates.set(row + h + u, 0.0f);
                dGates.set(row + 2 * h + u, 0.0f);
                dGates.set(row + 3 * h + u, 0.0f);
            } else {
                float dh = dy.get(cell);
                float dcv = 0.0f;
                if (t < steps - 1) {
                    for (int s = 0; s < BACK_SPLIT; s++) {
                        dh += part.get(s * b * h + idx);
                    }
                    dcv = dc.get(cell + b * h) * gates.get(row + b * g4 + h + u);
                }
                float i = gates.get(row + u);
                float f = gates.get(row + h + u);
                float g = gates.get(row + 2 * h + u);
                float o = gates.get(row + 3 * h + u);
                float tc = ElementwiseKernels.tanh(cs.get(cell));
                dcv += dh * o * (1.0f - tc * tc);
                dc.set(cell, dcv);
                float cp;
                if (t == 0) {
                    cp = c0.get(bi * h + u);
                } else {
                    cp = cs.get(cell - b * h);
                }
                dGates.set(row + u, dcv * g * i * (1.0f - i));
                dGates.set(row + h + u, dcv * cp * f * (1.0f - f));
                dGates.set(row + 2 * h + u, dcv * i * (1.0f - g * g));
                dGates.set(row + 3 * h + u, dh * tc * o * (1.0f - o));
            }
        }
    }

    /** Gradients of h0 and c0 after {@link #backProduct} with t = -1; as {@link #lstmInitial}. */
    public static void lstmInitialCell(KernelContext kc, FloatArray part, FloatArray gates, FloatArray dc, FloatArray dh0, FloatArray dc0, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            float acc = 0.0f;
            for (int s = 0; s < BACK_SPLIT; s++) {
                acc += part.get(s * b * h + idx);
            }
            dh0.set(idx, acc);
            dc0.set(idx, dc.get(idx) * gates.get(bi * 4 * h + h + u));
        }
    }

    /**
     * GRU backward step t after {@link #backProduct}: as {@link #gruBackward} with the recurrent
     * product dHg[t + 1] W_hh read as the sum of the BACK_SPLIT partial products. One thread per
     * (b, u).
     */
    public static void gruBackCell(KernelContext kc, FloatArray dy, FloatArray part, FloatArray y, FloatArray h0, FloatArray gates, FloatArray hn, FloatArray dXg, FloatArray dHg,
            FloatArray dhs, IntArray lengths, int t, int steps, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            int g3 = 3 * h;
            int cell = (t * b + bi) * h + u;
            int row = (t * b + bi) * g3;
            if (t >= lengths.get(bi)) {
                dhs.set(cell, 0.0f);
                dXg.set(row + u, 0.0f);
                dXg.set(row + h + u, 0.0f);
                dXg.set(row + 2 * h + u, 0.0f);
                dHg.set(row + u, 0.0f);
                dHg.set(row + h + u, 0.0f);
                dHg.set(row + 2 * h + u, 0.0f);
            } else {
                float dh = dy.get(cell);
                if (t < steps - 1) {
                    for (int s = 0; s < BACK_SPLIT; s++) {
                        dh += part.get(s * b * h + idx);
                    }
                    dh += dhs.get(cell + b * h) * gates.get(row + b * g3 + h + u);
                }
                dhs.set(cell, dh);
                float hp;
                if (t == 0) {
                    hp = h0.get(bi * h + u);
                } else {
                    hp = y.get(cell - b * h);
                }
                float r = gates.get(row + u);
                float z = gates.get(row + h + u);
                float n = gates.get(row + 2 * h + u);
                float dan = dh * (1.0f - z) * (1.0f - n * n);
                float dar = dan * hn.get(cell) * r * (1.0f - r);
                float daz = dh * (hp - n) * z * (1.0f - z);
                dXg.set(row + u, dar);
                dXg.set(row + h + u, daz);
                dXg.set(row + 2 * h + u, dan);
                dHg.set(row + u, dar);
                dHg.set(row + h + u, daz);
                dHg.set(row + 2 * h + u, dan * r);
            }
        }
    }

    /** Gradient of h0 after {@link #backProduct} with t = -1; as {@link #gruInitial}. */
    public static void gruInitialCell(KernelContext kc, FloatArray part, FloatArray gates, FloatArray dhs, FloatArray dh0, int b, int h) {
        int idx = kc.globalIdx;
        if (idx < b * h) {
            int bi = idx / h;
            int u = idx % h;
            float acc = dhs.get(idx) * gates.get(bi * 3 * h + h + u);
            for (int s = 0; s < BACK_SPLIT; s++) {
                acc += part.get(s * b * h + idx);
            }
            dh0.set(idx, acc);
        }
    }
}
