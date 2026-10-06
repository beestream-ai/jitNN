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
 * c0). Everything is deterministic: each thread owns its outputs.
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
}
