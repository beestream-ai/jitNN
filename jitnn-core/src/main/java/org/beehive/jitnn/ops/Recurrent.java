package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.RnnKernels;

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * The recurrence of one LSTM or GRU layer over a whole sequence, PyTorch's equations and gate
 * order (LSTM i, f, g, o; GRU r, z, n). The input projection xg = x W_ih^T + b_ih [T, B, G·H] is
 * an ordinary GEMM before it; this op adds h W_hh^T + b_hh and runs the cells, one kernel per
 * timestep (see {@link RnnKernels}). Its backward runs one kernel per timestep in reverse and gets
 * dW_hh from a single GEMM over the whole sequence.
 *
 * <p>Gradients flow from the output sequence y (and so from h_n, a slice of it); the final cell
 * state c_n is returned without a gradient. INT32 {@code lengths} [B] masks padded sequences
 * (steps past a sequence's length output 0 and keep its state). iattr = [T, B, H].
 */
public final class Recurrent {

    static final LstmOp LSTM = new LstmOp();
    static final LstmBackward LSTM_BACKWARD = new LstmBackward();
    static final GruOp GRU = new GruOp();
    static final GruBackward GRU_BACKWARD = new GruBackward();

    private static final int T = 0, B = 1, H = 2;

    private Recurrent() {
    }

    /**
     * LSTM layer: xg [T, B, 4H], whh [4H, H], bhh [4H], h0 and c0 [B, H]. Returns {y [T, B, H],
     * c_n [B, H]} (c_n without gradient).
     */
    public static Tensor[] lstm(Tensor xg, Tensor whh, Tensor bhh, Tensor h0, Tensor c0) {
        return lstm(xg, whh, bhh, h0, c0, null);
    }

    /** As {@link #lstm(Tensor, Tensor, Tensor, Tensor, Tensor)} for padded sequences of {@code lengths} (null: all full); c_n is each sequence's last state. */
    public static Tensor[] lstm(Tensor xg, Tensor whh, Tensor bhh, Tensor h0, Tensor c0, Tensor lengths) {
        int[] a = check(xg, whh, bhh, h0, 4);
        Tensor len = lengthsOf(lengths, a);
        Checks.f32(c0);
        if (!c0.shape().equals(h0.shape())) {
            throw new IllegalArgumentException("lstm: c0 " + c0.shape() + " vs h0 " + h0.shape());
        }
        Tensor y = Tensor.result(Shape.of(a[T], a[B], a[H]), DType.F32);
        Tensor cs = Tensor.result(y.shape(), DType.F32);
        Tensor gates = Tensor.result(xg.shape(), DType.F32);
        Tensor.record(LSTM, new Tensor[] { xg, whh, bhh, h0, c0, len }, new Tensor[] { y, cs, gates }, a, null, true);
        return new Tensor[] { y, cs.detach().narrow(0, a[T] - 1, 1).reshape(a[B], a[H]) };
    }

    /** GRU layer: xg [T, B, 3H], whh [3H, H], bhh [3H], h0 [B, H]. Returns y [T, B, H]. */
    public static Tensor gru(Tensor xg, Tensor whh, Tensor bhh, Tensor h0) {
        return gru(xg, whh, bhh, h0, null);
    }

    /** As {@link #gru(Tensor, Tensor, Tensor, Tensor)} for padded sequences of {@code lengths} (null: all full). */
    public static Tensor gru(Tensor xg, Tensor whh, Tensor bhh, Tensor h0, Tensor lengths) {
        int[] a = check(xg, whh, bhh, h0, 3);
        Tensor len = lengthsOf(lengths, a);
        Tensor y = Tensor.result(Shape.of(a[T], a[B], a[H]), DType.F32);
        Tensor gates = Tensor.result(xg.shape(), DType.F32);
        Tensor hn = Tensor.result(y.shape(), DType.F32);
        Tensor.record(GRU, new Tensor[] { xg, whh, bhh, h0, len }, new Tensor[] { y, gates, hn }, a, null, true);
        return y;
    }

    private static int[] check(Tensor xg, Tensor whh, Tensor bhh, Tensor h0, int gates) {
        Checks.f32(xg, whh, bhh, h0);
        Checks.rank(xg, 3);
        Checks.rank(h0, 2);
        int h = h0.dim(1);
        int b = h0.dim(0);
        if (xg.dim(1) != b || xg.dim(2) != gates * h || whh.dim(0) != gates * h || whh.dim(1) != h || bhh.numel() != gates * h) {
            throw new IllegalArgumentException("recurrent layer: xg " + xg.shape() + ", W_hh " + whh.shape() + ", b_hh " + bhh.shape() + " and h0 " + h0.shape() + " do not match");
        }
        return new int[] { xg.dim(0), b, h };
    }

    /** INT32 [B] sequence lengths in 1 .. T; all T when null. */
    private static Tensor lengthsOf(Tensor lengths, int[] a) {
        if (lengths == null) {
            int[] full = new int[a[B]];
            java.util.Arrays.fill(full, a[T]);
            return Tensor.of(full, a[B]);
        }
        if (lengths.dtype() != DType.I32 || lengths.numel() != a[B]) {
            throw new IllegalArgumentException("sequence lengths must be INT32 [" + a[B] + "], got " + lengths.shape() + " " + lengths.dtype());
        }
        return lengths;
    }

    private static int[] hostLengths(Node n, int index) {
        return ((Tensor) n.in(index)).toIntArray();
    }

    private static WorkerGrid cells(int[] a) {
        return Emitter.grid1D(a[B] * a[H]);
    }

    /** The tiled step kernels unless {@code -Djitnn.rnn=simple} (the thread-per-cell kernels). */
    static final boolean TILED = !"simple".equals(System.getProperty("jitnn.rnn"));

    /** Work-groups of the tiled step kernels: batch tiles x unit tiles. */
    private static WorkerGrid steps(int[] a) {
        int groups = Emitter.ceilDiv(a[B], RnnKernels.STEP_ROWS) * Emitter.ceilDiv(a[H], RnnKernels.STEP_UNITS);
        return Emitter.blocks1D(groups, RnnKernels.STEP_ROWS * RnnKernels.STEP_UNITS);
    }

    /** Work-groups of the split recurrent product of the backward steps. */
    private static WorkerGrid backProduct(int[] a) {
        int groups = RnnKernels.BACK_SPLIT * Emitter.ceilDiv(a[B], RnnKernels.STEP_ROWS) * Emitter.ceilDiv(a[H], RnnKernels.BACK_UNITS);
        return Emitter.blocks1D(groups, RnnKernels.STEP_ROWS * RnnKernels.STEP_UNITS);
    }

    /** The previous hidden states [h0, y_0 .. y_{T-2}] as [T B, H], for dW_hh. */
    private static Tensor previous(Tensor h0, Tensor y, int[] a) {
        Tensor first = h0.reshape(1, a[B], a[H]);
        Tensor prev = a[T] == 1 ? first : Tensor.cat(0, first, y.narrow(0, 0, a[T] - 1));
        return prev.reshape(a[T] * a[B], a[H]);
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static double[] host(Tensor t) {
        FloatArray f = (FloatArray) t.storage().host();
        double[] d = new double[t.numel()];
        for (int i = 0; i < d.length; i++) {
            d[i] = f.get(i);
        }
        return d;
    }

    private static void store(Tensor t, double[] v) {
        FloatArray f = (FloatArray) t.storage().host();
        for (int i = 0; i < v.length; i++) {
            f.set(i, (float) v[i]);
        }
    }

    /** sum_k h[k] W[row, k] for row-major W [*, h]. */
    private static double dot(double[] hv, int hOff, double[] w, int row, int h) {
        double acc = 0;
        for (int k = 0; k < h; k++) {
            acc += hv[hOff + k] * w[row * h + k];
        }
        return acc;
    }

    static final class LstmOp implements Op {

        @Override
        public String name() {
            return "lstm";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int[] a = n.iattr;
            FloatArray xg = e.f32(n.in(0));
            FloatArray whh = e.f32(n.in(1));
            FloatArray bhh = e.f32(n.in(2));
            FloatArray h0 = e.f32(n.in(3));
            FloatArray c0 = e.f32(n.in(4));
            FloatArray y = e.f32(n.out(0));
            FloatArray cs = e.f32(n.out(1));
            FloatArray gates = e.f32(n.out(2));
            IntArray len = e.i32(n.in(5));
            int b = a[B], h = a[H];
            for (int t = 0; t < a[T]; t++) {
                int step = t;
                if (TILED) {
                    e.task(steps(a), (g, id) -> g.task(id, RnnKernels::lstmStep, e.context(), xg, whh, bhh, h0, c0, y, cs, gates, len, step, b, h));
                } else {
                    e.task(cells(a), (g, id) -> g.task(id, RnnKernels::lstmForward, e.context(), xg, whh, bhh, h0, c0, y, cs, gates, len, step, b, h));
                }
            }
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            int steps = a[T], b = a[B], h = a[H];
            double[] xg = host(n.in(0));
            double[] w = host(n.in(1));
            double[] bias = host(n.in(2));
            double[] hp = host(n.in(3));
            double[] cp = host(n.in(4));
            double[] y = new double[steps * b * h];
            double[] cs = new double[y.length];
            double[] gates = new double[xg.length];
            int[] len = hostLengths(n, 5);
            for (int t = 0; t < steps; t++) {
                double[] hn = new double[b * h];
                double[] cn = new double[b * h];
                for (int bi = 0; bi < b; bi++) {
                    for (int u = 0; u < h; u++) {
                        if (t >= len[bi]) {
                            hn[bi * h + u] = hp[bi * h + u];
                            cn[bi * h + u] = cp[bi * h + u];
                            continue;
                        }
                        double[] act = new double[4];
                        for (int q = 0; q < 4; q++) {
                            act[q] = xg[(t * b + bi) * 4 * h + q * h + u] + bias[q * h + u] + dot(hp, bi * h, w, q * h + u, h);
                        }
                        double i = sigmoid(act[0]);
                        double f = sigmoid(act[1]);
                        double g = Math.tanh(act[2]);
                        double o = sigmoid(act[3]);
                        double c = f * cp[bi * h + u] + i * g;
                        cn[bi * h + u] = c;
                        hn[bi * h + u] = o * Math.tanh(c);
                        int row = (t * b + bi) * 4 * h;
                        gates[row + u] = i;
                        gates[row + h + u] = f;
                        gates[row + 2 * h + u] = g;
                        gates[row + 3 * h + u] = o;
                    }
                }
                for (int bi = 0; bi < b; bi++) {
                    for (int u = 0; u < h; u++) {
                        y[(t * b + bi) * h + u] = t >= len[bi] ? 0 : hn[bi * h + u];
                    }
                }
                System.arraycopy(cn, 0, cs, t * b * h, b * h);
                hp = hn;
                cp = cn;
            }
            store(n.out(0), y);
            store(n.out(1), cs);
            store(n.out(2), gates);
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            int[] a = n.iattr;
            Tensor xg = n.in(0);
            Tensor whh = n.in(1);
            Tensor dGates = Tensor.result(xg.shape(), DType.F32);
            Tensor dh0 = Tensor.result(n.in(3).shape(), DType.F32);
            Tensor dc0 = Tensor.result(n.in(4).shape(), DType.F32);
            Tensor.record(LSTM_BACKWARD, new Tensor[] { gy, whh, n.out(2), n.out(1), n.in(4), n.in(5) }, new Tensor[] { dGates, dh0, dc0 }, a.clone(), null, true);
            Tensor flat = dGates.reshape(a[T] * a[B], 4 * a[H]);
            Tensor dW = whh.requiresGrad() ? flat.matmul(previous(n.in(3), n.out(0), a), true, false) : null;
            Tensor db = n.in(2).requiresGrad() ? flat.sum(0, false) : null;
            return new Tensor[] { xg.requiresGrad() ? dGates : null, dW, db, n.in(3).requiresGrad() ? dh0 : null, n.in(4).requiresGrad() ? dc0 : null, null };
        }
    }

    /** inputs [dy, W_hh, gates, cell states, c0, lengths]; outputs [dGates, dh0, dc0]. */
    static final class LstmBackward implements Op {

        @Override
        public String name() {
            return "lstm_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int[] a = n.iattr;
            FloatArray dy = e.f32(n.in(0));
            FloatArray whh = e.f32(n.in(1));
            FloatArray gates = e.f32(n.in(2));
            FloatArray cs = e.f32(n.in(3));
            FloatArray c0 = e.f32(n.in(4));
            FloatArray dGates = e.f32(n.out(0));
            FloatArray dh0 = e.f32(n.out(1));
            FloatArray dc0 = e.f32(n.out(2));
            FloatArray dc = e.scratch(a[T] * a[B] * a[H]);
            IntArray len = e.i32(n.in(5));
            int steps = a[T], b = a[B], h = a[H];
            if (TILED) {
                FloatArray part = e.scratch(RnnKernels.BACK_SPLIT * b * h);
                int gw = 4 * h;
                for (int t = steps - 1; t >= 0; t--) {
                    int step = t;
                    if (t < steps - 1) {
                        e.task(backProduct(a), (g, id) -> g.task(id, RnnKernels::backProduct, e.context(), dGates, whh, part, step, b, h, gw));
                    }
                    e.task(cells(a), (g, id) -> g.task(id, RnnKernels::lstmBackCell, e.context(), dy, part, gates, cs, c0, dGates, dc, len, step, steps, b, h));
                }
                e.task(backProduct(a), (g, id) -> g.task(id, RnnKernels::backProduct, e.context(), dGates, whh, part, -1, b, h, gw));
                e.task(cells(a), (g, id) -> g.task(id, RnnKernels::lstmInitialCell, e.context(), part, gates, dc, dh0, dc0, b, h));
                return;
            }
            for (int t = steps - 1; t >= 0; t--) {
                int step = t;
                e.task(cells(a), (g, id) -> g.task(id, RnnKernels::lstmBackward, e.context(), dy, whh, gates, cs, c0, dGates, dc, len, step, steps, b, h));
            }
            e.task(cells(a), (g, id) -> g.task(id, RnnKernels::lstmInitial, e.context(), whh, gates, dGates, dc, dh0, dc0, b, h));
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            int steps = a[T], b = a[B], h = a[H];
            double[] dy = host(n.in(0));
            double[] w = host(n.in(1));
            double[] gates = host(n.in(2));
            double[] cs = host(n.in(3));
            double[] c0 = host(n.in(4));
            double[] dG = new double[gates.length];
            double[] dhNext = new double[b * h];
            double[] dcNext = new double[b * h];
            int[] len = hostLengths(n, 5);
            for (int t = steps - 1; t >= 0; t--) {
                for (int bi = 0; bi < b; bi++) {
                    for (int u = 0; u < h; u++) {
                        int cell = (t * b + bi) * h + u;
                        int row = (t * b + bi) * 4 * h;
                        if (t >= len[bi]) {
                            dcNext[bi * h + u] = 0;
                            continue;
                        }
                        double i = gates[row + u], f = gates[row + h + u], g = gates[row + 2 * h + u], o = gates[row + 3 * h + u];
                        double dh = dy[cell] + dhNext[bi * h + u];
                        double tc = Math.tanh(cs[cell]);
                        double dc = dcNext[bi * h + u] + dh * o * (1 - tc * tc);
                        double cp = t == 0 ? c0[bi * h + u] : cs[cell - b * h];
                        dG[row + u] = dc * g * i * (1 - i);
                        dG[row + h + u] = dc * cp * f * (1 - f);
                        dG[row + 2 * h + u] = dc * i * (1 - g * g);
                        dG[row + 3 * h + u] = dh * tc * o * (1 - o);
                        dcNext[bi * h + u] = dc * f;
                    }
                }
                // dh_{t-1} = dGates[t] W_hh
                for (int bi = 0; bi < b; bi++) {
                    for (int u = 0; u < h; u++) {
                        double acc = 0;
                        for (int j = 0; j < 4 * h; j++) {
                            acc += dG[(t * b + bi) * 4 * h + j] * w[j * h + u];
                        }
                        dhNext[bi * h + u] = acc;
                    }
                }
            }
            store(n.out(0), dG);
            store(n.out(1), dhNext);
            store(n.out(2), dcNext);
        }
    }

    static final class GruOp implements Op {

        @Override
        public String name() {
            return "gru";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int[] a = n.iattr;
            FloatArray xg = e.f32(n.in(0));
            FloatArray whh = e.f32(n.in(1));
            FloatArray bhh = e.f32(n.in(2));
            FloatArray h0 = e.f32(n.in(3));
            FloatArray y = e.f32(n.out(0));
            FloatArray gates = e.f32(n.out(1));
            FloatArray hn = e.f32(n.out(2));
            IntArray len = e.i32(n.in(4));
            int b = a[B], h = a[H];
            for (int t = 0; t < a[T]; t++) {
                int step = t;
                if (TILED) {
                    e.task(steps(a), (g, id) -> g.task(id, RnnKernels::gruStep, e.context(), xg, whh, bhh, h0, y, gates, hn, len, step, b, h));
                } else {
                    e.task(cells(a), (g, id) -> g.task(id, RnnKernels::gruForward, e.context(), xg, whh, bhh, h0, y, gates, hn, len, step, b, h));
                }
            }
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            int steps = a[T], b = a[B], h = a[H];
            double[] xg = host(n.in(0));
            double[] w = host(n.in(1));
            double[] bias = host(n.in(2));
            double[] hp = host(n.in(3));
            double[] y = new double[steps * b * h];
            double[] gates = new double[xg.length];
            double[] hns = new double[y.length];
            int[] len = hostLengths(n, 4);
            for (int t = 0; t < steps; t++) {
                double[] next = new double[b * h];
                for (int bi = 0; bi < b; bi++) {
                    for (int u = 0; u < h; u++) {
                        int row = (t * b + bi) * 3 * h;
                        if (t >= len[bi]) {
                            next[bi * h + u] = hp[bi * h + u];
                            continue;
                        }
                        double r = sigmoid(xg[row + u] + bias[u] + dot(hp, bi * h, w, u, h));
                        double z = sigmoid(xg[row + h + u] + bias[h + u] + dot(hp, bi * h, w, h + u, h));
                        double hn = bias[2 * h + u] + dot(hp, bi * h, w, 2 * h + u, h);
                        double nv = Math.tanh(xg[row + 2 * h + u] + r * hn);
                        next[bi * h + u] = (1 - z) * nv + z * hp[bi * h + u];
                        gates[row + u] = r;
                        gates[row + h + u] = z;
                        gates[row + 2 * h + u] = nv;
                        hns[(t * b + bi) * h + u] = hn;
                    }
                }
                for (int bi = 0; bi < b; bi++) {
                    for (int u = 0; u < h; u++) {
                        y[(t * b + bi) * h + u] = t >= len[bi] ? 0 : next[bi * h + u];
                    }
                }
                hp = next;
            }
            store(n.out(0), y);
            store(n.out(1), gates);
            store(n.out(2), hns);
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            int[] a = n.iattr;
            Tensor xg = n.in(0);
            Tensor whh = n.in(1);
            Tensor dXg = Tensor.result(xg.shape(), DType.F32);
            Tensor dHg = Tensor.result(xg.shape(), DType.F32);
            Tensor dh0 = Tensor.result(n.in(3).shape(), DType.F32);
            Tensor.record(GRU_BACKWARD, new Tensor[] { gy, whh, n.out(0), n.in(3), n.out(1), n.out(2), n.in(4) }, new Tensor[] { dXg, dHg, dh0 }, a.clone(), null, true);
            Tensor flat = dHg.reshape(a[T] * a[B], 3 * a[H]);
            Tensor dW = whh.requiresGrad() ? flat.matmul(previous(n.in(3), n.out(0), a), true, false) : null;
            Tensor db = n.in(2).requiresGrad() ? flat.sum(0, false) : null;
            return new Tensor[] { xg.requiresGrad() ? dXg : null, dW, db, n.in(3).requiresGrad() ? dh0 : null, null };
        }
    }

    /** inputs [dy, W_hh, y, h0, gates, hn, lengths]; outputs [dXg, dHg, dh0]. */
    static final class GruBackward implements Op {

        @Override
        public String name() {
            return "gru_bwd";
        }

        @Override
        public void emit(Emitter e, Node n) {
            int[] a = n.iattr;
            FloatArray dy = e.f32(n.in(0));
            FloatArray whh = e.f32(n.in(1));
            FloatArray y = e.f32(n.in(2));
            FloatArray h0 = e.f32(n.in(3));
            FloatArray gates = e.f32(n.in(4));
            FloatArray hn = e.f32(n.in(5));
            FloatArray dXg = e.f32(n.out(0));
            FloatArray dHg = e.f32(n.out(1));
            FloatArray dh0 = e.f32(n.out(2));
            FloatArray dhs = e.scratch(a[T] * a[B] * a[H]);
            IntArray len = e.i32(n.in(6));
            int steps = a[T], b = a[B], h = a[H];
            if (TILED) {
                FloatArray part = e.scratch(RnnKernels.BACK_SPLIT * b * h);
                int gw = 3 * h;
                for (int t = steps - 1; t >= 0; t--) {
                    int step = t;
                    if (t < steps - 1) {
                        e.task(backProduct(a), (g, id) -> g.task(id, RnnKernels::backProduct, e.context(), dHg, whh, part, step, b, h, gw));
                    }
                    e.task(cells(a), (g, id) -> g.task(id, RnnKernels::gruBackCell, e.context(), dy, part, y, h0, gates, hn, dXg, dHg, dhs, len, step, steps, b, h));
                }
                e.task(backProduct(a), (g, id) -> g.task(id, RnnKernels::backProduct, e.context(), dHg, whh, part, -1, b, h, gw));
                e.task(cells(a), (g, id) -> g.task(id, RnnKernels::gruInitialCell, e.context(), part, gates, dhs, dh0, b, h));
                return;
            }
            for (int t = steps - 1; t >= 0; t--) {
                int step = t;
                e.task(cells(a), (g, id) -> g.task(id, RnnKernels::gruBackward, e.context(), dy, whh, y, h0, gates, hn, dXg, dHg, dhs, len, step, steps, b, h));
            }
            e.task(cells(a), (g, id) -> g.task(id, RnnKernels::gruInitial, e.context(), whh, gates, dHg, dhs, dh0, b, h));
        }

        @Override
        public void reference(Node n) {
            int[] a = n.iattr;
            int steps = a[T], b = a[B], h = a[H];
            double[] dy = host(n.in(0));
            double[] w = host(n.in(1));
            double[] y = host(n.in(2));
            double[] h0 = host(n.in(3));
            double[] gates = host(n.in(4));
            double[] hns = host(n.in(5));
            double[] dXg = new double[gates.length];
            double[] dHg = new double[gates.length];
            double[] dhNext = new double[b * h];
            int[] len = hostLengths(n, 6);
            for (int t = steps - 1; t >= 0; t--) {
                double[] dhPrev = new double[b * h];
                for (int bi = 0; bi < b; bi++) {
                    for (int u = 0; u < h; u++) {
                        int cell = (t * b + bi) * h + u;
                        int row = (t * b + bi) * 3 * h;
                        if (t >= len[bi]) {
                            continue;
                        }
                        double r = gates[row + u], z = gates[row + h + u], nv = gates[row + 2 * h + u];
                        double dh = dy[cell] + dhNext[bi * h + u];
                        double hp = t == 0 ? h0[bi * h + u] : y[cell - b * h];
                        double dan = dh * (1 - z) * (1 - nv * nv);
                        double dar = dan * hns[cell] * r * (1 - r);
                        double daz = dh * (hp - nv) * z * (1 - z);
                        dXg[row + u] = dar;
                        dXg[row + h + u] = daz;
                        dXg[row + 2 * h + u] = dan;
                        dHg[row + u] = dar;
                        dHg[row + h + u] = daz;
                        dHg[row + 2 * h + u] = dan * r;
                        dhPrev[bi * h + u] += dh * z;
                    }
                }
                for (int bi = 0; bi < b; bi++) {
                    for (int u = 0; u < h; u++) {
                        double acc = 0;
                        for (int j = 0; j < 3 * h; j++) {
                            acc += dHg[(t * b + bi) * 3 * h + j] * w[j * h + u];
                        }
                        dhPrev[bi * h + u] += acc;
                    }
                }
                dhNext = dhPrev;
            }
            store(n.out(0), dXg);
            store(n.out(1), dHg);
            store(n.out(2), dhNext);
        }
    }
}
