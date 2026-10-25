package org.beehive.jitnn.djl;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.recurrent.RNN;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.ops.OptimStep;

/**
 * DJL's internal operators (layers, pooling, optimiser updates) on jitNN. Layers use PyTorch's
 * layouts and equations, which DJL's blocks follow; optimisers run jitNN's fused in-place updates
 * on persistent parameters and states. Variants jitNN lacks (non-square windows, ceil mode, LP
 * pooling, image augmentation, multibox) throw {@link UnsupportedOperationException}.
 */
final class JitnnNDArrayEx implements NDArrayEx {

    /** Optimiser states DJL does not pass (jitNN's RMSprop keeps three), per weight. */
    private static final Map<Tensor, Tensor[]> EXTRA_STATES = new WeakHashMap<>();

    private final JitnnNDArray array;

    JitnnNDArrayEx(JitnnNDArray array) {
        this.array = array;
    }

    private Tensor t() {
        return array.tensor();
    }

    private Tensor of(NDArray a) {
        return a == null ? null : array.of(a);
    }

    private JitnnNDArray wrap(Tensor t) {
        return array.wrap(t, DataType.FLOAT32);
    }

    private static UnsupportedOperationException unsupported(String what) {
        return new UnsupportedOperationException("jitNN's DJL engine does not support " + what);
    }

    /** One size for every spatial axis of a DJL shape (jitNN windows are square). */
    private static int square(Shape s, String what) {
        if (s == null || s.dimension() == 0) {
            return 0;
        }
        long v = s.get(0);
        for (int i = 1; i < s.dimension(); i++) {
            if (s.get(i) != v) {
                throw unsupported(what + " " + s + " (jitNN takes the same size on every axis)");
            }
        }
        return (int) v;
    }

    @Override
    public NDArray rdivi(NDArray b) {
        return array.replace(wrap(of(b).div(t())));
    }

    @Override
    public NDArray rmodi(NDArray b) {
        return array.replace(wrap(of(b).remainder(t())));
    }

    @Override
    public NDArray rpowi(Number n) {
        return array.replace(wrap(Tensor.full(n.floatValue(), t().shape().dims()).log().mul(t()).exp()));
    }

    // ---------------------------------------------------------------- activations

    @Override
    public NDArray relu() {
        return wrap(t().relu());
    }

    @Override
    public NDArray sigmoid() {
        return wrap(t().sigmoid());
    }

    @Override
    public NDArray tanh() {
        return wrap(t().tanh());
    }

    @Override
    public NDArray softPlus() {
        return wrap(t().softplus());
    }

    @Override
    public NDArray softSign() {
        return wrap(t().softsign());
    }

    @Override
    public NDArray leakyRelu(float alpha) {
        return wrap(t().leakyRelu(alpha));
    }

    @Override
    public NDArray elu(float alpha) {
        return wrap(t().elu(alpha));
    }

    @Override
    public NDArray selu() {
        return wrap(t().selu());
    }

    @Override
    public NDArray gelu() {
        return wrap(t().gelu());
    }

    @Override
    public NDArray swish(float beta) {
        return beta == 1 ? wrap(t().silu()) : wrap(t().mul(t().mul(beta).sigmoid()));
    }

    @Override
    public NDArray mish() {
        return wrap(t().mish());
    }

    // ---------------------------------------------------------------- pooling

    @Override
    public NDArray maxPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return wrap(pool(F.PoolMode.MAX, kernelShape, stride, padding, ceilMode, true, 0));
    }

    @Override
    public NDArray avgPool(Shape kernelShape, Shape stride, Shape padding, boolean ceilMode, boolean countIncludePad) {
        return wrap(pool(F.PoolMode.AVG, kernelShape, stride, padding, ceilMode, countIncludePad, 0));
    }

    /** Pooling with a size per axis: the 2-D kernels when square and symmetric, else {@link F#pool}. */
    private Tensor pool(F.PoolMode mode, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode, boolean countIncludePad, float p) {
        int r = kernelShape.dimension();
        if (r < 1 || r > 3) {
            throw unsupported("pooling over " + r + " axes");
        }
        int[] k = axes(kernelShape, r, 1);
        int[] s = axes(stride, r, 1);
        int[] pd = axes(padding, r, 0);
        boolean square = uniform(k) && uniform(s) && uniform(pd);
        if (r == 2 && square && !ceilMode && mode != F.PoolMode.LP && (mode == F.PoolMode.MAX || countIncludePad || pd[0] == 0)) {
            return mode == F.PoolMode.MAX ? F.maxPool2d(t(), k[0], s[0], pd[0]) : F.avgPool2d(t(), k[0], s[0], pd[0]);
        }
        int[] pads = new int[2 * r];
        System.arraycopy(pd, 0, pads, 0, r);
        System.arraycopy(pd, 0, pads, r, r);
        int[] dil = new int[r];
        java.util.Arrays.fill(dil, 1);
        return F.pool(t(), mode, k, s, pads, dil, ceilMode, countIncludePad, p);
    }

    /** A shape as one value per axis ({@code otherwise} when absent; one value repeats). */
    private static int[] axes(Shape v, int r, int otherwise) {
        int[] a = new int[r];
        for (int i = 0; i < r; i++) {
            a[i] = v == null || v.dimension() == 0 ? otherwise : (int) v.get(v.dimension() == 1 ? 0 : i);
        }
        return a;
    }

    private static boolean uniform(int[] v) {
        for (int x : v) {
            if (x != v[0]) {
                return false;
            }
        }
        return true;
    }

    /** x [N, C, ...] as [N, C, spatial], reduced over the last axis to [N, C]. */
    private Tensor spatial() {
        return t().reshape(t().dim(0), t().dim(1), -1);
    }

    @Override
    public NDArray globalMaxPool() {
        return wrap(spatial().max(2, false));
    }

    @Override
    public NDArray globalAvgPool() {
        return wrap(spatial().mean(2, false));
    }

    @Override
    public NDArray lpPool(float normType, Shape kernelShape, Shape stride, Shape padding, boolean ceilMode) {
        return wrap(pool(F.PoolMode.LP, kernelShape, stride, padding, ceilMode, true, normType));
    }

    @Override
    public NDArray globalLpPool(float normType) {
        return wrap(spatial().abs().pow(normType).sum(2, false).pow(1 / normType));
    }

    // ---------------------------------------------------------------- optimisers

    /** A DJL array's tensor made persistent in place (optimiser state, running statistics). */
    private Tensor persistent(NDArray a) {
        return ((JitnnNDArray) a).makePersistent();
    }

    private Tensor grad(NDArray g, float rescaleGrad, float clipGrad) {
        Tensor t = of(g);
        if (rescaleGrad != 1) {
            t = t.mul(rescaleGrad);
        }
        if (clipGrad > 0) {
            t = t.clamp(-clipGrad, clipGrad);
        }
        return t;
    }

    private Tensor[] extraStates(Tensor weight, int count) {
        synchronized (EXTRA_STATES) {
            return EXTRA_STATES.computeIfAbsent(weight, w -> {
                Tensor[] s = new Tensor[count];
                for (int i = 0; i < count; i++) {
                    s[i] = Tensor.persistent(Tensor.zeros(w.shape().dims()));
                }
                return s;
            });
        }
    }

    private static void step(OptimStep.Kind kind, Tensor weight, Tensor grad, Tensor[] states, float... hyper) {
        List<List<Tensor>> st = new ArrayList<>();
        for (Tensor s : states) {
            st.add(List.of(s));
        }
        OptimStep.apply(kind, List.of(weight), List.of(grad), st, Tensor.of(hyper, hyper.length), null);
    }

    @Override
    public void adamUpdate(NDList inputs, NDList weights, float learningRate, float learningRateBiasCorrection, float weightDecay, float rescaleGrad, float clipGrad, float beta1,
            float beta2, float epsilon, boolean lazyUpdate, boolean adamw) {
        // DJL folds the bias corrections into the learning rate: lr' m / (sqrt(v) + eps)
        Tensor w = persistent(inputs.get(0));
        step(OptimStep.Kind.ADAM, w, grad(inputs.get(1), rescaleGrad, clipGrad), new Tensor[] { persistent(inputs.get(2)), persistent(inputs.get(3)) }, learningRateBiasCorrection,
                beta1, beta2, epsilon, 1, 1, weightDecay, adamw ? 1 : 0);
    }

    @Override
    public void sgdUpdate(NDList inputs, NDList weights, float learningRate, float weightDecay, float rescaleGrad, float clipGrad, float momentum, boolean lazyUpdate) {
        Tensor w = persistent(inputs.get(0));
        Tensor g = grad(inputs.get(1), rescaleGrad, clipGrad);
        if (momentum == 0 || inputs.size() < 3) {
            step(OptimStep.Kind.SGD, w, g, new Tensor[0], learningRate, 0, weightDecay, 0);
        } else {
            step(OptimStep.Kind.SGD_MOMENTUM, w, g, new Tensor[] { persistent(inputs.get(2)) }, learningRate, momentum, weightDecay, 0);
        }
    }

    @Override
    public void nagUpdate(NDList inputs, NDList weights, float learningRate, float weightDecay, float rescaleGrad, float clipGrad, float momentum) {
        Tensor w = persistent(inputs.get(0));
        step(OptimStep.Kind.SGD_MOMENTUM, w, grad(inputs.get(1), rescaleGrad, clipGrad), new Tensor[] { persistent(inputs.get(2)) }, learningRate, momentum, weightDecay, 1);
    }

    @Override
    public void adagradUpdate(NDList inputs, NDList weights, float learningRate, float weightDecay, float rescaleGrad, float clipGrad, float epsilon) {
        Tensor w = persistent(inputs.get(0));
        step(OptimStep.Kind.ADAGRAD, w, grad(inputs.get(1), rescaleGrad, clipGrad), new Tensor[] { persistent(inputs.get(2)) }, learningRate, epsilon, weightDecay);
    }

    @Override
    public void rmspropUpdate(NDList inputs, NDList weights, float learningRate, float weightDecay, float rescaleGrad, float clipGrad, float rho, float momentum, float epsilon,
            boolean centered) {
        Tensor w = persistent(inputs.get(0));
        Tensor[] extra = extraStates(w, 2);
        Tensor sq = persistent(inputs.get(2));
        Tensor gavg = centered && inputs.size() > 3 ? persistent(inputs.get(3)) : extra[0];
        Tensor buf = centered && inputs.size() > 4 ? persistent(inputs.get(4)) : extra[1];
        step(OptimStep.Kind.RMSPROP, w, grad(inputs.get(1), rescaleGrad, clipGrad), new Tensor[] { sq, buf, gavg }, learningRate, rho, epsilon, weightDecay, momentum,
                centered ? 1 : 0);
    }

    @Override
    public void adadeltaUpdate(NDList inputs, NDList weights, float weightDecay, float rescaleGrad, float clipGrad, float rho, float epsilon) {
        Tensor w = persistent(inputs.get(0));
        step(OptimStep.Kind.ADADELTA, w, grad(inputs.get(1), rescaleGrad, clipGrad), new Tensor[] { persistent(inputs.get(2)), persistent(inputs.get(3)) }, 1, rho, epsilon,
                weightDecay);
    }

    // ---------------------------------------------------------------- layers

    @Override
    public NDList convolution(NDArray input, NDArray weight, NDArray bias, Shape stride, Shape padding, Shape dilation, int groups) {
        Tensor x = of(input);
        Tensor w = of(weight);
        Tensor b = of(bias);
        int spatial = w.rank() - 2;
        if (spatial < 1 || spatial > 3) {
            throw unsupported(spatial + "-D convolution");
        }
        int[] pd = axes(padding, spatial, 0);
        int[] pads = new int[2 * spatial];
        System.arraycopy(pd, 0, pads, 0, spatial);
        System.arraycopy(pd, 0, pads, spatial, spatial);
        Tensor y = F.conv(x, w, b, axes(stride, spatial, 1), pads, axes(dilation, spatial, 1), groups);
        return new NDList(wrap(y));
    }

    @Override
    public NDList deconvolution(NDArray input, NDArray weight, NDArray bias, Shape stride, Shape padding, Shape outPadding, Shape dilation, int groups) {
        Tensor w = of(weight);
        if (w.rank() != 4) {
            throw unsupported((w.rank() - 2) + "-D transposed convolution");
        }
        Tensor y = F.convTranspose2d(of(input), w, of(bias), square(stride, "stride"), square(padding, "padding"), square(outPadding, "output padding"),
                square(dilation, "dilation"), groups);
        return new NDList(wrap(y));
    }

    @Override
    public NDList linear(NDArray input, NDArray weight, NDArray bias) {
        Tensor x = of(input);
        Tensor y = F.linear(x, of(weight), of(bias));
        if (x.rank() > 2) {
            int[] d = x.shape().dims().clone();
            d[d.length - 1] = y.dim(-1);
            y = y.reshape(d);
        }
        return new NDList(wrap(y));
    }

    @Override
    public NDList embedding(NDArray input, NDArray weight, SparseFormat sparse) {
        return new NDList(wrap(F.embedding(JitnnNDArray.ints(of(input)), of(weight))));
    }

    @Override
    public NDList prelu(NDArray input, NDArray alpha) {
        return new NDList(wrap(F.prelu(of(input), of(alpha))));
    }

    @Override
    public NDList dropout(NDArray input, float rate, boolean training) {
        if (!training || rate == 0) {
            return new NDList(input);
        }
        return new NDList(wrap(F.dropout(of(input), rate, Tensor.seed(JitnnEngine.getInstance().nextSeed()), 0)));
    }

    @Override
    public NDList layerNorm(NDArray input, Shape normalizedShape, NDArray gamma, NDArray beta, float eps) {
        Tensor x = of(input);
        int features = Math.toIntExact(normalizedShape.size());
        Tensor g = gamma == null ? Tensor.ones(features) : of(gamma).reshape(features);
        Tensor b = beta == null ? null : of(beta).reshape(features);
        Tensor y = F.layerNorm(x.reshape(-1, features), g, b, eps).reshape(x.shape().dims());
        return new NDList(wrap(y));
    }

    @Override
    public NDList batchNorm(NDArray input, NDArray runningMean, NDArray runningVar, NDArray gamma, NDArray beta, int axis, float momentum, float eps, boolean training) {
        Tensor x = of(input);
        int rank = x.rank();
        int ax = axis < 0 ? axis + rank : axis;
        if (ax != 1) {
            x = x.transpose(1, ax);
        }
        int[] dims = x.shape().dims();
        Tensor x4 = rank == 2 || rank == 4 ? x : x.reshape(dims[0], dims[1], -1, 1);
        int c = dims[1];
        Tensor g = gamma == null ? Tensor.ones(c) : of(gamma);
        Tensor b = beta == null ? Tensor.zeros(c) : of(beta);
        // DJL's momentum keeps the running value (0.9); jitNN's, as PyTorch's, weighs the batch (0.1)
        Tensor y = F.batchNorm(x4, persistent(runningMean), persistent(runningVar), g, b, training, 1 - momentum, eps).reshape(dims);
        if (ax != 1) {
            y = y.transpose(1, ax);
        }
        return new NDList(wrap(y));
    }

    // ---------------------------------------------------------------- recurrent layers

    /** Time first, as jitNN's recurrent operators take it. */
    private static Tensor timeFirst(Tensor x, boolean batchFirst) {
        return batchFirst ? x.transpose(0, 1) : x;
    }

    private interface Layer {
        /** One direction of one layer over x [T, B, I] from the initial states; returns {y, h_n, (c_n)}. */
        Tensor[] run(Tensor x, Tensor wih, Tensor whh, Tensor bih, Tensor bhh, Tensor[] initial);
    }

    /**
     * Stacked, optionally bidirectional recurrence with PyTorch's gate order and state layout
     * ([layers x directions, B, H]), and DJL's parameter order (per layer: w_ih, w_hh of each
     * direction, then b_ih, b_hh of each direction).
     */
    private NDList recurrent(NDArray input, Tensor[] states, NDList params, boolean hasBiases, int numLayers, double dropRate, boolean training, boolean bidirectional,
            boolean batchFirst, Layer layer) {
        Tensor x = timeFirst(of(input), batchFirst);
        int dirs = bidirectional ? 2 : 1;
        int per = hasBiases ? 4 : 2;
        List<List<Tensor>> finals = new ArrayList<>();
        for (int s = 0; s < states.length; s++) {
            finals.add(new ArrayList<>());
        }
        for (int l = 0; l < numLayers; l++) {
            Tensor[] outs = new Tensor[dirs];
            for (int dir = 0; dir < dirs; dir++) {
                // DJL's order within a layer: the weights of every direction, then the biases
                int base = l * per * dirs;
                Tensor wih = of(params.get(base + 2 * dir));
                Tensor whh = of(params.get(base + 2 * dir + 1));
                Tensor bih = hasBiases ? of(params.get(base + 2 * dirs + 2 * dir)) : Tensor.zeros(wih.dim(0));
                Tensor bhh = hasBiases ? of(params.get(base + 2 * dirs + 2 * dir + 1)) : Tensor.zeros(wih.dim(0));
                Tensor[] initial = new Tensor[states.length];
                for (int s = 0; s < states.length; s++) {
                    initial[s] = states[s].narrow(0, l * dirs + dir, 1).squeeze(0);
                }
                Tensor in = dir == 0 ? x : F.flip(x, 0);
                Tensor[] r = layer.run(in, wih, whh, bih, bhh, initial);
                outs[dir] = dir == 0 ? r[0] : F.flip(r[0], 0);
                for (int s = 0; s < states.length; s++) {
                    finals.get(s).add(r[1 + s]);
                }
            }
            x = dirs == 1 ? outs[0] : Tensor.cat(2, outs);
            if (training && dropRate > 0 && l + 1 < numLayers) {
                x = F.dropout(x, (float) dropRate, Tensor.seed(JitnnEngine.getInstance().nextSeed()), 0);
            }
        }
        NDList out = new NDList(wrap(timeFirst(x, batchFirst)));
        for (List<Tensor> f : finals) {
            out.add(wrap(Tensor.stack(0, f.toArray(new Tensor[0]))));
        }
        return out;
    }

    @Override
    public NDList lstm(NDArray input, NDList states, NDList params, boolean hasBiases, int numLayers, double dropRate, boolean training, boolean bidirectional,
            boolean batchFirst) {
        return recurrent(input, new Tensor[] { of(states.get(0)), of(states.get(1)) }, params, hasBiases, numLayers, dropRate, training, bidirectional, batchFirst,
                (x, wih, whh, bih, bhh, init) -> F.lstm(x, wih, whh, bih, bhh, init[0], init[1]));
    }

    @Override
    public NDList gru(NDArray input, NDArray state, NDList params, boolean hasBiases, int numLayers, double dropRate, boolean training, boolean bidirectional,
            boolean batchFirst) {
        return recurrent(input, new Tensor[] { of(state) }, params, hasBiases, numLayers, dropRate, training, bidirectional, batchFirst,
                (x, wih, whh, bih, bhh, init) -> F.gru(x, wih, whh, bih, bhh, init[0]));
    }

    @Override
    public NDList rnn(NDArray input, NDArray state, NDList params, boolean hasBiases, int numLayers, RNN.Activation activation, double dropRate, boolean training,
            boolean bidirectional, boolean batchFirst) {
        boolean relu = activation == RNN.Activation.RELU;
        return recurrent(input, new Tensor[] { of(state) }, params, hasBiases, numLayers, dropRate, training, bidirectional, batchFirst, (x, wih, whh, bih, bhh, init) -> {
            // h_t = act(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh), unrolled: the input projection for all steps at once
            int steps = x.dim(0);
            Tensor xp = F.linear(x.reshape(-1, x.dim(2)), wih, bih).reshape(steps, x.dim(1), -1);
            Tensor h = init[0];
            Tensor[] ys = new Tensor[steps];
            for (int s = 0; s < steps; s++) {
                Tensor a = xp.narrow(0, s, 1).squeeze(0).add(F.linear(h, whh, bhh));
                h = relu ? a.relu() : a.tanh();
                ys[s] = h;
            }
            return new Tensor[] { Tensor.stack(0, ys), h };
        });
    }

    // ---------------------------------------------------------------- images

    @Override
    public NDArray interpolation(long[] size, int mode, boolean alignCorners) {
        // DJL's modes: 0 nearest, 1 bilinear
        return wrap(F.interpolate(t(), (int) size[0], (int) size[1], mode == 1, alignCorners));
    }

    @Override
    public NDArray resize(int width, int height, int interpolation) {
        // DJL images are HWC (or NHWC): resize through NCHW
        Tensor x = t();
        boolean batched = x.rank() == 4;
        Tensor nchw = batched ? x.permute(0, 3, 1, 2) : x.permute(2, 0, 1).unsqueeze(0);
        Tensor y = F.interpolate(nchw, height, width, interpolation != 0, false);
        y = batched ? y.permute(0, 2, 3, 1) : y.squeeze(0).permute(1, 2, 0);
        return array.wrap(y, array.getDataType().isFloating() ? array.getDataType() : DataType.FLOAT32);
    }

    @Override
    public NDArray randomFlipLeftRight() {
        throw unsupported("random image augmentation");
    }

    @Override
    public NDArray randomFlipTopBottom() {
        throw unsupported("random image augmentation");
    }

    @Override
    public NDArray randomBrightness(float brightness) {
        throw unsupported("random image augmentation");
    }

    @Override
    public NDArray randomHue(float hue) {
        throw unsupported("random image augmentation");
    }

    @Override
    public NDArray randomColorJitter(float brightness, float contrast, float saturation, float hue) {
        throw unsupported("random image augmentation");
    }

    // ---------------------------------------------------------------- misc

    @Override
    public NDArrayIndexer getIndexer(NDManager manager) {
        return new JitnnIndexer((JitnnNDManager) (manager instanceof JitnnNDManager ? manager : array.getManager()));
    }

    @Override
    public NDArray where(NDArray condition, NDArray other) {
        return array.wrap(F.where(of(condition), t(), of(other)), array.getDataType());
    }

    @Override
    public NDArray stack(NDList arrays, int axis) {
        Tensor[] parts = new Tensor[arrays.size() + 1];
        parts[0] = t();
        for (int i = 0; i < arrays.size(); i++) {
            parts[i + 1] = of(arrays.get(i));
        }
        int ax = axis < 0 ? axis + t().rank() + 1 : axis;
        return array.wrap(Tensor.stack(ax, parts), array.getDataType());
    }

    @Override
    public NDArray concat(NDList arrays, int axis) {
        Tensor[] parts = new Tensor[arrays.size() + 1];
        parts[0] = t();
        for (int i = 0; i < arrays.size(); i++) {
            parts[i + 1] = of(arrays.get(i));
        }
        int ax = axis < 0 ? axis + t().rank() : axis;
        return array.wrap(Tensor.cat(ax, parts), array.getDataType());
    }

    @Override
    public NDList multiBoxTarget(NDList inputs, float iouThreshold, float ignoreLabel, float negativeMiningRatio, float negativeMiningThreshold, int minNegativeSamples) {
        throw unsupported("multibox operators");
    }

    @Override
    public NDList multiBoxPrior(List<Float> sizes, List<Float> ratios, List<Float> steps, List<Float> offsets, boolean clip) {
        throw unsupported("multibox operators");
    }

    @Override
    public NDList multiBoxDetection(NDList inputs, boolean clip, float threshold, int backgroundId, float nmsThreshold, boolean forceSuppress, int nmsTopK) {
        throw unsupported("multibox operators");
    }

    @Override
    public NDArray getArray() {
        return array;
    }
}
