package org.beehive.jitnn.ops;

import java.util.List;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Shape;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.LowPrecision;
import org.beehive.jitnn.kernels.MultiTensorKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Statistics of a set of gradients in one operator: stats = [norm, factor, skip] where norm is the
 * global L2 norm of the gradients times {@code preScale} (a loss scaler's 1 / scale, or 1),
 * factor = preScale times the clip coefficient min(1, maxNorm / (norm + 1e-6)) when clipping,
 * and skip = 1 when any gradient holds a NaN or infinity (tested on the bit pattern). One
 * work-group per gradient, eight gradients per launch, and one combining thread. Inputs [preScale, g_1 .. g_P]; iattr = [P,
 * clip]; fattr = [maxNorm].
 */
public final class GradStats implements Op {

    static final GradStats OP = new GradStats();

    private GradStats() {
    }

    public static Tensor apply(List<Tensor> grads, Tensor preScale, float maxNorm, boolean clip) {
        Tensor[] in = new Tensor[grads.size() + 1];
        in[0] = preScale != null ? preScale : Tensor.of(new float[] { 1.0f }, 1);
        for (int i = 0; i < grads.size(); i++) {
            Checks.f32(grads.get(i));
            in[i + 1] = grads.get(i);
        }
        Tensor stats = Tensor.result(Shape.of(3), DType.F32);
        Tensor.record(OP, in, new Tensor[] { stats }, new int[] { grads.size(), clip ? 1 : 0 }, new float[] { maxNorm }, true);
        return stats;
    }

    @Override
    public String name() {
        return "grad_stats";
    }

    @Override
    public void emit(Emitter e, Node n) {
        int count = n.iattr[0];
        int clip = n.iattr[1];
        float maxNorm = n.fattr[0];
        FloatArray preScale = e.f32(n.in(0));
        FloatArray stats = e.f32(n.out(0));
        FloatArray partial = e.scratch(2 * count);
        int per = MultiTensorKernels.STATS_SLOTS;
        for (int start = 0; start < count; start += per) {
            // up to 8 gradients per launch, one work-group each; unused slots have size 0
            FloatArray[] g = new FloatArray[per];
            int[] sizes = new int[per];
            for (int k = 0; k < per; k++) {
                boolean used = start + k < count;
                g[k] = used ? e.f32(n.in(start + k + 1)) : e.scratch(1);
                sizes[k] = used ? n.in(start + k + 1).numel() : 0;
            }
            IntArray sz = e.constant(sizes);
            int slot0 = start;
            e.task(Emitter.blocks1D(per, 256), (tg, id) -> tg.task(id, MultiTensorKernels::sumSquaresFinite, e.context(), g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
                    partial, sz, slot0));
        }
        e.task(Emitter.grid1D(1), (tg, id) -> tg.task(id, LowPrecision::gradStats, e.context(), partial, preScale, stats, count, maxNorm, clip));
    }

    @Override
    public void reference(Node n) {
        double total = 0;
        boolean bad = false;
        for (int i = 1; i < n.inputs.length; i++) {
            FloatArray g = (FloatArray) n.in(i).storage().host();
            for (int j = 0; j < n.in(i).numel(); j++) {
                double v = g.get(j);
                total += v * v;
                bad |= !Float.isFinite(g.get(j));
            }
        }
        double scale = ((FloatArray) n.in(0).storage().host()).get(0);
        double norm = Math.sqrt(total) * scale;
        double factor = n.iattr[1] != 0 ? scale * Math.min(1.0, n.fattr[0] / (norm + 1e-6)) : scale;
        FloatArray stats = (FloatArray) n.out(0).storage().host();
        stats.set(0, (float) norm);
        stats.set(1, (float) factor);
        stats.set(2, bad ? 1.0f : 0.0f);
    }
}
