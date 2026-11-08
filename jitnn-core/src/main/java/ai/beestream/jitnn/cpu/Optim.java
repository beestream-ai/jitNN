package ai.beestream.jitnn.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Optimiser updates on the CPU in FP32 vector lanes (the reference updates in double). */
public final class Optim {

    private static final VectorSpecies<Float> S = FloatVector.SPECIES_PREFERRED;
    private static final int L = S.length();
    private static final ByteOrder ORDER = ByteOrder.nativeOrder();
    private static final ValueLayout.OfFloat F = ValueLayout.JAVA_FLOAT;

    private Optim() {
    }

    /**
     * Adam (AdamW when {@code decoupled}) on elements [from, to): m and v moments, gradients scaled
     * by {@code factor}, c1 and c2 the bias corrections 1 - beta^t.
     */
    public static void adam(MemorySegment p, MemorySegment g, MemorySegment m, MemorySegment v, int from, int to, float lr, float b1, float b2, float eps, float c1, float c2,
            float wd, boolean decoupled, float factor) {
        float step = lr / c1;
        float rc2 = 1 / c2;
        int i = from;
        int vEnd = from + S.loopBound(to - from);
        for (; i < vEnd; i += L) {
            long at = (long) i * 4;
            FloatVector w = FloatVector.fromMemorySegment(S, p, at, ORDER);
            FloatVector grad = FloatVector.fromMemorySegment(S, g, at, ORDER).mul(factor);
            if (decoupled) {
                w = w.sub(w.mul(lr * wd));
            } else {
                grad = w.fma(wd, 0f).add(grad);
            }
            FloatVector mi = FloatVector.fromMemorySegment(S, m, at, ORDER).mul(b1).add(grad.mul(1 - b1));
            FloatVector vi = FloatVector.fromMemorySegment(S, v, at, ORDER).mul(b2).add(grad.mul(grad).mul(1 - b2));
            mi.intoMemorySegment(m, at, ORDER);
            vi.intoMemorySegment(v, at, ORDER);
            w.sub(mi.mul(step).div(vi.mul(rc2).sqrt().add(eps))).intoMemorySegment(p, at, ORDER);
        }
        for (; i < to; i++) {
            float w = p.getAtIndex(F, i);
            float grad = g.getAtIndex(F, i) * factor;
            if (decoupled) {
                w -= lr * wd * w;
            } else {
                grad += wd * w;
            }
            float mi = b1 * m.getAtIndex(F, i) + (1 - b1) * grad;
            float vi = b2 * v.getAtIndex(F, i) + (1 - b2) * grad * grad;
            m.setAtIndex(F, i, mi);
            v.setAtIndex(F, i, vi);
            p.setAtIndex(F, i, w - mi * step / ((float) Math.sqrt(vi * rc2) + eps));
        }
    }
}
