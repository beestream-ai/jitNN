package org.beehive.jitnn.ops;

import java.util.HashMap;
import java.util.Map;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.ElementwiseKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Rotary position embedding (RoPE) of x [..., T, D], or of x [..., T, H, D] (every head of a
 * token shares its position), for positions offset .. offset + T - 1, with frequencies
 * base^(-2i / D). Inputs [x, cos, sin]; iattr = [T, D, offset, interleaved, group] with group the
 * rows per position (1, or H); fattr = [sign]. The tables are cached per (positions, D, base,
 * scaling) as persistent tensors. A {@link Scaling} adjusts the frequencies for longer contexts.
 */
public final class Rope implements Op {

    static final Rope OP = new Rope();
    private static final Map<String, Tensor[]> TABLES = new HashMap<>();

    private Rope() {
    }

    /** A rescaling of the RoPE frequencies, as Hugging Face's rope_scaling. */
    public sealed interface Scaling {

        /** The frequency used in place of {@code frequency} (radians per position). */
        double apply(double frequency);

        /** Positions divided by {@code factor} (rope_type linear). */
        static Scaling linear(double factor) {
            return new Linear(factor);
        }

        /**
         * Llama 3.1's rescaling (rope_type llama3): frequencies with wavelengths above
         * originalContext / lowFreqFactor are divided by {@code factor}, those below
         * originalContext / highFreqFactor are kept, and the band between is interpolated.
         */
        static Scaling llama3(double factor, double lowFreqFactor, double highFreqFactor, int originalContext) {
            return new Llama3(factor, lowFreqFactor, highFreqFactor, originalContext);
        }
    }

    record Linear(double factor) implements Scaling {
        @Override
        public double apply(double frequency) {
            return frequency / factor;
        }
    }

    record Llama3(double factor, double lowFreqFactor, double highFreqFactor, int originalContext) implements Scaling {
        @Override
        public double apply(double frequency) {
            double wavelength = 2 * Math.PI / frequency;
            if (wavelength < originalContext / highFreqFactor) {
                return frequency;
            }
            if (wavelength > originalContext / lowFreqFactor) {
                return frequency / factor;
            }
            double smooth = (originalContext / wavelength - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
            return (1 - smooth) * frequency / factor + smooth * frequency;
        }
    }

    public static Tensor apply(Tensor x, float base, int offset, boolean interleaved) {
        return apply(x, base, null, offset, interleaved);
    }

    /** As {@link #apply(Tensor, float, int, boolean)} with the frequencies rescaled by {@code scaling} (null for none). */
    public static Tensor apply(Tensor x, float base, Scaling scaling, int offset, boolean interleaved) {
        Checks.f32(x);
        if (x.rank() < 2 || x.dim(-1) % 2 != 0) {
            throw new IllegalArgumentException("rope needs [..., T, D] with even D, got " + x.shape());
        }
        int seq = x.dim(-2);
        int dim = x.dim(-1);
        Tensor[] t = tables(offset + seq, dim, base, scaling);
        return record(x, t, new int[] { seq, dim, offset, interleaved ? 1 : 0, 1 }, 1.0f);
    }

    /** RoPE of x [..., T, H, D]: the H heads of a token share its position. */
    public static Tensor applyHeads(Tensor x, float base, int offset, boolean interleaved) {
        return applyHeads(x, base, null, offset, interleaved);
    }

    /** As {@link #applyHeads(Tensor, float, int, boolean)} with the frequencies rescaled by {@code scaling} (null for none). */
    public static Tensor applyHeads(Tensor x, float base, Scaling scaling, int offset, boolean interleaved) {
        Checks.f32(x);
        if (x.rank() < 3 || x.dim(-1) % 2 != 0) {
            throw new IllegalArgumentException("rope needs [..., T, H, D] with even D, got " + x.shape());
        }
        int seq = x.dim(-3);
        int dim = x.dim(-1);
        Tensor[] t = tables(offset + seq, dim, base, scaling);
        return record(x, t, new int[] { seq, dim, offset, interleaved ? 1 : 0, x.dim(-2) }, 1.0f);
    }

    private static Tensor record(Tensor x, Tensor[] t, int[] iattr, float sign) {
        Tensor out = Tensor.result(x.shape(), DType.F32);
        Tensor.record(OP, new Tensor[] { x, t[0], t[1] }, new Tensor[] { out }, iattr, new float[] { sign }, true);
        return out;
    }

    /** cos and sin of t * base^(-2i / D) (rescaled by {@code scaling} when not null) for t < positions, i < D / 2, computed in double. */
    static synchronized Tensor[] tables(int positions, int dim, float base, Scaling scaling) {
        String key = positions + "/" + dim + "/" + base + "/" + scaling;
        return TABLES.computeIfAbsent(key, k -> {
            int half = dim / 2;
            float[] c = new float[positions * half];
            float[] s = new float[positions * half];
            for (int t = 0; t < positions; t++) {
                for (int i = 0; i < half; i++) {
                    double frequency = Math.pow(base, -2.0 * i / dim);
                    double angle = t * (scaling != null ? scaling.apply(frequency) : frequency);
                    c[t * half + i] = (float) Math.cos(angle);
                    s[t * half + i] = (float) Math.sin(angle);
                }
            }
            return new Tensor[] { Tensor.persistent(Tensor.of(c, positions, half)), Tensor.persistent(Tensor.of(s, positions, half)) };
        });
    }

    @Override
    public String name() {
        return "rope";
    }

    @Override
    public void emit(Emitter e, Node n) {
        FloatArray x = e.f32(n.in(0));
        FloatArray cos = e.f32(n.in(1));
        FloatArray sin = e.f32(n.in(2));
        FloatArray out = e.f32(n.out(0));
        int seq = n.iattr[0];
        int dim = n.iattr[1];
        int offset = n.iattr[2];
        int interleaved = n.iattr[3];
        int group = n.iattr[4];
        float sign = n.fattr[0];
        int pairs = n.out(0).numel() / 2;
        e.task(Emitter.grid1D(pairs), (g, id) -> g.task(id, ElementwiseKernels::rope, e.context(), x, out, cos, sin, seq, dim, group, offset, interleaved, sign, pairs));
    }

    @Override
    public void reference(Node n) {
        FloatArray x = (FloatArray) n.in(0).storage().host();
        FloatArray cos = (FloatArray) n.in(1).storage().host();
        FloatArray sin = (FloatArray) n.in(2).storage().host();
        FloatArray out = (FloatArray) n.out(0).storage().host();
        int seq = n.iattr[0];
        int dim = n.iattr[1];
        int offset = n.iattr[2];
        boolean interleaved = n.iattr[3] != 0;
        double sign = n.fattr[0];
        int half = dim / 2;
        for (int row = 0; row < n.out(0).numel() / dim; row++) {
            int t = (row / n.iattr[4]) % seq + offset;
            for (int p = 0; p < half; p++) {
                int ia = interleaved ? row * dim + 2 * p : row * dim + p;
                int ib = interleaved ? ia + 1 : ia + half;
                double c = cos.get(t * half + p);
                double s = sign * sin.get(t * half + p);
                double a = x.get(ia);
                double b = x.get(ib);
                out.set(ia, (float) (a * c - b * s));
                out.set(ib, (float) (a * s + b * c));
            }
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gy) {
        return new Tensor[] { record(gy, new Tensor[] { n.in(1), n.in(2) }, n.iattr.clone(), -n.fattr[0]), null, null };
    }
}
