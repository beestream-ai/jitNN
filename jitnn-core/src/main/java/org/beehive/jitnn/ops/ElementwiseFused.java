package org.beehive.jitnn.ops;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Emitter;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Op;
import org.beehive.jitnn.kernels.FusedKernels;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * A chain of unary and binary element-wise operators run as one kernel ({@link FusedKernels}),
 * built by {@link Fusion} from operators whose intermediates nobody else reads. iattr is the
 * kernel's meta (inputs, instructions, rank, dims, strides, program); fattr the per-instruction
 * alpha and beta. No gradient: it only exists after autograd has recorded the trace.
 */
final class ElementwiseFused implements Op {

    static final ElementwiseFused OP = new ElementwiseFused();

    private ElementwiseFused() {
    }

    @Override
    public String name() {
        return "fused";
    }

    @Override
    public void emit(Emitter e, Node n) {
        int inputs = n.iattr[0];
        FloatArray[] in = new FloatArray[FusedKernels.MAX_INPUTS];
        for (int k = 0; k < FusedKernels.MAX_INPUTS; k++) {
            in[k] = e.f32(n.in(Math.min(k, inputs - 1)));
        }
        FloatArray out = e.f32(n.out(0));
        IntArray meta = e.constant(n.iattr);
        FloatArray params = e.constantF32(n.fattr.length == 0 ? new float[2] : n.fattr);
        int count = n.out(0).numel();
        FloatArray i0 = in[0], i1 = in[1], i2 = in[2], i3 = in[3];
        e.task(Emitter.grid1D(count), (g, id) -> g.task(id, FusedKernels::fused, e.context(), i0, i1, i2, i3, out, meta, params, count));
    }

    @Override
    public void reference(Node n) {
        int[] m = n.iattr;
        int inputs = m[0], instructions = m[1], rank = m[2];
        FloatArray out = (FloatArray) n.out(0).storage().host();
        FloatArray[] in = new FloatArray[inputs];
        for (int k = 0; k < inputs; k++) {
            in[k] = (FloatArray) n.in(k).storage().host();
        }
        double[] regs = new double[FusedKernels.REGISTERS];
        for (int i = 0; i < n.out(0).numel(); i++) {
            int rem = i;
            int[] off = new int[FusedKernels.MAX_INPUTS];
            for (int d = rank - 1; d >= 0; d--) {
                int c = rem % m[3 + d];
                rem /= m[3 + d];
                for (int k = 0; k < FusedKernels.MAX_INPUTS; k++) {
                    off[k] += c * m[3 + (1 + k) * rank + d];
                }
            }
            for (int k = 0; k < inputs; k++) {
                regs[k] = in[k].get(off[k]);
            }
            int p = 3 + rank + FusedKernels.MAX_INPUTS * rank;
            double r = 0;
            for (int j = 0; j < instructions; j++) {
                double x = regs[m[p + 2]];
                double y = regs[m[p + 3]];
                double alpha = n.fattr[2 * j];
                double beta = n.fattr[2 * j + 1];
                // each step rounds to FP32, as the unfused operators' outputs do
                r = (float) (m[p] == FusedKernels.UNARY ? RefMath.unary(m[p + 1], x, alpha, beta) : RefMath.binary(m[p + 1], x, y, alpha, beta));
                regs[m[p + 4]] = r;
                p += 5;
            }
            out.set(i, (float) r);
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gradOut) {
        throw new UnsupportedOperationException("fused element-wise chains exist only after autograd");
    }
}
