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

    /** Chains run as FusedKernels.fusedProgram1..6 (specialised per chain) unless {@code -Djitnn.fuse.program=false}. */
    private static final boolean PROGRAM = !"false".equals(System.getProperty("jitnn.fuse.program"));

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
        FloatArray[] out = new FloatArray[FusedKernels.MAX_OUTPUTS];
        for (int k = 0; k < FusedKernels.MAX_OUTPUTS; k++) {
            out[k] = e.f32(n.out(Math.min(k, n.outputs.length - 1)));
        }
        IntArray meta = e.constant(n.iattr);
        FloatArray params = e.constantF32(n.fattr.length == 0 ? new float[2] : n.fattr);
        int count = n.out(0).numel();
        FloatArray i0 = in[0], i1 = in[1], i2 = in[2], i3 = in[3];
        FloatArray o0 = out[0], o1 = out[1], o2 = out[2], o3 = out[3];
        int[] program = PROGRAM && flat(n.iattr) ? program(n.iattr) : null;
        if (program != null) {
            int nin = inputs, q0 = program[0], q1 = program[1], q2 = program[2], q3 = program[3], q4 = program[4], q5 = program[5], outs = program[6];
            // one kernel per chain length: a short chain compiles only its own instructions
            switch (n.iattr[1]) {
                case 1 -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, FusedKernels::fusedProgram1, e.context(), i0, i1, i2, i3, o0, o1, o2, o3, params, count, nin, q0, outs));
                case 2 -> e.task(Emitter.grid1D(count), (g, id) -> g.task(id, FusedKernels::fusedProgram2, e.context(), i0, i1, i2, i3, o0, o1, o2, o3, params, count, nin, q0, q1, outs));
                case 3 -> e.task(Emitter.grid1D(count),
                        (g, id) -> g.task(id, FusedKernels::fusedProgram3, e.context(), i0, i1, i2, i3, o0, o1, o2, o3, params, count, nin, q0, q1, q2, outs));
                case 4 -> e.task(Emitter.grid1D(count),
                        (g, id) -> g.task(id, FusedKernels::fusedProgram4, e.context(), i0, i1, i2, i3, o0, o1, o2, o3, params, count, nin, q0, q1, q2, q3, outs));
                case 5 -> e.task(Emitter.grid1D(count),
                        (g, id) -> g.task(id, FusedKernels::fusedProgram5, e.context(), i0, i1, i2, i3, o0, o1, o2, o3, params, count, nin, q0, q1, q2, q3, q4, outs));
                default -> e.task(Emitter.grid1D(count),
                        (g, id) -> g.task(id, FusedKernels::fusedProgram6, e.context(), i0, i1, i2, i3, o0, o1, o2, o3, params, count, nin, q0, q1, q2, q3, q4, q5, outs));
            }
        } else if (flat(n.iattr)) {
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, FusedKernels::fusedFlat, e.context(), i0, i1, i2, i3, o0, o1, o2, o3, meta, params, count));
        } else {
            e.task(Emitter.grid1D(count), (g, id) -> g.task(id, FusedKernels::fused, e.context(), i0, i1, i2, i3, o0, o1, o2, o3, meta, params, count));
        }
    }

    /**
     * The chain as the fusedProgram kernels' scalars: six packed instructions (-1 for
     * none) and the packed outputs; null when it has more instructions than that kernel takes.
     * Registers are renumbered: inputs 0-3, instruction j's result 4 + j.
     */
    static int[] program(int[] m) {
        int inputs = m[0], instructions = m[1], rank = m[2];
        if (instructions > FusedKernels.PROGRAM_INSTRUCTIONS) {
            return null;
        }
        int[] out = new int[FusedKernels.PROGRAM_INSTRUCTIONS + 1];
        java.util.Arrays.fill(out, -1);
        int p = 3 + rank + FusedKernels.MAX_INPUTS * rank;
        for (int j = 0; j < instructions; j++) {
            int kind = m[p], op = m[p + 1], a = m[p + 2], b = m[p + 3], dest = m[p + 4];
            if (dest != inputs + j || op > 511) {
                return null;
            }
            out[j] = kind | op << 1 | renumber(a, inputs) << 10 | renumber(b, inputs) << 14;
            p += 5;
        }
        int outputs = m[p];
        int outs = outputs;
        for (int k = 0; k < outputs; k++) {
            outs |= renumber(m[p + 1 + k], inputs) << (4 + 4 * k);
        }
        out[FusedKernels.PROGRAM_INSTRUCTIONS] = outs;
        return out;
    }

    private static int renumber(int r, int inputs) {
        return r < inputs ? r : 4 + r - inputs;
    }

    /** True when every input's strides are the output's contiguous ones (see {@link FusedKernels#fusedFlat}). */
    static boolean flat(int[] m) {
        int inputs = m[0], rank = m[2];
        for (int k = 0; k < inputs; k++) {
            int stride = 1;
            for (int d = rank - 1; d >= 0; d--) {
                if (m[3 + d] > 1 && m[3 + rank + k * rank + d] != stride) {
                    return false;
                }
                stride *= m[3 + d];
            }
        }
        return true;
    }

    @Override
    public void reference(Node n) {
        int[] m = n.iattr;
        int inputs = m[0], instructions = m[1], rank = m[2];
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
            for (int k = 0; k < m[p]; k++) {
                ((FloatArray) n.out(k).storage().host()).set(i, (float) regs[m[p + 1 + k]]);
            }
        }
    }

    @Override
    public Tensor[] backward(Node n, Tensor gradOut) {
        throw new UnsupportedOperationException("fused element-wise chains exist only after autograd");
    }
}
