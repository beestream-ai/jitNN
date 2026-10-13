package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * A chain of element-wise operators in one kernel: a small program over registers, run per output
 * element. The inputs are read through per-input strides over the output's coordinates, so a
 * broadcast or a permuted view costs no copy.
 *
 * <p>{@code meta} = [inputs, instructions, rank, dims[rank], strides[MAX_INPUTS][rank], then per
 * instruction: kind (0 unary, 1 binary), op code, register a, register b, destination register,
 * then outputs, then each output's register]. Registers 0 .. inputs-1 hold the inputs; instruction
 * j writes register inputs + j. {@code params} holds each instruction's alpha and beta. Output k
 * receives its register's value, so a chain can also store intermediates other operators read.
 */
public final class FusedKernels {

    public static final int MAX_INPUTS = 4;
    public static final int MAX_INSTRUCTIONS = 8;
    public static final int MAX_OUTPUTS = 4;
    public static final int REGISTERS = MAX_INPUTS + MAX_INSTRUCTIONS;
    public static final int UNARY = 0;
    public static final int BINARY = 1;

    private FusedKernels() {
    }

    public static void fused(KernelContext kc, FloatArray i0, FloatArray i1, FloatArray i2, FloatArray i3, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3,
            IntArray meta, FloatArray params, int n) {
        float[] regs = kc.allocateFloatLocalArray(256 * REGISTERS);
        int i = kc.globalIdx;
        int base = kc.localIdx * REGISTERS;
        if (i < n) {
            int inputs = meta.get(0);
            int instructions = meta.get(1);
            int rank = meta.get(2);
            int rem = i;
            int o0 = 0;
            int o1 = 0;
            int o2 = 0;
            int o3 = 0;
            for (int d = rank - 1; d >= 0; d--) {
                int size = meta.get(3 + d);
                int c = rem % size;
                rem = rem / size;
                o0 += c * meta.get(3 + rank + d);
                o1 += c * meta.get(3 + 2 * rank + d);
                o2 += c * meta.get(3 + 3 * rank + d);
                o3 += c * meta.get(3 + 4 * rank + d);
            }
            regs[base] = i0.get(o0);
            if (inputs > 1) {
                regs[base + 1] = i1.get(o1);
            }
            if (inputs > 2) {
                regs[base + 2] = i2.get(o2);
            }
            if (inputs > 3) {
                regs[base + 3] = i3.get(o3);
            }
            int p = 3 + rank + MAX_INPUTS * rank;
            float r;
            for (int j = 0; j < instructions; j++) {
                int kind = meta.get(p);
                int op = meta.get(p + 1);
                float x = regs[base + meta.get(p + 2)];
                float y = regs[base + meta.get(p + 3)];
                float alpha = params.get(2 * j);
                float beta = params.get(2 * j + 1);
                r = 0.0f;
                if (kind == UNARY) {
                    if (op >= ElementwiseKernels.IS_NAN) {
                        r = ElementwiseKernels.unaryNaN(op, x, alpha, beta);
                    } else if (op >= ElementwiseKernels.SIN) {
                        r = ElementwiseKernels.unaryMath(op, x, alpha, beta);
                    } else if (op >= ElementwiseKernels.ELU) {
                        r = ElementwiseKernels.unaryActivation(op, x, alpha, beta);
                    } else {
                        r = ElementwiseKernels.unaryBase(op, x, alpha, beta);
                    }
                } else {
                    if (op >= ElementwiseKernels.MAXIMUM) {
                        r = ElementwiseKernels.binaryNaN(op, x, y, alpha, beta);
                    } else if (op >= ElementwiseKernels.ATAN2) {
                        r = ElementwiseKernels.binaryMath(op, x, y, alpha, beta);
                    } else if (op >= ElementwiseKernels.ELU_GRAD) {
                        r = ElementwiseKernels.binaryActivationGrad(op, x, y, alpha, beta);
                    } else {
                        r = ElementwiseKernels.binaryBase(op, x, y, alpha, beta);
                    }
                }
                regs[base + meta.get(p + 4)] = r;
                p += 5;
            }
            int outputs = meta.get(p);
            out0.set(i, regs[base + meta.get(p + 1)]);
            if (outputs > 1) {
                out1.set(i, regs[base + meta.get(p + 2)]);
            }
            if (outputs > 2) {
                out2.set(i, regs[base + meta.get(p + 3)]);
            }
            if (outputs > 3) {
                out3.set(i, regs[base + meta.get(p + 4)]);
            }
        }
    }
}
