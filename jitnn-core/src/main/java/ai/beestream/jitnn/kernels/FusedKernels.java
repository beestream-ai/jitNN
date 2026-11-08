/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.kernels;

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

    /**
     * {@link #fused} when every input has the output's layout (no broadcast, no permuted view):
     * input element i goes with output element i, so no coordinates are computed.
     */
    public static void fusedFlat(KernelContext kc, FloatArray i0, FloatArray i1, FloatArray i2, FloatArray i3, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3,
            IntArray meta, FloatArray params, int n) {
        float[] regs = kc.allocateFloatLocalArray(256 * REGISTERS);
        int i = kc.globalIdx;
        int base = kc.localIdx * REGISTERS;
        if (i < n) {
            int inputs = meta.get(0);
            int instructions = meta.get(1);
            int rank = meta.get(2);
            // every input laid out as the output: element i of each
            regs[base] = i0.get(i);
            if (inputs > 1) {
                regs[base + 1] = i1.get(i);
            }
            if (inputs > 2) {
                regs[base + 2] = i2.get(i);
            }
            if (inputs > 3) {
                regs[base + 3] = i3.get(i);
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

    /** Instructions the {@code fusedProgram} kernels take, at most. */
    public static final int PROGRAM_INSTRUCTIONS = 6;

    /*
     * fusedProgram1 .. fusedProgram6 (written by scripts/codegen/fused_program.py): fusedFlat with
     * the program in scalar arguments, one kernel per instruction count so a short chain compiles
     * only its own instructions. One int per instruction: kind in bit 0, op code in bits 1-9,
     * register a in bits 10-13, register b in bits 14-17; outputs' registers in outs (count in bits
     * 0-3, then 4 bits per output). Registers 0-3 are the inputs and 4 + j instruction j's result.
     * TornadoVM specialises each kernel on its scalars, so the program compiles to straight-line code
     * with the registers in private variables.
     */

    /** fusedFlat with the program in scalars (see above), for a chain of 1 instruction. */
    public static void fusedProgram1(KernelContext kc, FloatArray i0, FloatArray i1, FloatArray i2, FloatArray i3, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3,
            FloatArray params, int n, int inputs, int p0, int outs) {
        int i = kc.globalIdx;
        if (i < n) {
            float x0 = i0.get(i);
            float x1 = inputs > 1 ? i1.get(i) : 0.0f;
            float x2 = inputs > 2 ? i2.get(i) : 0.0f;
            float x3 = inputs > 3 ? i3.get(i) : 0.0f;
            float t0;
            {
                float x = reg((p0 >> 10) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p0 >> 14) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p0 >> 1) & 511;
                if ((p0 & 1) == UNARY) {
                    t0 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(0), params.get(1)) : unaryLow(op, x, params.get(0), params.get(1));
                } else {
                    t0 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(0), params.get(1)) : binaryLow(op, x, y, params.get(0), params.get(1));
                }
            }
            float t1 = 0.0f;
            float t2 = 0.0f;
            float t3 = 0.0f;
            float t4 = 0.0f;
            float t5 = 0.0f;
            store(i, outs, out0, out1, out2, out3, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5);
        }
    }

    /** fusedFlat with the program in scalars (see above), for a chain of 2 instructions. */
    public static void fusedProgram2(KernelContext kc, FloatArray i0, FloatArray i1, FloatArray i2, FloatArray i3, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3,
            FloatArray params, int n, int inputs, int p0, int p1, int outs) {
        int i = kc.globalIdx;
        if (i < n) {
            float x0 = i0.get(i);
            float x1 = inputs > 1 ? i1.get(i) : 0.0f;
            float x2 = inputs > 2 ? i2.get(i) : 0.0f;
            float x3 = inputs > 3 ? i3.get(i) : 0.0f;
            float t0;
            {
                float x = reg((p0 >> 10) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p0 >> 14) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p0 >> 1) & 511;
                if ((p0 & 1) == UNARY) {
                    t0 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(0), params.get(1)) : unaryLow(op, x, params.get(0), params.get(1));
                } else {
                    t0 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(0), params.get(1)) : binaryLow(op, x, y, params.get(0), params.get(1));
                }
            }
            float t1;
            {
                float x = reg((p1 >> 10) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p1 >> 14) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p1 >> 1) & 511;
                if ((p1 & 1) == UNARY) {
                    t1 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(2), params.get(3)) : unaryLow(op, x, params.get(2), params.get(3));
                } else {
                    t1 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(2), params.get(3)) : binaryLow(op, x, y, params.get(2), params.get(3));
                }
            }
            float t2 = 0.0f;
            float t3 = 0.0f;
            float t4 = 0.0f;
            float t5 = 0.0f;
            store(i, outs, out0, out1, out2, out3, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5);
        }
    }

    /** fusedFlat with the program in scalars (see above), for a chain of 3 instructions. */
    public static void fusedProgram3(KernelContext kc, FloatArray i0, FloatArray i1, FloatArray i2, FloatArray i3, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3,
            FloatArray params, int n, int inputs, int p0, int p1, int p2, int outs) {
        int i = kc.globalIdx;
        if (i < n) {
            float x0 = i0.get(i);
            float x1 = inputs > 1 ? i1.get(i) : 0.0f;
            float x2 = inputs > 2 ? i2.get(i) : 0.0f;
            float x3 = inputs > 3 ? i3.get(i) : 0.0f;
            float t0;
            {
                float x = reg((p0 >> 10) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p0 >> 14) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p0 >> 1) & 511;
                if ((p0 & 1) == UNARY) {
                    t0 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(0), params.get(1)) : unaryLow(op, x, params.get(0), params.get(1));
                } else {
                    t0 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(0), params.get(1)) : binaryLow(op, x, y, params.get(0), params.get(1));
                }
            }
            float t1;
            {
                float x = reg((p1 >> 10) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p1 >> 14) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p1 >> 1) & 511;
                if ((p1 & 1) == UNARY) {
                    t1 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(2), params.get(3)) : unaryLow(op, x, params.get(2), params.get(3));
                } else {
                    t1 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(2), params.get(3)) : binaryLow(op, x, y, params.get(2), params.get(3));
                }
            }
            float t2;
            {
                float x = reg((p2 >> 10) & 15, x0, x1, x2, x3, t0, t1, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p2 >> 14) & 15, x0, x1, x2, x3, t0, t1, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p2 >> 1) & 511;
                if ((p2 & 1) == UNARY) {
                    t2 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(4), params.get(5)) : unaryLow(op, x, params.get(4), params.get(5));
                } else {
                    t2 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(4), params.get(5)) : binaryLow(op, x, y, params.get(4), params.get(5));
                }
            }
            float t3 = 0.0f;
            float t4 = 0.0f;
            float t5 = 0.0f;
            store(i, outs, out0, out1, out2, out3, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5);
        }
    }

    /** fusedFlat with the program in scalars (see above), for a chain of 4 instructions. */
    public static void fusedProgram4(KernelContext kc, FloatArray i0, FloatArray i1, FloatArray i2, FloatArray i3, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3,
            FloatArray params, int n, int inputs, int p0, int p1, int p2, int p3, int outs) {
        int i = kc.globalIdx;
        if (i < n) {
            float x0 = i0.get(i);
            float x1 = inputs > 1 ? i1.get(i) : 0.0f;
            float x2 = inputs > 2 ? i2.get(i) : 0.0f;
            float x3 = inputs > 3 ? i3.get(i) : 0.0f;
            float t0;
            {
                float x = reg((p0 >> 10) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p0 >> 14) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p0 >> 1) & 511;
                if ((p0 & 1) == UNARY) {
                    t0 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(0), params.get(1)) : unaryLow(op, x, params.get(0), params.get(1));
                } else {
                    t0 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(0), params.get(1)) : binaryLow(op, x, y, params.get(0), params.get(1));
                }
            }
            float t1;
            {
                float x = reg((p1 >> 10) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p1 >> 14) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p1 >> 1) & 511;
                if ((p1 & 1) == UNARY) {
                    t1 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(2), params.get(3)) : unaryLow(op, x, params.get(2), params.get(3));
                } else {
                    t1 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(2), params.get(3)) : binaryLow(op, x, y, params.get(2), params.get(3));
                }
            }
            float t2;
            {
                float x = reg((p2 >> 10) & 15, x0, x1, x2, x3, t0, t1, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p2 >> 14) & 15, x0, x1, x2, x3, t0, t1, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p2 >> 1) & 511;
                if ((p2 & 1) == UNARY) {
                    t2 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(4), params.get(5)) : unaryLow(op, x, params.get(4), params.get(5));
                } else {
                    t2 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(4), params.get(5)) : binaryLow(op, x, y, params.get(4), params.get(5));
                }
            }
            float t3;
            {
                float x = reg((p3 >> 10) & 15, x0, x1, x2, x3, t0, t1, t2, 0.0f, 0.0f, 0.0f);
                float y = reg((p3 >> 14) & 15, x0, x1, x2, x3, t0, t1, t2, 0.0f, 0.0f, 0.0f);
                int op = (p3 >> 1) & 511;
                if ((p3 & 1) == UNARY) {
                    t3 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(6), params.get(7)) : unaryLow(op, x, params.get(6), params.get(7));
                } else {
                    t3 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(6), params.get(7)) : binaryLow(op, x, y, params.get(6), params.get(7));
                }
            }
            float t4 = 0.0f;
            float t5 = 0.0f;
            store(i, outs, out0, out1, out2, out3, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5);
        }
    }

    /** fusedFlat with the program in scalars (see above), for a chain of 5 instructions. */
    public static void fusedProgram5(KernelContext kc, FloatArray i0, FloatArray i1, FloatArray i2, FloatArray i3, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3,
            FloatArray params, int n, int inputs, int p0, int p1, int p2, int p3, int p4, int outs) {
        int i = kc.globalIdx;
        if (i < n) {
            float x0 = i0.get(i);
            float x1 = inputs > 1 ? i1.get(i) : 0.0f;
            float x2 = inputs > 2 ? i2.get(i) : 0.0f;
            float x3 = inputs > 3 ? i3.get(i) : 0.0f;
            float t0;
            {
                float x = reg((p0 >> 10) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p0 >> 14) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p0 >> 1) & 511;
                if ((p0 & 1) == UNARY) {
                    t0 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(0), params.get(1)) : unaryLow(op, x, params.get(0), params.get(1));
                } else {
                    t0 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(0), params.get(1)) : binaryLow(op, x, y, params.get(0), params.get(1));
                }
            }
            float t1;
            {
                float x = reg((p1 >> 10) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p1 >> 14) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p1 >> 1) & 511;
                if ((p1 & 1) == UNARY) {
                    t1 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(2), params.get(3)) : unaryLow(op, x, params.get(2), params.get(3));
                } else {
                    t1 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(2), params.get(3)) : binaryLow(op, x, y, params.get(2), params.get(3));
                }
            }
            float t2;
            {
                float x = reg((p2 >> 10) & 15, x0, x1, x2, x3, t0, t1, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p2 >> 14) & 15, x0, x1, x2, x3, t0, t1, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p2 >> 1) & 511;
                if ((p2 & 1) == UNARY) {
                    t2 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(4), params.get(5)) : unaryLow(op, x, params.get(4), params.get(5));
                } else {
                    t2 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(4), params.get(5)) : binaryLow(op, x, y, params.get(4), params.get(5));
                }
            }
            float t3;
            {
                float x = reg((p3 >> 10) & 15, x0, x1, x2, x3, t0, t1, t2, 0.0f, 0.0f, 0.0f);
                float y = reg((p3 >> 14) & 15, x0, x1, x2, x3, t0, t1, t2, 0.0f, 0.0f, 0.0f);
                int op = (p3 >> 1) & 511;
                if ((p3 & 1) == UNARY) {
                    t3 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(6), params.get(7)) : unaryLow(op, x, params.get(6), params.get(7));
                } else {
                    t3 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(6), params.get(7)) : binaryLow(op, x, y, params.get(6), params.get(7));
                }
            }
            float t4;
            {
                float x = reg((p4 >> 10) & 15, x0, x1, x2, x3, t0, t1, t2, t3, 0.0f, 0.0f);
                float y = reg((p4 >> 14) & 15, x0, x1, x2, x3, t0, t1, t2, t3, 0.0f, 0.0f);
                int op = (p4 >> 1) & 511;
                if ((p4 & 1) == UNARY) {
                    t4 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(8), params.get(9)) : unaryLow(op, x, params.get(8), params.get(9));
                } else {
                    t4 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(8), params.get(9)) : binaryLow(op, x, y, params.get(8), params.get(9));
                }
            }
            float t5 = 0.0f;
            store(i, outs, out0, out1, out2, out3, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5);
        }
    }

    /** fusedFlat with the program in scalars (see above), for a chain of 6 instructions. */
    public static void fusedProgram6(KernelContext kc, FloatArray i0, FloatArray i1, FloatArray i2, FloatArray i3, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3,
            FloatArray params, int n, int inputs, int p0, int p1, int p2, int p3, int p4, int p5, int outs) {
        int i = kc.globalIdx;
        if (i < n) {
            float x0 = i0.get(i);
            float x1 = inputs > 1 ? i1.get(i) : 0.0f;
            float x2 = inputs > 2 ? i2.get(i) : 0.0f;
            float x3 = inputs > 3 ? i3.get(i) : 0.0f;
            float t0;
            {
                float x = reg((p0 >> 10) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p0 >> 14) & 15, x0, x1, x2, x3, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p0 >> 1) & 511;
                if ((p0 & 1) == UNARY) {
                    t0 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(0), params.get(1)) : unaryLow(op, x, params.get(0), params.get(1));
                } else {
                    t0 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(0), params.get(1)) : binaryLow(op, x, y, params.get(0), params.get(1));
                }
            }
            float t1;
            {
                float x = reg((p1 >> 10) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p1 >> 14) & 15, x0, x1, x2, x3, t0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p1 >> 1) & 511;
                if ((p1 & 1) == UNARY) {
                    t1 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(2), params.get(3)) : unaryLow(op, x, params.get(2), params.get(3));
                } else {
                    t1 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(2), params.get(3)) : binaryLow(op, x, y, params.get(2), params.get(3));
                }
            }
            float t2;
            {
                float x = reg((p2 >> 10) & 15, x0, x1, x2, x3, t0, t1, 0.0f, 0.0f, 0.0f, 0.0f);
                float y = reg((p2 >> 14) & 15, x0, x1, x2, x3, t0, t1, 0.0f, 0.0f, 0.0f, 0.0f);
                int op = (p2 >> 1) & 511;
                if ((p2 & 1) == UNARY) {
                    t2 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(4), params.get(5)) : unaryLow(op, x, params.get(4), params.get(5));
                } else {
                    t2 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(4), params.get(5)) : binaryLow(op, x, y, params.get(4), params.get(5));
                }
            }
            float t3;
            {
                float x = reg((p3 >> 10) & 15, x0, x1, x2, x3, t0, t1, t2, 0.0f, 0.0f, 0.0f);
                float y = reg((p3 >> 14) & 15, x0, x1, x2, x3, t0, t1, t2, 0.0f, 0.0f, 0.0f);
                int op = (p3 >> 1) & 511;
                if ((p3 & 1) == UNARY) {
                    t3 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(6), params.get(7)) : unaryLow(op, x, params.get(6), params.get(7));
                } else {
                    t3 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(6), params.get(7)) : binaryLow(op, x, y, params.get(6), params.get(7));
                }
            }
            float t4;
            {
                float x = reg((p4 >> 10) & 15, x0, x1, x2, x3, t0, t1, t2, t3, 0.0f, 0.0f);
                float y = reg((p4 >> 14) & 15, x0, x1, x2, x3, t0, t1, t2, t3, 0.0f, 0.0f);
                int op = (p4 >> 1) & 511;
                if ((p4 & 1) == UNARY) {
                    t4 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(8), params.get(9)) : unaryLow(op, x, params.get(8), params.get(9));
                } else {
                    t4 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(8), params.get(9)) : binaryLow(op, x, y, params.get(8), params.get(9));
                }
            }
            float t5;
            {
                float x = reg((p5 >> 10) & 15, x0, x1, x2, x3, t0, t1, t2, t3, t4, 0.0f);
                float y = reg((p5 >> 14) & 15, x0, x1, x2, x3, t0, t1, t2, t3, t4, 0.0f);
                int op = (p5 >> 1) & 511;
                if ((p5 & 1) == UNARY) {
                    t5 = op >= ElementwiseKernels.SIN ? unaryHigh(op, x, params.get(10), params.get(11)) : unaryLow(op, x, params.get(10), params.get(11));
                } else {
                    t5 = op >= ElementwiseKernels.ATAN2 ? binaryHigh(op, x, y, params.get(10), params.get(11)) : binaryLow(op, x, y, params.get(10), params.get(11));
                }
            }
            store(i, outs, out0, out1, out2, out3, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5);
        }
    }

    private static void store(int i, int outs, FloatArray out0, FloatArray out1, FloatArray out2, FloatArray out3, float x0, float x1, float x2, float x3, float t0, float t1,
            float t2, float t3, float t4, float t5) {
        int count = outs & 15;
        out0.set(i, reg((outs >> 4) & 15, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5));
        if (count > 1) {
            out1.set(i, reg((outs >> 8) & 15, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5));
        }
        if (count > 2) {
            out2.set(i, reg((outs >> 12) & 15, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5));
        }
        if (count > 3) {
            out3.set(i, reg((outs >> 16) & 15, x0, x1, x2, x3, t0, t1, t2, t3, t4, t5));
        }
    }

    private static float reg(int k, float x0, float x1, float x2, float x3, float t0, float t1, float t2, float t3, float t4, float t5) {
        float v = t5;
        if (k == 0) {
            v = x0;
        } else if (k == 1) {
            v = x1;
        } else if (k == 2) {
            v = x2;
        } else if (k == 3) {
            v = x3;
        } else if (k == 4) {
            v = t0;
        } else if (k == 5) {
            v = t1;
        } else if (k == 6) {
            v = t2;
        } else if (k == 7) {
            v = t3;
        } else if (k == 8) {
            v = t4;
        }
        return v;
    }

    // each helper stays under TornadoVM's inlining limit; the kernel picks one per instruction

    private static float unaryHigh(int op, float x, float alpha, float beta) {
        return op >= ElementwiseKernels.IS_NAN ? ElementwiseKernels.unaryNaN(op, x, alpha, beta) : ElementwiseKernels.unaryMath(op, x, alpha, beta);
    }

    private static float unaryLow(int op, float x, float alpha, float beta) {
        return op >= ElementwiseKernels.ELU ? ElementwiseKernels.unaryActivation(op, x, alpha, beta) : ElementwiseKernels.unaryBase(op, x, alpha, beta);
    }

    private static float binaryHigh(int op, float x, float y, float alpha, float beta) {
        return op >= ElementwiseKernels.MAXIMUM ? ElementwiseKernels.binaryNaN(op, x, y, alpha, beta) : ElementwiseKernels.binaryMath(op, x, y, alpha, beta);
    }

    private static float binaryLow(int op, float x, float y, float alpha, float beta) {
        return op >= ElementwiseKernels.ELU_GRAD ? ElementwiseKernels.binaryActivationGrad(op, x, y, alpha, beta) : ElementwiseKernels.binaryBase(op, x, y, alpha, beta);
    }
}
