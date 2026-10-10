package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Bitwise operations and integer remainders on INT32 tensors, with broadcasting through the same metadata as the binary element-wise kernels. */
public final class BitwiseKernels {

    public static final int AND = 0;
    public static final int OR = 1;
    public static final int XOR = 2;
    /** x << y. */
    public static final int SHIFT_LEFT = 3;
    /** x >> y, arithmetic (the sign is kept), as torch.bitwise_right_shift on signed integers. */
    public static final int SHIFT_RIGHT = 4;
    /** x - floor(x / y) y: the remainder with the sign of y (0 where y is 0). */
    public static final int REMAINDER = 5;
    /** x - trunc(x / y) y: the remainder with the sign of x (0 where y is 0). */
    public static final int FMOD = 6;

    private BitwiseKernels() {
    }

    public static int apply(int op, int x, int y) {
        int r;
        if (op == AND) {
            r = x & y;
        } else if (op == OR) {
            r = x | y;
        } else if (op == XOR) {
            r = x ^ y;
        } else if (op == SHIFT_LEFT) {
            r = x << y;
        } else if (op == SHIFT_RIGHT) {
            r = x >> y;
        } else if (y == 0) {
            r = 0;
        } else {
            r = x % y;
            if (op == REMAINDER && r != 0 && (r < 0) != (y < 0)) {
                r += y;
            }
        }
        return r;
    }

    /** meta = [rank, out dims, strides of a, strides of b] (0 strides broadcast); one thread per output. */
    public static void binary(KernelContext kc, IntArray a, IntArray b, IntArray out, IntArray meta, int op, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            int rank = meta.get(0);
            int rem = i;
            int ai = 0;
            int bi = 0;
            for (int d = rank - 1; d >= 0; d--) {
                int size = meta.get(1 + d);
                int c = rem % size;
                rem = rem / size;
                ai += c * meta.get(1 + rank + d);
                bi += c * meta.get(1 + 2 * rank + d);
            }
            out.set(i, apply(op, a.get(ai), b.get(bi)));
        }
    }

    /** ~x (not named "not": an alternative token in C++, which the CUDA backend compiles). */
    public static void complement(KernelContext kc, IntArray x, IntArray out, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            out.set(i, ~x.get(i));
        }
    }
}
