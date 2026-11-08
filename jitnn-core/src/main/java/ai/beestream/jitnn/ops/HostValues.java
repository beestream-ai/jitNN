package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

/** Host reads of any floating-point storage, for the reference implementations. */
final class HostValues {

    private HostValues() {
    }

    /** Stores v into a host FloatArray or HalfFloatArray. */
    static void set(Object array, int i, float v) {
        if (array instanceof uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray h) {
            h.set(i, new uk.ac.manchester.tornado.api.types.HalfFloat(v));
        } else {
            ((FloatArray) array).set(i, v);
        }
    }

    static float get(Object array, int i) {
        if (array instanceof FloatArray f) {
            return f.get(i);
        }
        if (array instanceof HalfFloatArray h) {
            return h.get(i).getFloat32();
        }
        return Float.intBitsToFloat(((int) ((ShortArray) array).get(i)) << 16);
    }

    /** Checks that a and b are both FP32, or both the same half-precision type. */
    static void gemmOperands(String op, Tensor a, Tensor b) {
        DType da = a.dtype();
        if (!da.isFloat() || da != b.dtype()) {
            throw new IllegalArgumentException(op + " needs two FP32 or two same-type half-precision operands, got " + a + " and " + b);
        }
    }

    static boolean isHalf(Tensor t) {
        return t.dtype() == DType.F16 || t.dtype() == DType.BF16;
    }
}
