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
