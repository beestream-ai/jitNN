package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;

final class Checks {

    private Checks() {
    }

    static void f32(Tensor... ts) {
        for (Tensor t : ts) {
            // in half-precision inference FP16 inputs are converted when the operator is recorded
            if (t.dtype() != DType.F32 && !(t.dtype() == DType.F16 && org.beehive.jitnn.Autocast.halfActivations())) {
                throw new IllegalArgumentException("expected an F32 tensor, got " + t);
            }
        }
    }

    static void rank(Tensor t, int rank) {
        if (t.rank() != rank) {
            throw new IllegalArgumentException("expected rank " + rank + ", got " + t);
        }
    }
}
