package org.beehive.jitnn.ops;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;

final class Checks {

    private Checks() {
    }

    static void f32(Tensor... ts) {
        for (Tensor t : ts) {
            if (t.dtype() != DType.F32) {
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
