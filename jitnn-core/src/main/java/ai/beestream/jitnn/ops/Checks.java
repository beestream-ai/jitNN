package ai.beestream.jitnn.ops;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;

final class Checks {

    private Checks() {
    }

    static void f32(Tensor... ts) {
        for (Tensor t : ts) {
            // in half-precision inference FP16 inputs are converted when the operator is recorded
            if (t.dtype() != DType.F32 && !(t.dtype() == DType.F16 && ai.beestream.jitnn.Autocast.halfActivations())) {
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
