package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.models.Bert;
import ai.beestream.jitnn.nn.models.GPT2;
import ai.beestream.jitnn.nn.models.Llama;

/** Activation checkpointing checks shared by the reference and device tests. */
final class ActivationCheckpointRun {

    private ActivationCheckpointRun() {
    }

    /** Loss and gradients of one backward pass of a fresh model. */
    record Result(float loss, List<float[]> grads) {
    }

    static Result step(Module model, Function<Module, Tensor> loss) {
        Tensor l = loss.apply(model);
        l.backward();
        float v = l.item();
        List<float[]> g = model.parameters().stream().map(p -> p.grad() == null ? new float[0] : p.grad().toFloatArray()).toList();
        return new Result(v, g);
    }

    static Tensor tokens(int b, int t, int vocab, int salt) {
        int[] ids = new int[b * t];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = (i * 31 + salt) % vocab;
        }
        return Tensor.of(ids, b, t);
    }

    static Tensor lmLoss(Tensor logits, Tensor targets) {
        return F.crossEntropy(logits.reshape(-1, logits.dim(-1)), targets.reshape(targets.numel()));
    }

    /** The cases: name, a model factory (fresh, same weights each call), and its loss. */
    record Case(String name, Function<Boolean, Module> model, Function<Module, Tensor> loss) {
    }

    static List<Case> cases(int b, int t, int dim, int layers) {
        int vocab = 97;
        Tensor ids = tokens(b, t, vocab, 1);
        Tensor next = tokens(b, t, vocab, 2);
        return List.of(
                new Case("gpt2", ck -> new GPT2(vocab, t, dim, 4, layers, 3).checkpointing(ck), m -> lmLoss(m.forward(ids), next)),
                new Case("llama", ck -> new Llama(vocab, dim, 4, layers, 2 * dim, 10000.0f, 5).checkpointing(ck), m -> lmLoss(m.forward(ids), next)),
                new Case("bert", ck -> new Bert(vocab, t, dim, 4, layers, 2 * dim, 2, true, 7).checkpointing(ck), m -> {
                    Bert bert = (Bert) m;
                    float[] mask = new float[b * t];
                    for (int i = t - 3; i < t; i++) {
                        mask[i] = -10000.0f;
                    }
                    return bert.pooled(bert.encode(ids, null, Tensor.of(mask, b, 1, 1, t))).square().mean();
                }),
                // dropout inside the segment: the recomputation must draw the same mask
                new Case("dropout_mlp", ck -> new Segmented(ck, dim), m -> m.forward(Tensor.uniform(9, -1, 1, b * t, dim)).square().mean()));
    }

    /**
     * Two MLP segments with dropout, checkpointed or not. The dropout seeds come from a per-model call
     * counter (fixed streams), so a recomputation that did not replay them would draw other masks.
     */
    static final class Segmented extends Module {
        private final boolean ck;
        private final Linear a1;
        private final Linear a2;
        private final Linear b1;
        private final Linear b2;
        private int calls;

        Segmented(boolean ck, int dim) {
            this.ck = ck;
            this.a1 = child(new Linear(dim, 2 * dim, 11));
            this.a2 = child(new Linear(2 * dim, dim, 13));
            this.b1 = child(new Linear(dim, 2 * dim, 14));
            this.b2 = child(new Linear(2 * dim, dim, 16));
        }

        private Tensor dropout(Tensor x, int stream) {
            return F.dropout(x, 0.3f, Tensor.of(new int[] { calls++, 77 }, 2), stream);
        }

        @Override
        public Tensor forward(Tensor x) {
            Function<Tensor[], Tensor> a = in -> a2.forward(dropout(a1.forward(in[0]).relu(), 1));
            Function<Tensor[], Tensor> b = in -> b2.forward(dropout(b1.forward(in[0]).relu(), 2));
            Tensor h = ck ? Grad.checkpoint(a, x) : a.apply(new Tensor[] { x });
            return ck ? Grad.checkpoint(b, h) : b.apply(new Tensor[] { h });
        }
    }

    /** Checkpointed and plain runs give the same loss and gradients, within {@code tol} of each gradient's largest value. */
    static void same(Case c, Supplier<Result> plain, Supplier<Result> checkpointed, double tol) {
        Result p = plain.get();
        Result k = checkpointed.get();
        assertEquals(p.loss(), k.loss(), Math.abs(p.loss()) * tol + 1e-12, c.name() + " loss");
        assertEquals(p.grads().size(), k.grads().size());
        for (int i = 0; i < p.grads().size(); i++) {
            float[] a = p.grads().get(i);
            float[] g = k.grads().get(i);
            assertEquals(a.length, g.length, c.name() + " gradient " + i);
            double scale = 1e-30;
            double err = 0;
            for (int j = 0; j < a.length; j++) {
                scale = Math.max(scale, Math.abs(a[j]));
                err = Math.max(err, Math.abs(a[j] - g[j]));
            }
            assertTrue(err <= tol * scale, c.name() + " gradient " + i + ": max error " + err + " of " + scale);
        }
    }
}
