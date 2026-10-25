package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.beehive.jitnn.Autocast;
import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.engine.Storage;
import org.beehive.jitnn.nn.models.GPT2;
import org.junit.jupiter.api.Test;

/** Diagnostic ({@code -Djitnn.castCensus=true}): the casts of one FP16-activation GPT-2 training step, by direction and producer. */
class CastCensusAccelTest {

    @Test
    void census() {
        assumeTrue(Boolean.getBoolean("jitnn.castCensus"), "-Djitnn.castCensus=true");
        GPT2 m = new GPT2(256, 128, 768, 12, 12, 1);
        Tensor ids = ActivationCheckpointRun.tokens(16, 128, 256, 1);
        Tensor next = ActivationCheckpointRun.tokens(16, 128, 256, 2);
        List<Node> nodes = Engine.capture(() -> {
            Tensor loss;
            try (var s = Autocast.enableHalf()) {
                loss = ActivationCheckpointRun.lmLoss(m.forward(ids), next);
            }
            loss.mul(1024.0f).backward();
        });
        Map<Storage, Node> producer = new IdentityHashMap<>();
        for (Node n : nodes) {
            for (Tensor t : n.outputs) {
                producer.put(t.storage(), n);
            }
        }
        Map<String, long[]> byKind = new TreeMap<>();
        for (Node n : nodes) {
            if (!n.op.name().equals("cast")) {
                continue;
            }
            Tensor in = n.in(0);
            Node p = producer.get(in.storage());
            String from = in.storage().isPersistent() ? "parameter" : p == null ? "leaf" : p.op.name();
            String consumer = "?";
            for (Node c : nodes) {
                for (Tensor t : c.inputs) {
                    if (t.storage() == n.out(0).storage()) {
                        consumer = c.op.name();
                    }
                }
            }
            String key = String.format("%s -> %s  from %-16s to %-16s", in.dtype(), n.out(0).dtype(), from, consumer);
            long[] a = byKind.computeIfAbsent(key, k -> new long[2]);
            a[0]++;
            a[1] += in.numel();
        }
        System.out.printf("[census] %d nodes%n", nodes.size());
        byKind.forEach((k, v) -> System.out.printf("[census] %-70s %4d casts %8.1f M values%n", k, v[0], v[1] / 1e6));
        Engine.reset();
    }
}
