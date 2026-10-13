package org.beehive.jitnn.bench;

import java.util.Map;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * An LSTM training step (forward, backward) at the shapes of DJL's MNIST LSTM example (784 steps
 * of 1 input, batch 32, 64 units) and two wider ones: step time and device time per kernel. Only
 * with {@code -Djitnn.lstmprobe=true}; run with {@code -Djitnn.profile=true -Djitnn.cudaGraph=false}
 * for the kernel breakdown.
 */
class LstmProbeAccelTest {

    @Test
    void probe() {
        Assumptions.assumeTrue(Boolean.getBoolean("jitnn.lstmprobe"));
        int[][] shapes = { { 784, 32, 1, 64 }, { 100, 32, 64, 256 }, { 50, 64, 128, 512 } };
        for (int[] s : shapes) {
            int t = s[0], b = s[1], in = s[2], h = s[3];
            Tensor x = Tensor.persistent(Tensor.uniform(1, -1, 1, t, b, in));
            float k = (float) (1 / Math.sqrt(h));
            Tensor wih = Tensor.parameter(Tensor.uniform(2, -k, k, 4 * h, in));
            Tensor whh = Tensor.parameter(Tensor.uniform(3, -k, k, 4 * h, h));
            Tensor bih = Tensor.parameter(Tensor.zeros(4 * h));
            Tensor bhh = Tensor.parameter(Tensor.zeros(4 * h));
            Tensor h0 = Tensor.zeros(b, h);
            Tensor c0 = Tensor.zeros(b, h);
            int runs = 20;
            long best = Long.MAX_VALUE;
            for (int r = 0; r < runs + 3; r++) {
                wih.zeroGrad();
                whh.zeroGrad();
                long t0 = System.nanoTime();
                Tensor loss = F.lstm(x, wih, whh, bih, bhh, h0, c0)[0].square().mean();
                loss.backward();
                Engine.sync(loss, whh.grad(), wih.grad());
                long dt = System.nanoTime() - t0;
                if (r >= 3) {
                    best = Math.min(best, dt);
                }
                if (r == 2) {
                    Engine.kernelBreakdown();
                }
            }
            System.out.printf("[lstmprobe] T=%d B=%d I=%d H=%d: step %.2f ms (best of %d)%n", t, b, in, h, best / 1e6, runs);
            Map<String, long[]> k2 = Engine.kernelBreakdown();
            k2.entrySet().stream().sorted((p, q) -> Long.compare(q.getValue()[1], p.getValue()[1])).limit(8)
                    .forEach(e -> System.out.printf("[lstmprobe]   %-40s %6d launches %9.3f ms/step%n", e.getKey(), e.getValue()[0] / runs, e.getValue()[1] / 1e6 / runs));
        }
    }
}
