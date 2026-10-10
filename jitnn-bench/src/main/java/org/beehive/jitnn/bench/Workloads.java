package org.beehive.jitnn.bench;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import org.beehive.jitnn.Autocast;
import org.beehive.jitnn.DType;
import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Node;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.ReLU;
import org.beehive.jitnn.nn.Sequential;
import org.beehive.jitnn.nn.models.Bert;
import org.beehive.jitnn.nn.models.CharRnn;
import org.beehive.jitnn.nn.models.GPT2;
import org.beehive.jitnn.nn.models.Llama;
import org.beehive.jitnn.nn.models.MobileNetV2;
import org.beehive.jitnn.nn.models.ResNet;
import org.beehive.jitnn.nn.models.UNet;
import org.beehive.jitnn.nn.models.Vae;
import org.beehive.jitnn.nn.optim.Adam;
import org.beehive.jitnn.nn.optim.Optimizer;
import org.beehive.jitnn.nn.optim.SGD;

/**
 * The reference workloads whose operators the kernel benchmark times: inference at the batch sizes
 * we report, and one training step (forward, backward, optimiser) of each training model. Each
 * workload builds its model, then {@link #capture} records one step's operators without running
 * them.
 */
public final class Workloads {

    /** A named step: builds what it needs once, then records one step each time it runs. */
    public record Workload(String name, Supplier<Runnable> setup) {
    }

    private Workloads() {
    }

    static float[] uniform(int n, long seed) {
        float[] v = new float[n];
        long x = seed * 0x9E3779B97F4A7C15L + 1;
        for (int i = 0; i < n; i++) {
            x ^= x << 13;
            x ^= x >>> 7;
            x ^= x << 17;
            v[i] = (float) ((x >>> 11) * 0x1.0p-53 * 2 - 1);
        }
        return v;
    }

    static int[] tokens(int n, int vocab) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = (int) ((i * 7919L + 13) % vocab);
        }
        return t;
    }

    static Runnable inference(Module m, Supplier<Tensor> input) {
        m.train(false);
        return () -> Grad.inference(() -> m.forward(input.get()));
    }

    static Runnable training(Module m, Optimizer opt, Supplier<Tensor> loss) {
        m.train(true);
        return () -> {
            Tensor l = loss.get();
            l.backward();
            opt.step();
        };
    }

    /** The step under autocast to FP16. */
    static Runnable half(Runnable step) {
        return () -> {
            try (Grad.Scope ignored = Autocast.enable(DType.F16)) {
                step.run();
            }
        };
    }

    static Tensor images(int b, int c, int h, int w) {
        return Tensor.of(uniform(b * c * h * w, b + c + h), b, c, h, w);
    }

    static Tensor labels(int b, int classes) {
        int[] y = new int[b];
        for (int i = 0; i < b; i++) {
            y[i] = i % classes;
        }
        return Tensor.of(y, b);
    }

    /** Language-model loss: logits [B, T, V] against the next tokens. */
    static Tensor lmLoss(Tensor logits, int b, int t, int vocab) {
        return F.crossEntropy(logits.reshape(b * t, vocab), Tensor.of(tokens(b * t, vocab), b * t));
    }

    public static List<Workload> all() {
        List<Workload> w = new ArrayList<>();
        for (int b : new int[] { 1, 32 }) {
            w.add(new Workload("resnet50_infer_b" + b, () -> inference(ResNet.resnet50(3, 1000, false, 1), () -> images(b, 3, 224, 224))));
        }
        for (int[] s : new int[][] { { 1, 128 }, { 32, 128 } }) {
            w.add(new Workload("bert_infer_b" + s[0] + "_s" + s[1], () -> {
                Bert m = new Bert(30522, 512, 768, 12, 12, 1);
                return inference(m, () -> Tensor.of(tokens(s[0] * s[1], 30522), s[0], s[1]));
            }));
        }
        // half precision (autocast to FP16), at the batch sizes the comparison with ONNX Runtime uses
        w.add(new Workload("resnet50_infer_b32_fp16", () -> half(inference(ResNet.resnet50(3, 1000, false, 1), () -> images(32, 3, 224, 224)))));
        w.add(new Workload("bert_infer_b32_s128_fp16", () -> {
            Bert m = new Bert(30522, 512, 768, 12, 12, 1);
            return half(inference(m, () -> Tensor.of(tokens(32 * 128, 30522), 32, 128)));
        }));
        w.add(new Workload("resnet18_train_b64", () -> {
            Module m = ResNet.resnet18(3, 10, true, 1);
            SGD opt = new SGD(m.parameters(), 0.1f, 0.9f, 5e-4f, false);
            return training(m, opt, () -> F.crossEntropy(m.forward(images(64, 3, 32, 32)), labels(64, 10)));
        }));
        w.add(new Workload("mobilenetv2_train_b64", () -> {
            Module m = MobileNetV2.mobilenetV2(3, 10, true, 0, 1);
            SGD opt = new SGD(m.parameters(), 0.1f, 0.9f, 5e-4f, false);
            return training(m, opt, () -> F.crossEntropy(m.forward(images(64, 3, 32, 32)), labels(64, 10)));
        }));
        w.add(new Workload("gpt2_train_b8_s256", () -> {
            GPT2 m = new GPT2(50257, 1024, 768, 12, 12, 1);
            Adam opt = new Adam(m.parameters(), 3e-4f);
            return training(m, opt, () -> lmLoss(m.forward(Tensor.of(tokens(8 * 256, 50257), 8, 256)), 8, 256, 50257));
        }));
        w.add(new Workload("llama_train_b4_s256", () -> {
            Llama m = new Llama(32000, 1024, 16, 8, 2816, 10000.0f, 1);
            Adam opt = new Adam(m.parameters(), 3e-4f);
            return training(m, opt, () -> lmLoss(m.forward(Tensor.of(tokens(4 * 256, 32000), 4, 256)), 4, 256, 32000));
        }));
        w.add(new Workload("bert_train_b8_s128", () -> {
            Bert m = new Bert(30522, 512, 768, 12, 12, 1);
            Adam opt = new Adam(m.parameters(), 1e-4f);
            return training(m, opt, () -> m.forward(Tensor.of(tokens(8 * 128, 30522), 8, 128)).square().mean());
        }));
        w.add(new Workload("unet_train_b32", () -> {
            UNet m = new UNet(1, 32, 1);
            Adam opt = new Adam(m.parameters(), 1e-3f);
            return training(m, opt, () -> F.mseLoss(m.forward(images(32, 1, 32, 32)), images(32, 1, 32, 32)));
        }));
        w.add(new Workload("vae_train_b128", () -> {
            Vae m = new Vae(784, 400, 20, 1);
            Adam opt = new Adam(m.parameters(), 1e-3f);
            return training(m, opt, () -> {
                Tensor x = Tensor.of(uniform(128 * 784, 5), 128, 784).sigmoid();
                return Vae.loss(m.run(x, Tensor.of(uniform(128 * 20, 6), 128, 20)), x);
            });
        }));
        for (boolean gru : new boolean[] { false, true }) {
            w.add(new Workload((gru ? "gru" : "lstm") + "_train_b32_s128", () -> {
                CharRnn m = new CharRnn(256, 128, 512, 2, gru, 1);
                Adam opt = new Adam(m.parameters(), 2e-3f);
                return training(m, opt, () -> lmLoss(m.forward(Tensor.of(tokens(32 * 128, 256), 32, 128)), 32, 128, 256));
            }));
        }
        w.add(new Workload("mlp_train_b256", () -> {
            Module m = new Sequential(new Linear(784, 1024, 1), new ReLU(), new Linear(1024, 1024, 2), new ReLU(), new Linear(1024, 10, 3));
            Adam opt = new Adam(m.parameters(), 1e-3f);
            return training(m, opt, () -> F.crossEntropy(m.forward(Tensor.of(uniform(256 * 784, 7), 256, 784)), labels(256, 10)));
        }));
        return w;
    }

    /** One step's operators of each workload (names matching {@code filter}, all when empty). */
    public static Map<String, List<Node>> capture(String filter) {
        Map<String, List<Node>> out = new LinkedHashMap<>();
        for (Workload w : all()) {
            if (!filter.isEmpty() && !java.util.Arrays.asList(filter.split(",")).contains(w.name())) {
                continue;
            }
            Runnable step = w.setup().get();
            out.put(w.name(), Engine.capture(step));
            Engine.reset();
        }
        return out;
    }
}
