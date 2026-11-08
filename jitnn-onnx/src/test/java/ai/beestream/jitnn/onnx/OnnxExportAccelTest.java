package ai.beestream.jitnn.onnx;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.models.Bert;
import ai.beestream.jitnn.nn.models.GPT2;
import ai.beestream.jitnn.nn.models.ResNet;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Export on a device: the operators are captured from the device backend and the weights read
 * back from it. With {@code -Djitnn.hf=<dir>} (scripts/hf/hf_reference.py) also Hugging Face's
 * gpt2 and bert-base-uncased; {@code -Djitnn.onnx.exportDir} keeps the files for check_export.py.
 */
class OnnxExportAccelTest {

    @TempDir
    Path tmp;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void resnet50() throws IOException {
        Module m = ResNet.resnet50(3, 1000, false, 3).train(false);
        OnnxExportTest.check(tmp, "resnet50_device", x -> m.forward(x[0]), Tensor.uniform(4, -1, 1, 2, 3, 224, 224));
    }

    @Test
    void huggingFace() throws IOException {
        String d = System.getProperty("jitnn.hf", "");
        assumeTrue(!d.isEmpty() && Files.exists(Path.of(d, "gpt2")), "set -Djitnn.hf=<dir>");
        GPT2 gpt2 = GPT2.fromHuggingFace(Path.of(d, "gpt2"));
        OnnxExportTest.check(tmp, "hf_gpt2", x -> gpt2.forward(x[0]), OnnxExportTest.tokens(1, 32, 50257));
        Bert bert = Bert.fromHuggingFace(Path.of(d, "bert-base-uncased"));
        Tensor rows = Tensor.of(new int[] { 3, 40 }, 2);
        OnnxExportTest.check(tmp, "hf_bert_mlm", x -> bert.mlmLogits(x[0], null, x[1]), OnnxExportTest.tokens(2, 32, 30522), rows);
    }
}
