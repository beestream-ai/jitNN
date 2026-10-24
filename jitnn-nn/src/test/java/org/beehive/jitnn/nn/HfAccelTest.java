package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.io.Json;
import org.beehive.jitnn.nn.models.Bert;
import org.beehive.jitnn.nn.models.GPT2;
import org.beehive.jitnn.nn.text.Tokenizer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Hugging Face gpt2 and bert-base-uncased against transformers and the Rust tokenizers library.
 * Needs {@code -Djitnn.hf=<dir>} prepared by scripts/hf/hf_reference.py.
 */
class HfAccelTest {

    private static Path dir;

    @BeforeAll
    static void data() {
        String d = System.getProperty("jitnn.hf", "");
        assumeTrue(!d.isEmpty() && Files.exists(Path.of(d, "reference.json")), "set -Djitnn.hf=<dir> (scripts/hf/hf_reference.py)");
        dir = Path.of(d);
    }

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void tokenizersMatch() throws IOException {
        Map<?, ?> all = (Map<?, ?>) Json.parse(dir.resolve("tokens.json"));
        for (String name : List.of("gpt2", "bert-base-uncased")) {
            Tokenizer tok = Tokenizer.load(dir.resolve(name));
            Map<?, ?> ref = (Map<?, ?>) all.get(name);
            for (Object o : (List<?>) ref.get("single")) {
                Map<?, ?> c = (Map<?, ?>) o;
                Tokenizer.Encoding e = tok.encode((String) c.get("text"));
                assertArrayEquals(ints(c.get("ids")), e.ids(), name + ": " + c.get("text"));
                assertArrayEquals(ints(c.get("type_ids")), e.typeIds(), name + " types: " + c.get("text"));
            }
            for (Object o : (List<?>) ref.get("pairs")) {
                Map<?, ?> c = (Map<?, ?>) o;
                Tokenizer.Encoding e = tok.encode((String) c.get("a"), (String) c.get("b"));
                assertArrayEquals(ints(c.get("ids")), e.ids(), name + ": " + c.get("a") + " | " + c.get("b"));
                assertArrayEquals(ints(c.get("type_ids")), e.typeIds(), name + " types: " + c.get("a"));
            }
            for (Object o : (List<?>) ref.get("decoded")) {
                Map<?, ?> c = (Map<?, ?>) o;
                assertEquals(c.get("text"), tok.decode(ints(c.get("ids")), true), name + " decode");
            }
        }
    }

    @Test
    void gpt2Logits() throws IOException {
        Map<?, ?> ref = (Map<?, ?>) ((Map<?, ?>) Json.parse(dir.resolve("reference.json"))).get("gpt2");
        int[] ids = ints(ref.get("ids"));
        GPT2 model = GPT2.fromHuggingFace(dir.resolve("gpt2"));
        float[] logits = model.forward(Tensor.of(ids, 1, ids.length)).toFloatArray();
        close(floats("gpt2_logits.f32"), logits, 2e-4f, "gpt2 logits");
    }

    @Test
    void bertOutputs() throws IOException {
        Map<?, ?> ref = (Map<?, ?>) ((Map<?, ?>) Json.parse(dir.resolve("reference.json"))).get("bert");
        Tokenizer tok = Tokenizer.load(dir.resolve("bert-base-uncased"));
        @SuppressWarnings("unchecked")
        Tokenizer.Batch batch = tok.encodeBatch((List<String>) ref.get("texts"), 512);
        int[] expectedIds = ((List<?>) ref.get("ids")).stream().flatMapToInt(r -> java.util.Arrays.stream(ints(r))).toArray();
        assertArrayEquals(expectedIds, batch.idArray());

        Bert model = Bert.fromHuggingFace(dir.resolve("bert-base-uncased"));
        Tensor hidden = model.encode(batch.ids(), batch.typeIds(), batch.additiveMask());
        float[] h = hidden.toFloatArray();
        float[] expected = floats("bert_hidden.f32");
        int dim = hidden.dim(2);
        // padded positions differ by design (HF masks with the most negative float)
        for (int i = 0; i < batch.attention().length; i++) {
            if (batch.attention()[i] == 0) {
                java.util.Arrays.fill(h, i * dim, (i + 1) * dim, 0);
                java.util.Arrays.fill(expected, i * dim, (i + 1) * dim, 0);
            }
        }
        close(expected, h, 2e-4f, "bert hidden");
        close(floats("bert_pooled.f32"), model.pooled(hidden).toFloatArray(), 2e-4f, "bert pooled");

        int row = ((Number) ref.get("mask_row")).intValue();
        float[] logits = model.mlmLogits(batch.ids(), batch.additiveMask(), Tensor.of(new int[] { row }, 1)).toFloatArray();
        close(floats("bert_mask_logits.f32"), logits, 2e-4f, "bert MLM logits");
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            best = logits[i] > logits[best] ? i : best;
        }
        assertEquals(ref.get("prediction"), tok.token(best));
    }

    /** Max |e - a| within {@code tol} x max |e|. */
    private static void close(float[] e, float[] a, float tol, String what) {
        assertEquals(e.length, a.length, what + " length");
        float max = 0;
        float err = 0;
        for (int i = 0; i < e.length; i++) {
            max = Math.max(max, Math.abs(e[i]));
            err = Math.max(err, Math.abs(e[i] - a[i]));
        }
        System.out.println("[hf] " + what + ": max error " + err + " of max " + max);
        assertTrue(err <= tol * max, what + ": max error " + err + " of max " + max);
    }

    private static float[] floats(String file) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(dir.resolve(file))).order(ByteOrder.LITTLE_ENDIAN);
        float[] f = new float[b.remaining() / 4];
        b.asFloatBuffer().get(f);
        return f;
    }

    private static int[] ints(Object list) {
        return ((List<?>) list).stream().mapToInt(x -> ((Number) x).intValue()).toArray();
    }
}
