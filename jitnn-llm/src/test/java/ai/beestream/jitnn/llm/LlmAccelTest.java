package ai.beestream.jitnn.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Text generation from a GGUF model on the test device. Needs a model:
 * {@code -Djitnn.test.extra=-Djitnn.llm.model=<file.gguf>} (an instruction-tuned Llama or Qwen).
 */
class LlmAccelTest {

    private static final String MODEL = System.getProperty("jitnn.llm.model", "");

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void generatesGreedily() throws Exception {
        assumeTrue(!MODEL.isEmpty(), "no -Djitnn.llm.model");
        try (Llm llm = Llm.load(Path.of(MODEL))) {
            System.out.println("model: " + llm.info());
            StringBuilder streamed = new StringBuilder();
            String a = llm.generate("What is the capital of France? Answer in one word.", 24, streamed::append);
            String b = llm.generate("What is the capital of France? Answer in one word.", 24);
            System.out.println("generated: " + a.replace('\n', ' '));
            assertFalse(a.isBlank());
            assertTrue(a.contains("Paris"), a);
            assertEquals(a, b, "greedy generation is deterministic");
            assertEquals(a, streamed.toString(), "streamed pieces add up to the text");
        }
    }

    @Test
    void weightsAreJitnnTensorsToo() throws Exception {
        assumeTrue(!MODEL.isEmpty(), "no -Djitnn.llm.model");
        try (Llm llm = Llm.load(Path.of(MODEL))) {
            String name = llm.gguf().tensorInfos().keySet().stream().filter(n -> n.endsWith("attn_q.weight")).findFirst().orElseThrow();
            Tensor wq = llm.gguf().tensor(name);
            int dim = llm.gguf().integer(llm.gguf().architecture() + ".embedding_length");
            assertEquals(dim, wq.dim(1));
            assertTrue(Float.isFinite(wq.toFloatArray()[0]));
        }
    }
}
