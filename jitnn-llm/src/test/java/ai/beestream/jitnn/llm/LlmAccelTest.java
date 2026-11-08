/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;

import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
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
        JitnnRuntime.reset();
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
