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
package ai.beestream.jitnn.examples;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.models.Bert;
import ai.beestream.jitnn.nn.text.Tokenizer;

/**
 * Fills in masked words with BERT (bert-base-uncased from Hugging Face), the masked-language-model
 * task BERT was trained on. The weights (440 MB) and tokenizer are downloaded on first use. All
 * sentences run as one padded batch, and the MLM head runs only on the masked positions.
 *
 * <pre>run.sh BertFillMask ["a sentence with [MASK] in it" ...]</pre>
 */
public final class BertFillMask {

    public static void main(String[] args) throws Exception {
        List<String> sentences = args.length > 0 ? List.of(args)
                : List.of("The capital of France is [MASK].", "I went to the [MASK] to buy some bread.", "The [MASK] barked at the mailman all night.",
                        "Java is a [MASK] language.");
        Path dir = Examples.huggingFace("bert-base-uncased", "config.json", "model.safetensors", "tokenizer.json");
        Bert bert = Bert.fromHuggingFace(dir);
        bert.train(false);
        Tokenizer tok = Tokenizer.load(dir);
        int maskId = tok.id("[MASK]");

        Tokenizer.Batch batch = tok.encodeBatch(sentences, 128);
        List<Integer> rows = new ArrayList<>();          // flat positions of the masks in [batch, length]
        for (int i = 0; i < batch.idArray().length; i++) {
            if (batch.idArray()[i] == maskId) {
                rows.add(i);
            }
        }
        int[] at = rows.stream().mapToInt(Integer::intValue).toArray();
        long t0 = System.nanoTime();
        float[] logits = Grad.inference(() -> bert.mlmLogits(batch.ids(), batch.additiveMask(), Tensor.of(at, at.length))).toFloatArray();
        System.out.printf("%d sentences, %d masks, %.2f s (includes compiling)%n%n", sentences.size(), at.length, Examples.seconds(t0));

        int vocab = tok.vocabSize();
        for (int m = 0; m < at.length; m++) {
            float[] p = Examples.softmax(logits, m * vocab, vocab, 1);
            StringBuilder s = new StringBuilder(sentences.get(at[m] / batch.length()) + "\n ");
            for (int id : Examples.topK(p, 5)) {
                s.append(String.format(" %s (%.0f%%)", tok.token(id), 100 * p[id]));
            }
            System.out.println(s);
        }
    }
}
