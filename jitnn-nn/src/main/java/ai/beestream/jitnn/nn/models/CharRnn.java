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
package ai.beestream.jitnn.nn.models;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Embedding;
import ai.beestream.jitnn.nn.GRU;
import ai.beestream.jitnn.nn.LSTM;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;

/**
 * A recurrent language model: token embedding, a stack of LSTM or GRU layers, a linear head over
 * the vocabulary. Inputs are time first, tokens [T, B]; logits [T, B, vocab]. Seeds: embedding
 * {@code seed} (uniform in [-0.1, 0.1)), recurrent layers {@code seed + 1 ..} (four per layer),
 * head {@code seed + 100}.
 */
public final class CharRnn extends Module {

    private final Embedding embedding;
    private final LSTM lstm;
    private final GRU gru;
    private final Linear head;

    /**
     * A model over {@code vocab} tokens: embeddings of {@code dim}, {@code layers} LSTM (or GRU, with
     * {@code useGru}) layers of {@code hidden} features, and the head.
     */
    public CharRnn(int vocab, int dim, int hidden, int layers, boolean useGru, long seed) {
        this.embedding = child(new Embedding(Tensor.uniform(seed, -0.1f, 0.1f, vocab, dim), -1));
        this.lstm = useGru ? null : child(new LSTM(dim, hidden, layers, seed + 1));
        this.gru = useGru ? child(new GRU(dim, hidden, layers, seed + 1)) : null;
        this.head = child(new Linear(hidden, vocab, seed + 100));
    }

    @Override
    public Tensor forward(Tensor tokens) {
        Tensor x = embedding.forward(tokens);
        Tensor y = lstm != null ? lstm.forward(x) : gru.forward(x);
        return head.forward(y);
    }
}
