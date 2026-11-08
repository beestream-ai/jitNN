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
package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.nn.models.Doc2Vec;
import ai.beestream.jitnn.nn.models.GloVe;
import ai.beestream.jitnn.nn.models.Word2Vec;
import ai.beestream.jitnn.nn.optim.AdaGrad;
import ai.beestream.jitnn.nn.optim.Adam;
import ai.beestream.jitnn.nn.optim.Optimizer;
import ai.beestream.jitnn.nn.text.Corpus;
import ai.beestream.jitnn.nn.text.Vocabulary;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Gates for the embedding models on a corpus with planted structure: 4 topics of 10 words, every
 * sentence drawn from one topic. A trained model must place each word nearest to a word of its
 * own topic (and each document nearest to a document of its own topic) for at least 90% of them,
 * and its loss must fall.
 */
class EmbeddingModelsAccelTest {

    static final int TOPICS = 4;
    static final int WORDS = 10;
    static final int SENTENCES = 400;
    static final int LENGTH = 10;
    static final int DIM = 16;
    static final int BATCH = 256;
    static final int K = 5;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    /** Sentences, and the topic of each sentence. */
    static List<String[]> corpus(int[] topicOf) {
        SplittableRandom r = new SplittableRandom(7);
        List<String[]> out = new ArrayList<>();
        for (int s = 0; s < SENTENCES; s++) {
            int topic = s % TOPICS;
            topicOf[s] = topic;
            String[] words = new String[LENGTH];
            for (int i = 0; i < LENGTH; i++) {
                words[i] = "t" + topic + "w" + r.nextInt(WORDS);
            }
            out.add(words);
        }
        return out;
    }

    static int topic(String word) {
        return word.charAt(1) - '0';
    }

    static float wordPurity(Vocabulary v, float[] vectors, int dim) {
        int good = 0;
        for (int i = 0; i < v.size(); i++) {
            if (topic(v.word(Corpus.nearest(vectors, dim, i))) == topic(v.word(i))) {
                good++;
            }
        }
        return (float) good / v.size();
    }

    private static float mean(List<Float> xs, int from, int to) {
        float s = 0;
        for (int i = from; i < to; i++) {
            s += xs.get(i);
        }
        return s / (to - from);
    }

    private static void assertLearned(String name, List<Float> losses, float purity) {
        int n = losses.size();
        float first = mean(losses, 0, Math.min(5, n));
        float last = mean(losses, Math.max(0, n - 5), n);
        System.out.printf("[embed] %s: loss %.4f -> %.4f, purity %.3f (%d steps)%n", name, first, last, purity, n);
        assertTrue(last < 0.8f * first, name + ": loss " + first + " -> " + last);
        assertTrue(purity >= 0.9f, name + ": nearest-neighbour purity " + purity);
    }

    @Test
    void word2vec() {
        int[] topics = new int[SENTENCES];
        List<String[]> text = corpus(topics);
        Vocabulary vocab = Vocabulary.build(text, 1);
        for (Word2Vec.Mode mode : Word2Vec.Mode.values()) {
            Corpus c = new Corpus(vocab.encode(text), vocab.counts(), 11);
            Word2Vec model = new Word2Vec(vocab.size(), DIM, mode, 3);
            Optimizer opt = new Adam(model.parameters(), 0.02f);
            List<Float> losses = new ArrayList<>();
            int window = 2;
            for (int epoch = 0; epoch < 6; epoch++) {
                int[][] ex = mode == Word2Vec.Mode.SKIP_GRAM ? c.skipGramPairs(window, false) : c.cbowSamples(window);
                int[] centres = mode == Word2Vec.Mode.SKIP_GRAM ? ex[0] : ex[1];
                int[] contexts = mode == Word2Vec.Mode.SKIP_GRAM ? ex[1] : ex[2];
                int per = contexts.length / centres.length;
                for (int at = 0; at + BATCH <= centres.length; at += BATCH) {
                    Tensor loss = model.loss(Arrays.copyOfRange(centres, at, at + BATCH), Arrays.copyOfRange(contexts, at * per, (at + BATCH) * per), c.negatives(BATCH * K), K);
                    loss.backward();
                    opt.step();
                    losses.add(loss.item());
                    opt.zeroGrad();
                }
            }
            assertLearned("word2vec " + mode, losses, wordPurity(vocab, model.vectors(), DIM));
        }
    }

    @Test
    void doc2vec() {
        int[] topics = new int[SENTENCES];
        List<String[]> text = corpus(topics);
        Vocabulary vocab = Vocabulary.build(text, 1);
        for (Doc2Vec.Mode mode : Doc2Vec.Mode.values()) {
            Corpus c = new Corpus(vocab.encode(text), vocab.counts(), 13);
            Doc2Vec model = new Doc2Vec(vocab.size(), SENTENCES, DIM, mode, 5);
            Optimizer opt = new Adam(model.parameters(), 0.02f);
            List<Float> losses = new ArrayList<>();
            for (int epoch = 0; epoch < 12; epoch++) {
                int[] docs;
                int[] targets;
                int[] contexts = null;
                int per = 0;
                if (mode == Doc2Vec.Mode.PV_DBOW) {
                    int[][] p = c.skipGramPairs(1, true);
                    docs = p[0];
                    targets = p[1];
                } else {
                    int[][] p = c.cbowSamples(2);
                    docs = p[0];
                    targets = p[1];
                    contexts = p[2];
                    per = 4;
                }
                for (int at = 0; at + BATCH <= docs.length; at += BATCH) {
                    int[] ctx = contexts == null ? null : Arrays.copyOfRange(contexts, at * per, (at + BATCH) * per);
                    Tensor loss = model.loss(Arrays.copyOfRange(docs, at, at + BATCH), Arrays.copyOfRange(targets, at, at + BATCH), ctx, c.negatives(BATCH * K), K);
                    loss.backward();
                    opt.step();
                    losses.add(loss.item());
                    opt.zeroGrad();
                }
            }
            float[] v = model.documentVectors();
            int good = 0;
            for (int d = 0; d < SENTENCES; d++) {
                if (topics[Corpus.nearest(v, DIM, d)] == topics[d]) {
                    good++;
                }
            }
            assertLearned("doc2vec " + mode, losses, (float) good / SENTENCES);
        }
    }

    @Test
    void glove() {
        int[] topics = new int[SENTENCES];
        List<String[]> text = corpus(topics);
        Vocabulary vocab = Vocabulary.build(text, 1);
        Corpus c = new Corpus(vocab.encode(text), vocab.counts(), 17);
        Object[] x = c.cooccurrence(3);
        int[] rows = (int[]) x[0];
        int[] cols = (int[]) x[1];
        float[] vals = (float[]) x[2];
        GloVe model = new GloVe(vocab.size(), DIM, 3);
        Optimizer opt = new AdaGrad(model.parameters(), 0.1f, 0, 0, 0, 1e-10f);
        List<Float> losses = new ArrayList<>();
        for (int step = 0; step < 300; step++) {
            Tensor loss = model.loss(rows, cols, vals);
            loss.backward();
            opt.step();
            losses.add(loss.item());
            opt.zeroGrad();
        }
        assertLearned("glove", losses, wordPurity(vocab, model.vectors(), DIM));
    }
}
