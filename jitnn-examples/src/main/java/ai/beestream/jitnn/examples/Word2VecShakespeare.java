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

import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.models.Word2Vec;
import ai.beestream.jitnn.nn.optim.Adam;
import ai.beestream.jitnn.nn.text.Corpus;
import ai.beestream.jitnn.nn.text.Vocabulary;

/**
 * Word vectors from raw text, after DL4J's Word2VecRawTextExample: skip-gram with negative
 * sampling over the words of Tiny Shakespeare, then the nearest neighbours of a few words.
 *
 * <pre>run.sh Word2VecShakespeare [epochs] [word...]</pre>
 */
public final class Word2VecShakespeare {

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 3);
        String[] probes = args.length > 1 ? Arrays.copyOfRange(args, 1, args.length) : new String[] { "king", "love", "night", "sword", "father" };
        int dim = 100;
        int batch = 1024;
        int k = 5;

        List<String[]> sentences = new ArrayList<>();
        for (String line : Files.readAllLines(Examples.shakespeare())) {
            String[] words = line.toLowerCase().replaceAll("[^a-z' ]", " ").trim().split("\\s+");
            if (words.length > 1) {
                sentences.add(words);
            }
        }
        Vocabulary vocab = Vocabulary.build(sentences, 5);
        Corpus corpus = new Corpus(vocab.encode(sentences), vocab.counts(), 11);
        Word2Vec w2v = new Word2Vec(vocab.size(), dim, Word2Vec.Mode.SKIP_GRAM, 3);
        Adam opt = new Adam(w2v.parameters(), 0.01f);
        System.out.printf("%,d sentences, %,d words in the vocabulary%n", sentences.size(), vocab.size());

        for (int epoch = 0; epoch < epochs; epoch++) {
            long t0 = System.nanoTime();
            int[][] pairs = corpus.skipGramPairs(5, false);
            float sum = 0;
            int steps = 0;
            for (int at = 0; at + batch <= pairs[0].length; at += batch) {
                Tensor loss = w2v.loss(Arrays.copyOfRange(pairs[0], at, at + batch), Arrays.copyOfRange(pairs[1], at, at + batch), corpus.negatives(batch * k), k);
                loss.backward();
                opt.step();
                sum += loss.item();
                opt.zeroGrad();
                steps++;
            }
            System.out.printf("epoch %d: %,d pairs, loss %.4f, %.1f s%n", epoch + 1, pairs[0].length, sum / steps, Examples.seconds(t0));
        }

        float[] vectors = w2v.vectors();
        for (String word : probes) {
            int id = vocab.id(word);
            if (id < 0) {
                System.out.println(word + ": not in the vocabulary");
                continue;
            }
            float[] similarity = new float[vocab.size()];
            for (int other = 0; other < vocab.size(); other++) {
                similarity[other] = other == id ? -2 : Corpus.cosine(vectors, dim, id, other);
            }
            StringBuilder s = new StringBuilder(word + ":");
            for (int other : Examples.topK(similarity, 6)) {
                s.append(String.format(" %s (%.2f)", vocab.word(other), similarity[other]));
            }
            System.out.println(s);
        }
    }
}
