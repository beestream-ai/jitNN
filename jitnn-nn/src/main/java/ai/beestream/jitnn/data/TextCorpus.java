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
package ai.beestream.jitnn.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A byte-level text corpus for language-model gates: every byte is a token (vocabulary 256).
 * The first 90% trains and the rest evaluates. Training windows are laid out in order without
 * shuffling, so another framework can reproduce the exact batches (scripts/pytorch/datasets.py).
 */
public final class TextCorpus {

    /** The vocabulary size: one token per byte value. */
    public static final int VOCAB = 256;

    private final int[] tokens;
    private final int split;

    private TextCorpus(int[] tokens) {
        this.tokens = tokens;
        this.split = tokens.length * 9 / 10;
    }

    /** The bytes of {@code file} as a corpus. */
    public static TextCorpus load(Path file) throws IOException {
        byte[] bytes = Files.readAllBytes(file);
        int[] tokens = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            tokens[i] = bytes[i] & 0xFF;
        }
        return new TextCorpus(tokens);
    }

    /** The number of tokens. */
    public int length() {
        return tokens.length;
    }

    /** Start of training row {@code r} of batch {@code b}: consecutive windows, wrapping around the training part. */
    public int trainStart(int b, int rows, int seq, int r) {
        long index = (long) b * rows + r;
        return (int) ((index * seq) % (split - seq - 1));
    }

    /** Fills inputs and next-token targets [rows, seq] for training batch {@code b}. */
    public void trainBatch(int b, int rows, int seq, int[] inputs, int[] targets) {
        for (int r = 0; r < rows; r++) {
            int start = trainStart(b, rows, seq, r);
            System.arraycopy(tokens, start, inputs, r * seq, seq);
            System.arraycopy(tokens, start + 1, targets, r * seq, seq);
        }
    }

    /** Number of non-overlapping evaluation windows of {@code seq} tokens. */
    public int evalWindows(int seq) {
        return (tokens.length - split - 1) / seq;
    }

    /** Evaluation window {@code w}: inputs and targets [seq]. */
    public void evalWindow(int w, int seq, int[] inputs, int[] targets, int at) {
        int start = split + w * seq;
        System.arraycopy(tokens, start, inputs, at, seq);
        System.arraycopy(tokens, start + 1, targets, at, seq);
    }

    /** Token {@code i}. */
    public int token(int i) {
        return tokens[i];
    }
}
