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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.io.Json;
import ai.beestream.jitnn.io.StateDict;

/**
 * A Hugging Face model directory: {@code config.json} and the weights in {@code model.safetensors}
 * or in shards listed by {@code model.safetensors.index.json}. Weights are looked up with or
 * without the architecture prefix ({@code bert.}, {@code transformer.}, {@code model.}), and old
 * LayerNorm names ({@code gamma}, {@code beta}) are accepted for {@code weight} and {@code bias}.
 * The model classes use it in their {@code fromHuggingFace} factories; it also serves custom models.
 */
public final class HfCheckpoint {

    private static final List<String> PREFIXES = List.of("", "bert.", "transformer.", "model.", "roberta.", "distilbert.");

    private final Path dir;
    private final Map<?, ?> config;
    private final Map<String, StateDict> files = new HashMap<>();
    private final Map<String, StateDict> byName = new HashMap<>();
    private final Set<String> used = new LinkedHashSet<>();

    private HfCheckpoint(Path dir) throws IOException {
        this.dir = dir;
        this.config = (Map<?, ?>) Json.parse(dir.resolve("config.json"));
        Path single = dir.resolve("model.safetensors");
        Path index = dir.resolve("model.safetensors.index.json");
        if (Files.exists(single)) {
            add(single);
        } else if (Files.exists(index)) {
            Map<?, ?> map = (Map<?, ?>) ((Map<?, ?>) Json.parse(index)).get("weight_map");
            for (Object file : new LinkedHashSet<>(map.values())) {
                add(dir.resolve((String) file));
            }
        } else {
            throw new IOException(dir + ": no model.safetensors or model.safetensors.index.json (PyTorch .bin files are not read)");
        }
    }

    private void add(Path file) {
        StateDict d = files.computeIfAbsent(file.toString(), f -> StateDict.load(file));
        for (String n : d.tensorNames()) {
            byName.put(n, d);
        }
    }

    /** Opens a model directory. */
    public static HfCheckpoint open(Path dir) {
        try {
            return new HfCheckpoint(dir);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** The directory the checkpoint was read from. */
    public Path directory() {
        return dir;
    }

    /** The parsed {@code config.json}. */
    public Map<?, ?> config() {
        return config;
    }

    /** The first of {@code keys} present in the config, as an int; throws when none is. */
    public int configInt(String... keys) {
        for (String k : keys) {
            if (config.get(k) instanceof Number n) {
                return n.intValue();
            }
        }
        throw new IllegalArgumentException("config.json has none of " + Arrays.toString(keys));
    }

    /** Config value {@code key} as a float, or {@code otherwise} when absent. */
    public float configFloat(String key, float otherwise) {
        return config.get(key) instanceof Number n ? n.floatValue() : otherwise;
    }

    /** Config value {@code key} as a string, or {@code otherwise} when absent or null. */
    public String configString(String key, String otherwise) {
        Object v = config.get(key);
        return v == null ? otherwise : String.valueOf(v);
    }

    /** True when the checkpoint holds weight {@code name} (under any accepted spelling). */
    public boolean has(String name) {
        return resolve(name) != null;
    }

    /** Weight {@code name} as FP32 values, checking its shape. */
    public float[] floats(String name, int... shape) {
        String key = resolve(name);
        if (key == null) {
            throw new IllegalArgumentException(dir + ": no weight " + name);
        }
        StateDict d = byName.get(key);
        if (!Arrays.equals(d.shape(key), shape)) {
            throw new IllegalArgumentException(key + ": shape " + Arrays.toString(d.shape(key)) + ", expected " + Arrays.toString(shape));
        }
        used.add(key);
        Tensor t = d.tensor(key);
        float[] v = t.toFloatArray();
        t.release();
        return v;
    }

    /** Copies weight {@code name} into parameter {@code target} of the same shape. */
    public void load(Tensor target, String name) {
        target.copyFrom(floats(name, target.shape().dims()));
    }

    /** Copies the transpose of 2-D weight {@code name} into {@code target} (GPT-2's Conv1D layers store [in, out]). */
    public void loadTransposed(Tensor target, String name) {
        int out = target.dim(0);
        int in = target.dim(1);
        float[] w = floats(name, in, out);
        float[] t = new float[w.length];
        for (int i = 0; i < in; i++) {
            for (int o = 0; o < out; o++) {
                t[o * in + i] = w[i * out + o];
            }
        }
        target.copyFrom(t);
    }

    /** Copies the concatenation, along the first axis, of {@code names} into {@code target}. */
    public void loadConcatenated(Tensor target, String... names) {
        int rows = target.dim(0) / names.length;
        int[] part = target.shape().dims().clone();
        part[0] = rows;
        float[] all = new float[target.numel()];
        int at = 0;
        for (String n : names) {
            float[] v = floats(n, part);
            System.arraycopy(v, 0, all, at, v.length);
            at += v.length;
        }
        target.copyFrom(all);
    }

    /** Weights of the checkpoint that no {@code load} call read, e.g. a head the model does not have. */
    public Set<String> unused() {
        Set<String> left = new LinkedHashSet<>(byName.keySet());
        left.removeAll(used);
        return left;
    }

    private String resolve(String name) {
        for (String spelling : new String[] { name, name.replace(".weight", ".gamma"), name.replace(".bias", ".beta") }) {
            for (String p : PREFIXES) {
                if (byName.containsKey(p + spelling)) {
                    return p + spelling;
                }
            }
        }
        return null;
    }
}
