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
package ai.beestream.jitnn.visrec;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import ai.beestream.jitnn.io.Json;
import ai.beestream.jitnn.nn.Conv2d;
import ai.beestream.jitnn.nn.Flatten;
import ai.beestream.jitnn.nn.LeakyReLU;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.MaxPool2d;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.ReLU;
import ai.beestream.jitnn.nn.Sequential;
import ai.beestream.jitnn.nn.Sigmoid;
import ai.beestream.jitnn.nn.Tanh;

/**
 * A network described in Deep Netts' JSON architecture format, the format JSR 381's reference
 * implementation reads for {@code NeuralNetImageClassifier.builder().networkArchitecture(path)}:
 * {@code {"networkType": "CONVOLUTIONAL" | "FEEDFORWARD", "layers": [...]}} with layers of type
 * {@code INPUT} (width, height, channels), {@code CONVOLUTIONAL} (filterWidth, filterHeight, stride,
 * channels, activation; padded by (filter - 1) / 2 as Deep Netts does), {@code MAXPOOLING}
 * (filterWidth, filterHeight, stride), {@code FULLY_CONNECTED} (width, activation) and {@code OUTPUT}
 * (width, activation SIGMOID or SOFTMAX). The model ends in the output layer's logits; {@link
 * #softmax} says how to turn them into scores.
 */
final class DeepNettsArchitecture {

    /** The jitNN model, from the input to the output layer's logits. */
    final Module model;
    final int width;
    final int height;
    final int channels;
    final int outputs;
    /** True for a SOFTMAX output layer (scores sum to 1), false for SIGMOID (independent scores). */
    final boolean softmax;
    /** The JSON text, kept so that an exported model can be rebuilt. */
    final String json;

    private DeepNettsArchitecture(Module model, int width, int height, int channels, int outputs, boolean softmax, String json) {
        this.model = model;
        this.width = width;
        this.height = height;
        this.channels = channels;
        this.outputs = outputs;
        this.softmax = softmax;
        this.json = json;
    }

    static DeepNettsArchitecture read(Path file) throws IOException {
        return parse(java.nio.file.Files.readString(file));
    }

    @SuppressWarnings("unchecked")
    static DeepNettsArchitecture parse(String json) throws IOException {
        Map<String, Object> root = (Map<String, Object>) Json.parse(json);
        List<Object> layers = (List<Object>) root.get("layers");
        if (layers == null || layers.isEmpty()) {
            throw new IOException("network architecture has no layers");
        }
        List<Module> modules = new ArrayList<>();
        int w = 0;
        int h = 1;
        int c = 1;
        boolean flat = false;
        int outputs = -1;
        boolean softmax = false;
        int inW = 0;
        int inH = 0;
        int inC = 0;
        long seed = 1;
        for (Object o : layers) {
            Map<String, Object> l = (Map<String, Object>) o;
            String type = string(l, "layerType");
            switch (type) {
                case "INPUT" -> {
                    w = integer(l, "width");
                    h = l.containsKey("height") ? integer(l, "height") : 1;
                    c = l.containsKey("channels") ? integer(l, "channels") : 1;
                    flat = !l.containsKey("height");
                    inW = w;
                    inH = h;
                    inC = c;
                }
                case "CONVOLUTIONAL" -> {
                    int f = integer(l, "filterWidth");
                    int stride = l.containsKey("stride") ? integer(l, "stride") : 1;
                    int out = integer(l, "channels");
                    int pad = (f - 1) / 2;
                    modules.add(new Conv2d(c, out, f, stride, pad, seed++));
                    activation(modules, string(l, "activation"));
                    w = (w + 2 * pad - f) / stride + 1;
                    h = (h + 2 * pad - f) / stride + 1;
                    c = out;
                }
                case "MAXPOOLING" -> {
                    int f = integer(l, "filterWidth");
                    int stride = l.containsKey("stride") ? integer(l, "stride") : f;
                    modules.add(new MaxPool2d(f, stride, 0));
                    w = (w - f) / stride + 1;
                    h = (h - f) / stride + 1;
                }
                case "FULLY_CONNECTED", "OUTPUT" -> {
                    if (!flat) {
                        modules.add(new Flatten());
                        w = w * h * c;
                        h = 1;
                        c = 1;
                        flat = true;
                    }
                    int out = integer(l, "width");
                    modules.add(new Linear(w, out, seed++));
                    w = out;
                    if (type.equals("OUTPUT")) {
                        outputs = out;
                        String a = l.containsKey("activation") ? string(l, "activation") : "SOFTMAX";
                        softmax = a.equals("SOFTMAX");
                        if (!softmax && !a.equals("SIGMOID")) {
                            throw new IOException("output activation " + a + ": only SIGMOID and SOFTMAX are supported");
                        }
                    } else {
                        activation(modules, l.containsKey("activation") ? string(l, "activation") : "SIGMOID");
                    }
                }
                default -> throw new IOException("layer type " + type + " is not supported");
            }
        }
        if (outputs < 0) {
            throw new IOException("network architecture has no OUTPUT layer");
        }
        return new DeepNettsArchitecture(new Sequential(modules.toArray(new Module[0])), inW, inH, inC, outputs, softmax, json);
    }

    private static void activation(List<Module> modules, String name) throws IOException {
        switch (name) {
            case "RELU" -> modules.add(new ReLU());
            case "TANH" -> modules.add(new Tanh());
            case "SIGMOID" -> modules.add(new Sigmoid());
            case "LEAKY_RELU" -> modules.add(new LeakyReLU());
            case "LINEAR" -> {
            }
            default -> throw new IOException("activation " + name + " is not supported");
        }
    }

    private static String string(Map<String, Object> l, String key) throws IOException {
        Object v = l.get(key);
        if (v == null) {
            throw new IOException("layer " + l + " has no " + key);
        }
        return v.toString().toUpperCase(Locale.ROOT);
    }

    private static int integer(Map<String, Object> l, String key) throws IOException {
        Object v = l.get(key);
        if (!(v instanceof Number n)) {
            throw new IOException("layer " + l + " has no number " + key);
        }
        return n.intValue();
    }
}
