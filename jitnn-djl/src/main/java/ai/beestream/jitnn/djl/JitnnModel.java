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
package ai.beestream.jitnn.djl;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.stream.Stream;

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.Pair;

import ai.beestream.jitnn.onnx.OnnxSession;

/**
 * A DJL model on jitNN. {@code load} reads an ONNX file ({@code <prefix>.onnx} in the model
 * directory, the file itself, or the directory's only {@code .onnx} file) through jitnn-onnx; the
 * option {@code "fp16" = "true"} runs its products in half precision. A model whose block was set
 * in Java loads DJL's own parameter files instead.
 */
public final class JitnnModel extends BaseModel {

    JitnnModel(String name, Device device) {
        super(name);
        manager = JitnnEngine.getInstance().newBaseManager(device);
        manager.setName("jitnnModel");
        dataType = DataType.FLOAT32;
    }

    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options) throws IOException, MalformedModelException {
        setModelDir(Files.isRegularFile(modelPath) ? modelPath.toAbsolutePath().getParent() : modelPath);
        if (prefix == null) {
            prefix = modelName;
        }
        if (block != null) {
            loadBlock(prefix, options);
            wasLoaded = true;
            return;
        }
        Path file = onnxFile(modelPath, prefix);
        boolean half = options != null && Boolean.parseBoolean(String.valueOf(options.get("fp16")));
        block = new JitnnSymbolBlock((JitnnNDManager) manager, OnnxSession.load(file, half));
        wasLoaded = true;
    }

    /**
     * Releases the model's weights before closing it, so that loading the same model again reuses
     * the plans compiled for this one.
     */
    @Override
    public void close() {
        if (block instanceof JitnnSymbolBlock symbol && symbol.session() != null) {
            symbol.session().close();
        } else if (block != null) {
            for (Pair<String, Parameter> p : block.getParameters()) {
                if (p.getValue().isInitialized() && p.getValue().getArray() instanceof JitnnNDArray a && a.tensor().storage().isPersistent()) {
                    a.tensor().release();
                }
            }
        }
        super.close();
    }

    @Override
    public Trainer newTrainer(TrainingConfig config) {
        if (block == null) {
            throw new IllegalStateException("set a block on the model before creating a trainer");
        }
        if (wasLoaded) {
            block.freezeParameters(false);
        }
        for (Pair<Initializer, Predicate<Parameter>> pair : config.getInitializers()) {
            if (pair.getKey() != null && pair.getValue() != null) {
                block.setInitializer(pair.getKey(), pair.getValue());
            }
        }
        return new Trainer(this, config);
    }

    private Path onnxFile(Path modelPath, String prefix) throws IOException {
        if (Files.isRegularFile(modelPath)) {
            return modelPath;
        }
        Path named = modelDir.resolve(prefix.endsWith(".onnx") ? prefix : prefix + ".onnx");
        if (Files.isRegularFile(named)) {
            return named;
        }
        try (Stream<Path> files = Files.list(modelDir)) {
            List<Path> onnx = files.filter(p -> p.toString().endsWith(".onnx")).toList();
            if (onnx.size() == 1) {
                return onnx.get(0);
            }
            throw new IOException("no " + prefix + ".onnx in " + modelDir + (onnx.isEmpty() ? "" : ", and several other .onnx files: " + onnx));
        }
    }
}
