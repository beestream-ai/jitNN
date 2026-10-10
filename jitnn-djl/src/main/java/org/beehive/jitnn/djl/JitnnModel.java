package org.beehive.jitnn.djl;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.types.DataType;

import org.beehive.jitnn.onnx.OnnxSession;

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
