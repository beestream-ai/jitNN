package ai.beestream.jitnn.llm;

import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;

import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationResult;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelInfo;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.TextGenerationModel;
import ai.beestream.jitnn.gguf.GgufFile;

/**
 * A GGUF language model that generates text, run by jitllm (TornadoVM's LLM engine, with its own
 * tokenizers, KV cache and tuned kernels for Llama, Mistral, Qwen, Phi, Granite, Gemma and other
 * families). jitllm runs on TornadoVM's default device, which jitNN also uses when
 * {@code -Djitnn.device} is not set; TornadoVM's backend priorities choose it, for example
 * {@code -Dtornado.cuda.priority=100} for the CUDA device.
 *
 * <pre>{@code
 * try (Llm llm = Llm.load(Path.of("Llama-3.2-1B-Instruct-Q8_0.gguf"))) {
 *     String answer = llm.generate("What is the capital of France?", 64);
 * }
 * }</pre>
 *
 * <p>{@link #gguf()} gives the file's metadata and weights as jitNN tensors, and {@link #model()}
 * the jitllm model, for chat messages, tools, sampling settings and concurrent sessions.
 */
public final class Llm implements AutoCloseable {

    private final GgufFile gguf;
    private final LocalModel model;

    private Llm(GgufFile gguf, LocalModel model) {
        this.gguf = gguf;
        this.model = model;
    }

    /** Loads {@code file} with jitllm's default options. */
    public static Llm load(Path file) throws IOException {
        return load(file, ModelOptions.defaults());
    }

    /** Loads {@code file} with jitllm's {@code options}. */
    public static Llm load(Path file, ModelOptions options) throws IOException {
        GgufFile gguf = GgufFile.open(file);
        LocalModel model = LocalModels.load(file, options);
        if (!(model instanceof TextGenerationModel)) {
            model.close();
            throw new IllegalArgumentException(file + ": " + model.info().architecture() + " does not generate text");
        }
        return new Llm(gguf, model);
    }

    /** The greedy (temperature 0) continuation of {@code prompt}, at most {@code maxNewTokens} tokens. */
    public String generate(String prompt, int maxNewTokens) {
        return generate(GenerationRequest.builder().prompt(prompt).maxNewTokens(maxNewTokens).temperature(0f).build()).text();
    }

    /** {@link #generate(String, int)}, passing each piece of text to {@code onText} as it is produced. */
    public String generate(String prompt, int maxNewTokens, Consumer<String> onText) {
        return generate(GenerationRequest.builder().prompt(prompt).maxNewTokens(maxNewTokens).temperature(0f).onToken(onText).build()).text();
    }

    /** Runs {@code request} (sampling, stop sequences, chat messages, tools) in a new session. */
    public GenerationResult generate(GenerationRequest request) {
        try (GenerationSession session = ((TextGenerationModel) model).newSession()) {
            return session.generate(request);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("closing the generation session failed", e);
        }
    }

    /** The GGUF file: metadata, and weights as jitNN tensors. */
    public GgufFile gguf() {
        return gguf;
    }

    /** The jitllm model. */
    public LocalModel model() {
        return model;
    }

    public ModelInfo info() {
        return model.info();
    }

    @Override
    public void close() {
        try {
            model.close();
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("closing the model failed", e);
        }
    }
}
