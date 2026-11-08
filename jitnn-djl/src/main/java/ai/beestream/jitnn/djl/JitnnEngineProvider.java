package ai.beestream.jitnn.djl;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;

/** Registers {@link JitnnEngine} with DJL (through META-INF/services). */
public final class JitnnEngineProvider implements EngineProvider {

    @Override
    public String getEngineName() {
        return JitnnEngine.ENGINE_NAME;
    }

    @Override
    public int getEngineRank() {
        return JitnnEngine.RANK;
    }

    @Override
    public Engine getEngine() {
        return JitnnEngine.getInstance();
    }
}
