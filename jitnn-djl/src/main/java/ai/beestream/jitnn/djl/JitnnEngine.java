package ai.beestream.jitnn.djl;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.GradientCollector;

import ai.beestream.jitnn.engine.Engine.Backend;

/**
 * A DJL engine that runs DJL code on jitNN: NDArray operators become lazy jitNN operators,
 * compiled into one TornadoVM plan per step and replayed; DJL's blocks, losses, optimisers and
 * trainers work on top, and {@code Model.load} reads ONNX files through jitnn-onnx.
 *
 * <p>Its rank (0) makes it DJL's default engine when it is on the class path; set
 * {@code ai.djl.default_engine} to choose another. Operators jitNN lacks run on the engine named by
 * {@code -Djitnn.djl.fallback} (for example {@code PyTorch}), when one is given, else throw.
 */
public final class JitnnEngine extends Engine {

    public static final String ENGINE_NAME = "jitNN";
    static final int RANK = 0;

    private static volatile JitnnEngine instance;

    /**
     * Arrays that track gradients, so that a gradient collector can clear them; held weakly and by
     * identity (DJL arrays compare by contents).
     */
    private final List<WeakReference<JitnnNDArray>> trainable = new ArrayList<>();
    private final AtomicLong draws = new AtomicLong();
    private volatile long seed = 42;
    private volatile Engine alternative;
    private volatile boolean alternativeResolved;

    private JitnnEngine() {
        // DJL training loops run several distinct traces per step (data, forward and backward,
        // evaluators, optimiser updates): keep them all compiled
        if (System.getProperty("jitnn.plan.cache") == null) {
            System.setProperty("jitnn.plan.cache", "64");
        }
    }

    public static JitnnEngine getInstance() {
        if (instance == null) {
            synchronized (JitnnEngine.class) {
                if (instance == null) {
                    instance = new JitnnEngine();
                }
            }
        }
        return instance;
    }

    @Override
    public Engine getAlternativeEngine() {
        if (!alternativeResolved) {
            String name = System.getProperty("jitnn.djl.fallback", "").trim();
            alternative = name.isEmpty() || !Engine.hasEngine(name) ? null : Engine.getEngine(name);
            alternativeResolved = true;
        }
        return alternative;
    }

    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    @Override
    public int getRank() {
        return RANK;
    }

    @Override
    public String getVersion() {
        Package p = JitnnEngine.class.getPackage();
        String v = p == null ? null : p.getImplementationVersion();
        return v == null ? "0.1.0-SNAPSHOT" : v;
    }

    @Override
    public boolean hasCapability(String capability) {
        return false;
    }

    /** True when jitNN runs operators on a TornadoVM device (else on the host reference). */
    static boolean onDevice() {
        return ai.beestream.jitnn.engine.Engine.backend() == Backend.TORNADO;
    }

    @Override
    public Device defaultDevice() {
        return onDevice() ? Device.gpu() : Device.cpu();
    }

    @Override
    public int getGpuCount() {
        return onDevice() ? 1 : 0;
    }

    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        return new JitnnSymbolBlock((JitnnNDManager) manager);
    }

    @Override
    public Model newModel(String name, Device device) {
        return new JitnnModel(name, device);
    }

    @Override
    public NDManager newBaseManager() {
        return JitnnNDManager.getSystemManager().newSubManager();
    }

    @Override
    public NDManager newBaseManager(Device device) {
        return JitnnNDManager.getSystemManager().newSubManager(device);
    }

    @Override
    public GradientCollector newGradientCollector() {
        return new JitnnGradientCollector(this);
    }

    @Override
    public void setRandomSeed(int seed) {
        super.setRandomSeed(seed);
        this.seed = seed;
        draws.set(0);
    }

    /** A fresh seed for the next random array: the engine seed and a draw counter. */
    long nextSeed() {
        return seed * 0x9E3779B97F4A7C15L + draws.getAndIncrement();
    }

    void track(JitnnNDArray array) {
        synchronized (trainable) {
            for (WeakReference<JitnnNDArray> r : trainable) {
                if (r.get() == array) {
                    return;
                }
            }
            trainable.add(new WeakReference<>(array));
        }
    }

    /** After a backward pass: every trainable array's gradient into its persistent buffer. */
    void storeGradients() {
        synchronized (trainable) {
            for (WeakReference<JitnnNDArray> r : trainable) {
                JitnnNDArray a = r.get();
                if (a != null) {
                    a.storeGradient();
                }
            }
        }
    }

    void zeroGradients() {
        synchronized (trainable) {
            Iterator<WeakReference<JitnnNDArray>> it = trainable.iterator();
            while (it.hasNext()) {
                JitnnNDArray a = it.next().get();
                if (a == null) {
                    it.remove();
                } else {
                    a.clearGradient();
                }
            }
        }
    }

    @Override
    public String toString() {
        return ENGINE_NAME + ':' + getVersion() + (onDevice() ? " (TornadoVM)" : " (host reference)");
    }
}
