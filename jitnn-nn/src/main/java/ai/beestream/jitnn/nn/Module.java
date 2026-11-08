package ai.beestream.jitnn.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.io.StateDict;

/** A differentiable building block with parameters and child modules. */
public abstract class Module {

    private final List<Tensor> params = new ArrayList<>();
    private final List<Module> children = new ArrayList<>();
    private boolean training = true;

    /** Applies the module to {@code x}. */
    public abstract Tensor forward(Tensor x);

    /** Registers a trainable parameter initialised from {@code init}. */
    protected Tensor parameter(Tensor init) {
        Tensor p = Tensor.parameter(init);
        params.add(p);
        return p;
    }

    private final List<Tensor> buffers = new ArrayList<>();

    /** Registers persistent state that is not trained (running statistics). */
    protected Tensor buffer(Tensor init) {
        Tensor b = Tensor.persistent(init);
        buffers.add(b);
        return b;
    }

    /** This module's buffers and its children's. */
    public List<Tensor> buffers() {
        List<Tensor> all = new ArrayList<>(buffers);
        for (Module m : children) {
            all.addAll(m.buffers());
        }
        return Collections.unmodifiableList(all);
    }

    /** Adds a child module; returns it. */
    protected <M extends Module> M child(M module) {
        children.add(module);
        return module;
    }

    /** This module's parameters and its children's, in registration order. */
    public List<Tensor> parameters() {
        List<Tensor> all = new ArrayList<>(params);
        for (Module m : children) {
            all.addAll(m.parameters());
        }
        return Collections.unmodifiableList(all);
    }

    /** Clears the gradients of every parameter. */
    public void zeroGrad() {
        for (Tensor p : parameters()) {
            p.zeroGrad();
        }
    }

    /**
     * Switches this module and its children to training mode ({@code on}) or evaluation mode; returns this
     * module.
     */
    public Module train(boolean on) {
        training = on;
        for (Module m : children) {
            m.train(on);
        }
        return this;
    }

    /** Whether the module is in training mode (the default). */
    public boolean isTraining() {
        return training;
    }

    /** The number of trainable values: the elements of every parameter. */
    public long parameterCount() {
        long n = 0;
        for (Tensor p : parameters()) {
            n += p.numel();
        }
        return n;
    }

    /**
     * The parameters ({@code parameters.<i>}) and buffers ({@code buffers.<i>}) in registration
     * order, referring to the live tensors: save it with {@link StateDict#save}. Names are positions,
     * so a checkpoint loads into a model built the same way.
     */
    public StateDict stateDict() {
        StateDict d = new StateDict();
        List<Tensor> ps = parameters();
        for (int i = 0; i < ps.size(); i++) {
            d.put("parameters." + i, ps.get(i));
        }
        List<Tensor> bs = buffers();
        for (int i = 0; i < bs.size(); i++) {
            d.put("buffers." + i, bs.get(i));
        }
        return d;
    }

    /**
     * Overwrites the parameters and buffers with those of {@code d}, in place, so compiled plans keep
     * using them. Every name, dtype and shape must match.
     */
    public void loadStateDict(StateDict d) {
        List<Tensor> ps = parameters();
        List<Tensor> bs = buffers();
        java.util.Set<String> expected = new java.util.LinkedHashSet<>();
        for (int i = 0; i < ps.size(); i++) {
            expected.add("parameters." + i);
        }
        for (int i = 0; i < bs.size(); i++) {
            expected.add("buffers." + i);
        }
        if (!expected.equals(d.tensorNames())) {
            throw new IllegalArgumentException("checkpoint has " + d.tensorNames().size() + " tensors, model has " + ps.size() + " parameters and " + bs.size() + " buffers");
        }
        // every entry is checked before any is written
        for (int i = 0; i < ps.size(); i++) {
            d.check("parameters." + i, ps.get(i));
        }
        for (int i = 0; i < bs.size(); i++) {
            d.check("buffers." + i, bs.get(i));
        }
        for (int i = 0; i < ps.size(); i++) {
            d.copyInto("parameters." + i, ps.get(i));
        }
        for (int i = 0; i < bs.size(); i++) {
            d.copyInto("buffers." + i, bs.get(i));
        }
    }
}
