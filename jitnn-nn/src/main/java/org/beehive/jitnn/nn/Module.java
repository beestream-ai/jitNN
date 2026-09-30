package org.beehive.jitnn.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.beehive.jitnn.Tensor;

/** A differentiable building block with parameters and child modules. */
public abstract class Module {

    private final List<Tensor> params = new ArrayList<>();
    private final List<Module> children = new ArrayList<>();
    private boolean training = true;

    public abstract Tensor forward(Tensor x);

    /** Registers a trainable parameter initialised from {@code init}. */
    protected Tensor parameter(Tensor init) {
        Tensor p = Tensor.parameter(init);
        params.add(p);
        return p;
    }

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

    public void zeroGrad() {
        for (Tensor p : parameters()) {
            p.zeroGrad();
        }
    }

    public Module train(boolean on) {
        training = on;
        for (Module m : children) {
            m.train(on);
        }
        return this;
    }

    public boolean isTraining() {
        return training;
    }

    public long parameterCount() {
        long n = 0;
        for (Tensor p : parameters()) {
            n += p.numel();
        }
        return n;
    }
}
