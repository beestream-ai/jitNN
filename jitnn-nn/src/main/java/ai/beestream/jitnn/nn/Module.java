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
package ai.beestream.jitnn.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.io.StateDict;

/**
 * A differentiable building block with parameters and child modules.
 *
 * <p>A module owns its parameters and buffers, and its children's. {@link #close()} releases them
 * on the device. Training a module (forward and backward, {@code zeroGrad}, its optimiser) is for
 * one thread at a time; inference in evaluation mode ({@code Grad.inference}) may run from several
 * threads at once, sharing the weights.
 */
public abstract class Module implements AutoCloseable {

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

    private boolean closed;

    /**
     * Releases the parameters and buffers of this module and its children, their gradients, and
     * the weights jitNN derived from them (BatchNorm-folded and channels-last copies). Using the
     * module afterwards is an error. Plans compiled for it stay cached for a model of the same
     * structure, which reuses them without compiling, and are closed once idle
     * ({@code jitnn.plan.idleExecutions}) or by {@code JitnnRuntime.reset()}. Closing twice does
     * nothing.
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (Tensor p : parameters()) {
            Tensor g = p.grad();
            if (g != null) {
                g.release();
            }
            p.release();
        }
        for (Tensor b : buffers()) {
            b.release();
        }
    }

    /** Whether {@link #close()} was called. */
    public boolean isClosed() {
        return closed;
    }
}
