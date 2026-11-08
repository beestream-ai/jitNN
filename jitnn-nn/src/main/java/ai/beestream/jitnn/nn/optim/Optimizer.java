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
package ai.beestream.jitnn.nn.optim;

import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.io.StateDict;

/** Updates parameters in place from their gradients. */
public abstract class Optimizer {

    /** The parameters this optimiser updates. */
    protected final List<Tensor> params;
    /** The learning rate. */
    protected float lr;
    /** Steps taken (the bias corrections of the adaptive methods use it). */
    protected int t;

    /** An optimiser over {@code params} with learning rate {@code lr}. */
    protected Optimizer(List<Tensor> params, float lr) {
        this.params = List.copyOf(params);
        this.lr = lr;
        for (Tensor p : this.params) {
            if (!p.storage().isPersistent()) {
                throw new IllegalArgumentException("not a parameter: " + p);
            }
        }
    }

    /** Gradient statistics [norm, factor, skip] for the next step (see GradStats), or null. */
    private Tensor stats;
    /** A one-element factor applied to every gradient at the next step (a loss scaler's 1 / scale), or null. */
    private Tensor preScale;

    /** Records one update of every parameter that has a gradient. */
    public abstract void step();

    /**
     * Clips the gradients of this optimiser's parameters to a global L2 norm of at most
     * {@code maxNorm} at the next {@link #step()}, as PyTorch's {@code clip_grad_norm_}, and returns
     * the norm before clipping (after loss unscaling). One small reduction per gradient; the
     * gradients are scaled inside the update itself. Reading the norm is a sync point.
     */
    public Tensor clipGradNorm(float maxNorm) {
        stats = computeStats(maxNorm, true);
        return stats.narrow(0, 0, 1).reshape();
    }

    private Tensor computeStats(float maxNorm, boolean clip) {
        List<Tensor> grads = new java.util.ArrayList<>();
        for (Tensor p : params) {
            if (p.grad() != null) {
                grads.add(p.grad());
            }
        }
        return ai.beestream.jitnn.ops.GradStats.apply(grads, preScale, maxNorm, clip);
    }

    void setPreScale(Tensor factor) {
        this.preScale = factor;
    }

    /** The statistics the next update uses (computed now when a pre-scale is pending), or null; clears them. */
    protected Tensor takeStats() {
        Tensor s = stats;
        if (s == null && preScale != null) {
            s = computeStats(0.0f, false);
        }
        stats = null;
        preScale = null;
        lastStats = s;
        return s;
    }

    private Tensor lastStats;

    /** The statistics used by the last {@link #step()}, or null. */
    Tensor lastStats() {
        return lastStats;
    }

    /** Undoes the step counting of the last {@link #step()}, which a loss scaler skipped on the device. */
    void rewindStep() {
    }

    /** Clears the gradients of every parameter. */
    public void zeroGrad() {
        for (Tensor p : params) {
            p.zeroGrad();
        }
    }

    /** The current learning rate. */
    public float learningRate() {
        return lr;
    }

    /** Changes the learning rate; takes effect at the next step without recompiling. */
    public void setLearningRate(float lr) {
        this.lr = lr;
    }

    /** Per-parameter state by slot name, each array indexed like the parameters (null where none). */
    protected java.util.Map<String, Tensor[]> state() {
        return java.util.Map.of();
    }

    /** Scalar state beyond the step count and learning rate, for checkpoints. */
    protected void saveScalars(StateDict d) {
    }

    /** Restores what {@link #saveScalars} saved. */
    protected void loadScalars(StateDict d) {
    }

    /**
     * The optimiser's state for a checkpoint: its type, learning rate, step count and per-parameter
     * state ({@code state.<slot>.<parameter index>}), which refers to the live tensors. The
     * hyper-parameters are not included: build the optimiser with the same arguments before loading.
     */
    public StateDict stateDict() {
        StateDict d = new StateDict().put("type", getClass().getName()).put("lr", lr).put("step", t);
        saveScalars(d);
        new java.util.TreeMap<>(state()).forEach((slot, ts) -> {
            for (int i = 0; i < ts.length; i++) {
                if (ts[i] != null) {
                    d.put("state." + slot + "." + i, ts[i]);
                }
            }
        });
        return d;
    }

    /** Restores {@link #stateDict()} into this optimiser, in place; type and state shapes must match. */
    public void loadStateDict(StateDict d) {
        if (!d.value("type").equals(getClass().getName())) {
            throw new IllegalArgumentException("state of a " + d.value("type") + ", not a " + getClass().getName());
        }
        java.util.Set<String> expected = new java.util.HashSet<>();
        state().forEach((slot, ts) -> {
            for (int i = 0; i < ts.length; i++) {
                if (ts[i] != null) {
                    expected.add("state." + slot + "." + i);
                }
            }
        });
        if (!expected.equals(d.tensorNames())) {
            java.util.Set<String> missing = new java.util.TreeSet<>(expected);
            missing.removeAll(d.tensorNames());
            java.util.Set<String> extra = new java.util.TreeSet<>(d.tensorNames());
            extra.removeAll(expected);
            throw new IllegalArgumentException("optimiser state does not match: missing " + missing + ", unexpected " + extra);
        }
        // every entry is checked before any is written
        state().forEach((slot, ts) -> {
            for (int i = 0; i < ts.length; i++) {
                if (ts[i] != null) {
                    d.check("state." + slot + "." + i, ts[i]);
                }
            }
        });
        state().forEach((slot, ts) -> {
            for (int i = 0; i < ts.length; i++) {
                if (ts[i] != null) {
                    d.copyInto("state." + slot + "." + i, ts[i]);
                }
            }
        });
        lr = d.floatValue("lr");
        t = d.intValue("step");
        loadScalars(d);
    }
}
