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
package ai.beestream.jitnn;

import ai.beestream.jitnn.engine.Node;

/** Whether new operators record what autograd needs. On by default, per thread. */
public final class Grad {

    private static final ThreadLocal<Boolean> ENABLED = ThreadLocal.withInitial(() -> Boolean.TRUE);

    private Grad() {
    }

    /** Whether operators recorded now on this thread record what autograd needs. */
    public static boolean isEnabled() {
        return ENABLED.get();
    }

    /** Enables gradient recording until the returned scope is closed. */
    static Scope enabled() {
        boolean previous = ENABLED.get();
        ENABLED.set(Boolean.TRUE);
        return () -> ENABLED.set(previous);
    }

    /**
     * Activation checkpointing, as PyTorch's {@code torch.utils.checkpoint}: runs {@code segment} on
     * {@code inputs} and keeps none of its activations for the backward pass, which runs the segment
     * again to get them. The segment's intermediates are freed after their last forward reader, so a
     * training step holds the activations of one segment at a time instead of all of them, for about
     * one more forward pass of compute. Gradients reach the inputs and the parameters the segment
     * uses. Dropout draws the same masks when the segment runs again; BatchNorm in training mode
     * updates its running statistics twice, as in PyTorch.
     *
     * <pre>{@code
     * for (Block b : blocks) {
     *     x = Grad.checkpoint(in -> b.forward(in[0]), x);
     * }
     * }</pre>
     */
    public static Tensor checkpoint(java.util.function.Function<Tensor[], Tensor> segment, Tensor... inputs) {
        if (!isEnabled()) {
            return segment.apply(inputs);
        }
        java.util.ArrayDeque<Tensor> seeds = new java.util.ArrayDeque<>();
        Tensor y;
        try (AutoCloseable internal = ai.beestream.jitnn.engine.Storage.internalScope(); AutoCloseable recording = ai.beestream.jitnn.ops.Dropout.recordSeeds(seeds)) {
            y = segment.apply(inputs);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        y.storage().markVisible();
        if (!y.requiresGrad()) {
            return y;
        }
        // the leaves the segment reads (parameters) and its inputs: the checkpoint node's inputs
        java.util.Set<Tensor> boundary = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        boundary.addAll(java.util.Arrays.asList(inputs));
        java.util.Set<Tensor> leaves = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        java.util.Set<Node> seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        java.util.ArrayDeque<Node> todo = new java.util.ArrayDeque<>();
        todo.add(y.gradFn());
        while (!todo.isEmpty()) {
            Node n = todo.poll();
            if (!seen.add(n)) {
                continue;
            }
            for (Tensor t : n.inputs) {
                if (boundary.contains(t) || !t.requiresGrad()) {
                    continue;
                }
                if (t.gradFn() == null) {
                    leaves.add(t);
                } else {
                    todo.add(t.gradFn());
                }
            }
        }
        java.util.List<Tensor> in = new java.util.ArrayList<>();
        for (Tensor t : inputs) {
            if (t.requiresGrad()) {
                in.add(t);
            }
        }
        in.addAll(leaves);
        Node node = new Node(new Recompute(segment, inputs, seeds), in.toArray(new Tensor[0]), new Tensor[] { y }, null, null, false);
        y.setGradFn(node);
        return y;
    }

    /** The backward of a checkpointed segment: run it again with gradients, then backward through it. */
    private record Recompute(java.util.function.Function<Tensor[], Tensor> segment, Tensor[] inputs, java.util.ArrayDeque<Tensor> seeds) implements ai.beestream.jitnn.engine.Op {

        @Override
        public String name() {
            return "checkpoint";
        }

        @Override
        public void emit(ai.beestream.jitnn.engine.Emitter e, Node n) {
            throw new IllegalStateException("a checkpoint node is never traced");
        }

        @Override
        public void reference(Node n) {
            throw new IllegalStateException("a checkpoint node is never traced");
        }

        @Override
        public Tensor[] backward(Node n, Tensor gy) {
            Tensor[] xs = new Tensor[inputs.length];
            for (int i = 0; i < xs.length; i++) {
                xs[i] = inputs[i].requiresGrad() ? inputs[i].detach().requiresGrad(true) : inputs[i];
            }
            try (Scope on = enabled(); AutoCloseable replay = ai.beestream.jitnn.ops.Dropout.replaySeeds(new java.util.ArrayDeque<>(seeds))) {
                Tensor y = segment.apply(xs);
                Autograd.backward(y, gy);
            } catch (RuntimeException e) {
                throw e;
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
            // the parameters got their gradients from the inner backward pass; the inputs' are returned
            Tensor[] gx = new Tensor[n.inputs.length];
            int k = 0;
            for (int i = 0; i < inputs.length; i++) {
                if (inputs[i].requiresGrad()) {
                    gx[k++] = xs[i].grad();
                }
            }
            return gx;
        }
    }

    /** Disables gradient recording until the returned scope is closed. */
    public static Scope disabled() {
        boolean previous = ENABLED.get();
        ENABLED.set(Boolean.FALSE);
        return () -> ENABLED.set(previous);
    }

    /**
     * Runs a forward pass for inference: no gradient recording, and every tensor created inside
     * except the returned one is internal, so its buffer is reused as soon as the computation no
     * longer needs it. Reading those intermediates after the call is not supported.
     */
    public static Tensor inference(java.util.function.Supplier<Tensor> forward) {
        Tensor out;
        try (Scope ignored = disabled(); AutoCloseable internal = ai.beestream.jitnn.engine.Storage.internalScope()) {
            out = forward.get();
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
        out.storage().markVisible();
        return out;
    }

    /**
     * Rescales the gradients of {@code params} so that their global L2 norm is at most
     * {@code maxNorm}, as PyTorch's {@code clip_grad_norm_}: every gradient is multiplied by
     * min(1, maxNorm / (norm + 1e-6)). Runs lazily on the device with the rest of the step and
     * returns the norm before clipping as a scalar tensor (reading it is a sync point).
     */
    public static Tensor clipNorm(java.util.List<Tensor> params, float maxNorm) {
        java.util.List<Tensor> grads = new java.util.ArrayList<>();
        java.util.List<Tensor> owners = new java.util.ArrayList<>();
        for (Tensor p : params) {
            if (p.grad() != null) {
                grads.add(p.grad());
                owners.add(p);
            }
        }
        if (grads.isEmpty()) {
            return Tensor.scalar(0.0f);
        }
        try (Scope ignored = disabled()) {
            Tensor stats = ai.beestream.jitnn.ops.GradStats.apply(grads, null, maxNorm, true);
            Tensor factor = stats.narrow(0, 1, 1);
            for (Tensor p : owners) {
                p.setGrad(p.grad().mul(factor));
            }
            return stats.narrow(0, 0, 1).reshape();
        }
    }

    /** A try-with-resources scope that restores the previous gradient mode. */
    public interface Scope extends AutoCloseable {
        @Override
        void close();
    }
}
