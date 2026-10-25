package org.beehive.jitnn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.beehive.jitnn.engine.Node;

/**
 * Reverse-mode differentiation. {@link #backward} walks the recorded graph from the root and
 * adds the gradient operators to the current trace, so on TornadoVM the forward pass, the
 * backward pass and the optimiser step compile into one task graph.
 */
final class Autograd {

    private Autograd() {
    }

    static void backward(Tensor root, Tensor seed) {
        try {
            backwardInner(root, seed);
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static void backwardInner(Tensor root, Tensor seed) throws Exception {
        if (!root.requiresGrad()) {
            throw new IllegalStateException("tensor does not require grad");
        }
        List<Node> order = new ArrayList<>();
        Set<Node> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        topo(root.gradFn(), seen, order);

        Map<Tensor, Tensor> grads = new IdentityHashMap<>();
        grads.put(root, seed != null ? seed : Tensor.full(1.0f, root.shape().dims()));
        // a graph recorded with FP16 activations is differentiated with them (enableHalf), whatever
        // scope the backward call itself is in
        boolean half = false;
        for (Node n : order) {
            for (Tensor t : n.outputs) {
                half |= t.dtype() == DType.F16;
            }
        }
        try (Grad.Scope ignored = Grad.disabled(); AutoCloseable internal = org.beehive.jitnn.engine.Storage.internalScope();
                Grad.Scope halfScope = half && !Autocast.halfActivations() ? Autocast.enableHalf() : () -> {
                }; Grad.Scope backwardScope = Autocast.backward()) {
            for (int i = order.size() - 1; i >= 0; i--) {
                Node node = order.get(i);
                Tensor gy = grads.remove(node.out(0));
                if (gy == null) {
                    continue;
                }
                Tensor[] gx = node.op.backward(node, gy);
                for (int j = 0; j < node.inputs.length; j++) {
                    Tensor x = node.in(j);
                    if (gx[j] == null || !x.requiresGrad()) {
                        continue;
                    }
                    // a leaf (a parameter) gets a gradient of its own type: FP32 parameters FP32 gradients;
                    // activations' gradients stay as computed (FP32 from the products), so FP16 activations
                    // do not round-trip their gradients through FP16
                    if (x.gradFn() == null && gx[j].dtype() != x.dtype() && x.dtype().isFloat() && !x.dtype().isBlockQuantized()) {
                        gx[j] = gx[j].to(x.dtype());
                    }
                    if (!gx[j].shape().equals(x.shape())) {
                        throw new IllegalStateException(node.op.name() + " gradient " + gx[j].shape() + " for input " + x.shape());
                    }
                    if (x.gradFn() == null) {
                        x.accumulateGrad(gx[j]);
                        x.grad().storage().markVisible();
                    } else {
                        grads.merge(x, gx[j], Tensor::add);
                    }
                }
            }
        }
    }

    private static void topo(Node node, Set<Node> seen, List<Node> order) {
        if (node == null || !seen.add(node)) {
            return;
        }
        // Iterative post-order DFS: deep graphs (long RNN unrolls) would overflow the stack.
        List<Object[]> stack = new ArrayList<>();
        stack.add(new Object[] { node, 0 });
        while (!stack.isEmpty()) {
            Object[] top = stack.get(stack.size() - 1);
            Node n = (Node) top[0];
            int next = (Integer) top[1];
            if (next < n.inputs.length) {
                top[1] = next + 1;
                Node child = n.in(next).gradFn();
                if (child != null && seen.add(child)) {
                    stack.add(new Object[] { child, 0 });
                }
            } else {
                order.add(n);
                stack.remove(stack.size() - 1);
            }
        }
    }
}
