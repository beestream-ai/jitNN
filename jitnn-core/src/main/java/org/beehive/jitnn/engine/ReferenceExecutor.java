package org.beehive.jitnn.engine;

import org.beehive.jitnn.Tensor;

/** Runs one node on the host with the operator's Java reference implementation. */
final class ReferenceExecutor {

    private ReferenceExecutor() {
    }

    static void run(Node node) {
        for (Tensor t : node.inputs) {
            Engine.ensureHost(t.storage());
        }
        for (Tensor t : node.outputs) {
            t.storage().allocate();
            t.storage().hostWritten();
        }
        node.op.reference(node);
        node.executed = true;
    }
}
