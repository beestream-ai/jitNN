package org.beehive.jitnn.engine;

import java.util.ArrayList;
import java.util.List;

/** The operators recorded on this thread since the last sync point, in program order. */
final class Trace {

    private static final ThreadLocal<List<Node>> PENDING = ThreadLocal.withInitial(ArrayList::new);

    private Trace() {
    }

    static void record(Node node) {
        PENDING.get().add(node);
    }

    static List<Node> drain() {
        List<Node> nodes = PENDING.get();
        PENDING.set(new ArrayList<>());
        return nodes;
    }

    static int size() {
        return PENDING.get().size();
    }
}
