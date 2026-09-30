package org.beehive.jitnn.engine;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;

/**
 * The device compiled plans run on: {@code -Djitnn.device=<driver>:<device>} (the numbering of
 * {@code tornado --devices}), else TornadoVM's default device.
 */
final class Devices {

    private static final String SPEC = System.getProperty("jitnn.device", "").trim();

    private Devices() {
    }

    static String key() {
        return SPEC.isEmpty() ? "default" : SPEC;
    }

    static void apply(TornadoExecutionPlan plan) {
        if (SPEC.isEmpty()) {
            return;
        }
        String[] parts = SPEC.split(":");
        TornadoDevice device = TornadoExecutionPlan.getDevice(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
        plan.withDevice(device);
    }
}
