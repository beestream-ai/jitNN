package org.beehive.jitnn.engine;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;

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

    private static volatile TornadoVMBackendType backend;

    /** The backend type of the device plans run on. */
    static TornadoVMBackendType backend() {
        if (backend == null) {
            TornadoDevice d;
            if (SPEC.isEmpty()) {
                d = TornadoExecutionPlan.DEFAULT_DEVICE;
            } else {
                String[] parts = SPEC.split(":");
                d = TornadoExecutionPlan.getDevice(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
            }
            backend = d.getTornadoVMBackend();
        }
        return backend;
    }

    /**
     * Pins a task graph to the device before its tasks are added.
     *
     * <p>WORKAROUND(TornadoVM J10): a task's sketch is built when the task is added, for the
     * graph's device at that moment (the default device unless {@code <graph>.device} is set);
     * {@code withDevice} on the plan comes too late for backend-specific intrinsics such as the
     * Metal simdgroup matrix calls, which fail to sketch on OpenCL.
     */
    static void pin(String graphName) {
        if (!SPEC.isEmpty()) {
            System.setProperty(graphName + ".device", SPEC);
        }
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
