package org.beehive.jitnn.engine;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.common.TornadoDevice;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;

/**
 * The device compiled plans run on: {@code -Djitnn.device=<driver>:<device>} (the numbering of
 * {@code tornado --devices}), else TornadoVM's default device; a thread may run on another device
 * with {@link Engine#onDevice} (one thread per rank in data-parallel training).
 */
final class Devices {

    private static final String SPEC = System.getProperty("jitnn.device", "").trim();
    private static final ThreadLocal<String> CURRENT = ThreadLocal.withInitial(() -> SPEC);
    private static final java.util.Map<String, TornadoVMBackendType> BACKENDS = new java.util.concurrent.ConcurrentHashMap<>();

    private Devices() {
    }

    static String key() {
        String spec = CURRENT.get();
        return spec.isEmpty() ? "default" : spec;
    }

    /** The device spec of this thread ("" for TornadoVM's default device). */
    static String current() {
        return CURRENT.get();
    }

    static void set(String spec) {
        CURRENT.set(spec == null ? SPEC : spec.trim());
    }

    /** The backend type of the device plans run on. */
    static TornadoVMBackendType backend() {
        return BACKENDS.computeIfAbsent(key(), k -> {
            String spec = CURRENT.get();
            TornadoDevice d;
            if (spec.isEmpty()) {
                d = TornadoExecutionPlan.DEFAULT_DEVICE;
            } else {
                String[] parts = spec.split(":");
                d = TornadoExecutionPlan.getDevice(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
            }
            return d.getTornadoVMBackend();
        });
    }

    private static final java.util.Map<String, Long> LOCAL_MEMORY = new java.util.concurrent.ConcurrentHashMap<>();

    /** Bytes of local (shared, threadgroup) memory a work-group may use on the device plans run on. */
    static long localMemory() {
        return LOCAL_MEMORY.computeIfAbsent(key(), k -> {
            String spec = CURRENT.get();
            TornadoDevice d;
            if (spec.isEmpty()) {
                d = TornadoExecutionPlan.DEFAULT_DEVICE;
            } else {
                String[] parts = spec.split(":");
                d = TornadoExecutionPlan.getDevice(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
            }
            return d.getDeviceLocalMemorySize();
        });
    }

    private static final java.util.Map<String, String> NAMES = new java.util.concurrent.ConcurrentHashMap<>();

    /** The name of the device plans run on, as the driver reports it (e.g. "NVIDIA A100 80GB PCIe"). */
    static String name() {
        return NAMES.computeIfAbsent(key(), k -> {
            String spec = CURRENT.get();
            TornadoDevice d;
            if (spec.isEmpty()) {
                d = TornadoExecutionPlan.DEFAULT_DEVICE;
            } else {
                String[] parts = spec.split(":");
                d = TornadoExecutionPlan.getDevice(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
            }
            return d.getPhysicalDevice().getDeviceName();
        });
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
        String spec = CURRENT.get();
        if (!spec.isEmpty()) {
            System.setProperty(graphName + ".device", spec);
        }
    }

    static void apply(TornadoExecutionPlan plan) {
        String spec = CURRENT.get();
        if (spec.isEmpty()) {
            return;
        }
        String[] parts = spec.split(":");
        TornadoDevice device = TornadoExecutionPlan.getDevice(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
        plan.withDevice(device);
    }
}
