/**
 * The execution engine, which records operators in a per-thread {@link ai.beestream.jitnn.engine.Trace}
 * and, at a sync point, compiles the trace into one TornadoVM task graph
 * ({@link ai.beestream.jitnn.engine.TornadoExecutor}), cached by structure and replayed, or runs it on the
 * host with the double-precision reference ({@link ai.beestream.jitnn.engine.ReferenceExecutor}). Also the device
 * choice ({@link ai.beestream.jitnn.engine.Devices}), run-time autotuning, and the switch to vendor
 * libraries ({@link ai.beestream.jitnn.engine.Native}).
 */
package ai.beestream.jitnn.engine;
