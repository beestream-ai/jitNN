/**
 * The operators: each {@link ai.beestream.jitnn.engine.Op} here has a device implementation (one or
 * more kernels from {@link ai.beestream.jitnn.kernels}, or a vendor-library call), a host reference
 * and a gradient. {@link ai.beestream.jitnn.Tensor} and {@link ai.beestream.jitnn.F} are the public way to
 * call them; {@link ai.beestream.jitnn.ops.Fusion} rewrites traces to fuse them.
 */
package ai.beestream.jitnn.ops;
