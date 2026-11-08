/**
 * The device kernels behind the operators, written in Java with TornadoVM's {@code KernelContext}
 * API so that each runs on CUDA, OpenCL and Metal: element-wise, reductions, GEMM (and tensor-core
 * GEMMs on NVIDIA), convolutions, attention, normalisation, FFT, linear algebra, sparse, quantised
 * and FP8 kernels. Operators in {@link ai.beestream.jitnn.ops} choose and launch them.
 */
package ai.beestream.jitnn.kernels;
