/**
 * Kernels adapted from jitllm that read GGUF's quantised weights (Q4_0, Q4_1, Q4_K, Q5_K, Q6_K)
 * in the file's own block layout, without dequantising them first.
 */
package ai.beestream.jitnn.kernels.jitllm;
