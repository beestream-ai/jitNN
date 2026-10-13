/**
 * The calls into vendor libraries (cuBLAS, cuDNN, cuFFT, cuSPARSE on CUDA; MLX on Metal) that
 * operators make when {@code -Djitnn.native} enables them (see {@link org.beehive.jitnn.engine.Native}).
 * {@link org.beehive.jitnn.natives.Libs} is the only class operators use; it loads the
 * library-specific classes only once a library is known to be available.
 */
package org.beehive.jitnn.natives;
