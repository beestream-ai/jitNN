/**
 * Several devices: collectives (all-reduce, broadcast, reduce-scatter, all-gather) among ranks,
 * staged through the host ({@link ai.beestream.jitnn.dist.HostCollective}) or run by NCCL on NVIDIA GPUs
 * ({@link ai.beestream.jitnn.dist.NcclCollective}), and {@link ai.beestream.jitnn.dist.ProcessGroup}, which
 * runs one rank per device.
 */
package ai.beestream.jitnn.dist;
