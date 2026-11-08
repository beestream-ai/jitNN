// Copied from jitllm (kotselidis/jitllm, src/main/java/org/beehive/jitllm/backend/tornado/kernels/TransformerComputeKernelsQ4_K.java).
package ai.beestream.jitnn.kernels.jitllm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Device kernels that read {@code Q4_K} weights <b>in the file's own representation</b>.
 *
 * <p>They are deliberately shaped like their Q8_0 siblings in {@link
 * TransformerComputeKernelsLayered} — same signatures, same workgroup-per-row structure, same
 * reduction — so the layer classes differ only in which method reference they name. What changes is
 * the decode in the inner loop.
 *
 * <h2>The super-block</h2>
 *
 * <p>256 weights in 144 bytes: {@code d} (fp16) at 0, {@code dmin} (fp16) at 2, twelve bytes of
 * packed 6-bit scale/min pairs at 4, and 128 bytes of 4-bit weights at 16. A weight is {@code d *
 * scale(sub) * q - dmin * min(sub)}, where the 256 weights are covered by eight sub-blocks of 32.
 * Within a 64-weight pair the first 32 take the low nibble and the next 32 the high nibble of the
 * same byte.
 *
 * <p>The unpacking below is the same arithmetic as the host's {@code Q4_KFloatTensor}, which is the
 * reference this was written against — the two must agree, and {@code Q4_KDeviceDecodeAccelTest}
 * asserts exactly that on real weights rather than trusting the restatement.
 */
public final class TransformerComputeKernelsQ4_K {

    /** Weights per super-block. */
    private static final int QK_K = 256;

    /** Bytes per super-block: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs). */
    private static final int BLOCK_BYTES = 144;

    /** Byte offset of the packed 6-bit scale/min pairs within a super-block. */
    private static final int SCALES_OFFSET = 4;

    /** Byte offset of the 4-bit weights within a super-block. */
    private static final int QS_OFFSET = 16;

    private TransformerComputeKernelsQ4_K() {}

    /**
     * One weight, decoded from its super-block.
     *
     * <p>Kept as a static helper so every kernel below decodes identically; TornadoVM inlines it.
     * Package-private rather than private so {@code Q4_KDecodeTest} can hold it against the host's
     * {@code Q4_KFloatTensor} directly, on the same bytes.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's super-block
     * @param withinBlock the element's index inside the super-block, 0.255
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        float dmin = w.getHalfFloat(blockByteOffset + 2).getFloat32();

        int pairIndex = withinBlock / 64; // 0..3
        int posInPair = withinBlock - pairIndex * 64; // 0..63
        int highNibble = posInPair / 32; // 0 for the first 32, 1 for the next
        int subBlock = pairIndex * 2 + highNibble;

        int qByte =
                w.get(blockByteOffset + QS_OFFSET + pairIndex * 32 + (posInPair - highNibble * 32))
                        & 0xFF;
        int q = (highNibble == 0) ? (qByte & 0xF) : ((qByte >> 4) & 0xF);

        int scalesBase = blockByteOffset + SCALES_OFFSET;
        int sc;
        int m;
        if (subBlock < 4) {
            sc = w.get(scalesBase + subBlock) & 63;
            m = w.get(scalesBase + subBlock + 4) & 63;
        } else {
            int lowScale = w.get(scalesBase + subBlock + 4) & 0xFF;
            int highScale = w.get(scalesBase + subBlock - 4) & 0xFF;
            sc = (lowScale & 0xF) | ((highScale >> 6) << 4);
            m = ((lowScale >> 4) & 0xF) | (((w.get(scalesBase + subBlock) & 0xFF) >> 6) << 4);
        }
        return d * sc * q - dmin * m;
    }

    /**
     * One row's dot product against {@code x}, reduced across a 32-lane subgroup.
     *
     * <p>The Q4_K counterpart of {@code matrixVectorRowMajorOptimizedQ8_0Byte}'s reduction, in the
     * shuffle form — used where {@code DeviceCapability.SUBGROUP_SHUFFLE_32} holds.
     */
    private static float rowDotSimd32(
            KernelContext context, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            int blockIdx = j / QK_K;
            int withinBlock = j - blockIdx * QK_K;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            partialSum += decode(w, blockByteOffset, withinBlock) * x.get(j);
        }

        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);
        return partialSum;
    }

    /**
     * One row's dot product against {@code x}, reduced through shared memory.
     *
     * <p>The portable form, for devices without a verified 32-lane shuffle — the same choice {@code
     * matrixVectorRowMajorOptimizedQ8_0Byte} makes.
     */
    private static float rowDotShared(
            KernelContext context, int localSize, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int blockIdx = j / QK_K;
            int withinBlock = j - blockIdx * QK_K;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            partialSum += decode(w, blockByteOffset, withinBlock) * x.get(j);
        }

        localSums[localId] = partialSum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        return localSums[0];
    }

    /** {@code output[row] = w[row]·x}. Q4_K counterpart of {@code matrixVectorGenericQ8Byte}. */
    public static void matrixVectorGenericQ4_K(
            KernelContext context,
            FloatArray x,
            FloatArray output,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotShared(context, localWorkGroupSize, x, w, n, rowId);
        if (context.localIdx == 0) {
            output.set(rowId, sum);
        }
    }

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericQ4_K}. */
    public static void matrixVectorGenericQ4_KSimd32(
            KernelContext context, FloatArray x, FloatArray output, ByteArray w, int n, int d) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotSimd32(context, x, w, n, rowId);
        if (context.localIdx == 0) {
            output.set(rowId, sum);
        }
    }

}
