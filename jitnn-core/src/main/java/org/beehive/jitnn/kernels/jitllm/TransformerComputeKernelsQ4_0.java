// Copied from jitllm (kotselidis/jitllm, src/main/java/org/beehive/jitllm/backend/tornado/kernels/TransformerComputeKernelsQ4_0.java).
package org.beehive.jitnn.kernels.jitllm;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.utils.QuantizationUtils;

/**
 * Device kernels that read {@code Q4_0} weights <b>in the file's own representation</b>.
 *
 * <p>The Q4_K family's siblings, and shaped identically to them on purpose — same signatures, same
 * workgroup-per-row structure, same reductions — so a layer differs only in which method reference
 * it names. What differs is the decode, and Q4_0's is the simplest of the quantizations that reach
 * the device.
 *
 * <h2>The block</h2>
 *
 * <p>32 weights in 18 bytes: {@code d} (fp16) at 0, then sixteen bytes of packed nibbles. A weight
 * is {@code d * (q - 8)} — an <b>unsigned</b> nibble recentred by eight, with a single scale and no
 * minimum, which is what separates it from {@code Q4_1} and from the K-quants' per-sub-block
 * scales. Element {@code i} below 16 is the low nibble of byte {@code i}; element {@code i} at 16
 * or above is the high nibble of byte {@code i - 16}.
 *
 * <h2>Reading the packed nibbles</h2>
 *
 * <p>The packed kernels below read each four-byte group of nibbles as two sixteen-bit words via
 * {@code ByteArray.getHalfFloat(o).getHalfFloatValue()}. That accessor is used purely as a
 * bit-preserving load — these bytes are quants, never a number — and each half is masked to sixteen
 * bits before it is shifted, so the signed short cannot sign-extend into the packed word. A block
 * is 18 bytes with its quants at offset 2, so every such offset is even, which is all a pair read
 * needs; only every other quant run is four-byte aligned, which is why these are pairs and not
 * single words.
 *
 * <h2>Why it exists</h2>
 *
 * <p>Q4_0 used to be materialized as Q8_0 at load, which roughly doubles a model's device
 * footprint: 4.5 bits per weight become 8.5. That is the same problem retaining Q4_K solved for
 * Devstral, and it is what puts a Q4_0 file's own size — rather than twice it — against the
 * device's memory.
 *
 * <p>The unpacking below is the same arithmetic as the host's {@code Q4_0FloatTensor}, which is the
 * reference it was written against. {@code Q4_0DecodeTest} holds the two against each other on the
 * same bytes rather than trusting the restatement.
 */
public final class TransformerComputeKernelsQ4_0 {

    /** Weights per block. */
    private static final int QK = 32;

    /** Bytes per block: 2 (d) + 16 (packed nibbles). */
    private static final int BLOCK_BYTES = 18;

    /** Byte offset of the packed nibbles within a block. */
    private static final int QS_OFFSET = 2;

    private TransformerComputeKernelsQ4_0() {}

    /**
     * One weight, decoded from its block.
     *
     * <p>Package-private, like its Q4_K counterpart, so the decode test can hold it against the
     * host tensor directly on the same bytes. TornadoVM inlines it.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's block
     * @param withinBlock the element's index inside the block, 0..31
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        int half = withinBlock / 16; // 0 for the low nibble, 1 for the high
        int byteIndex = withinBlock - half * 16;
        int packed = w.get(blockByteOffset + QS_OFFSET + byteIndex) & 0xFF;
        int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
        return d * (q - 8);
    }

    /**
     * One row's dot product against {@code x}, reduced across a 32-lane subgroup.
     *
     * <p>Used where {@code DeviceCapability.SUBGROUP_SHUFFLE_32} holds.
     */
    private static float rowDotSimd32(
            KernelContext context, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
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

    /** One row's dot product against {@code x}, reduced through shared memory. */
    private static float rowDotShared(
            KernelContext context, int localSize, FloatArray x, ByteArray w, int n, int rowId) {
        return rowDotShared(context, localSize, x, 0, w, n, rowId);
    }

    /**
     * The same reduction over a row of a <b>batch</b> of activations.
     *
     * <p>{@code xOffset} is where this row's activation starts. Everything else — the block
     * addressing, the decode, the reduction — is the single-token path's, so a batched projection
     * is the same arithmetic in the same order over a different input offset.
     */
    private static float rowDotShared(
            KernelContext context,
            int localSize,
            FloatArray x,
            int xOffset,
            ByteArray w,
            int n,
            int rowId) {
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            partialSum += decode(w, blockByteOffset, withinBlock) * x.get(xOffset + j);
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

    /** {@code output[row] = w[row]·x}. Q4_0 counterpart of {@code matrixVectorGenericQ8Byte}. */
    public static void matrixVectorGenericQ4_0(
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

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericQ4_0}. */
    public static void matrixVectorGenericQ4_0Simd32(
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

    /** Rows a tiled batched projection decodes each weight for. */
    public static final int ROW_TILE = 8;

    // @formatter:off
    /**
     * Output rows a tiled batched projection covers per workgroup.
     *
     * <p>The row tile answers "how many times is a weight decoded"; this one answers "how many
     * times is an activation read". They are not the same question, and for this family the second
     * is the larger number: a workgroup covering one output row of {@code ffn_down} reads 8,704
     * bytes of weights and 557 KB of activations. Covering four output rows quarters the activation
     * traffic per output row and leaves the weight traffic where it was.
     *
     * <p>Four rather than eight: eight measured no better, the extra accumulators and the wider
     * shared-memory reduction costing what the saved traffic bought.
     */
    // @formatter:on
    public static final int COL_TILE = 4;

    /** Output rows a tiled batched projection covers per workgroup. */
    public static int colTile() {
        return COL_TILE;
    }

    // @formatter:off
    /**
     * {@code out[b][row] = w[row]·x[b]} for a <b>tile of rows at once</b>, decoding each weight
     * once for the whole tile.
     *
     * <p>The untiled batched kernel is one workgroup per (row, output row), so a chunk of B tokens
     * reads the weight matrix B times — exactly what B separate single-token invocations read, and
     * measurably no faster than them. A quantized projection is memory-bound, so what a chunk is
     * worth is weight reuse: this decodes a block once and applies it to eight activations.
     *
     * <p>Per row the arithmetic is unchanged — the same lane-strided order over the same values —
     * so the tiled and untiled kernels agree element for element, and the chunk width stays
     * unobservable in the result.
     *
     * <p>{@code groupIdx} is {@code tile * d + row}. A tile past the active rows returns; a tile
     * that is partly active accumulates only its real rows.
     */
    // @formatter:on
    public static void matrixVectorTiledBatchQ4_0(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int colGroups = (d + COL_TILE - 1) / COL_TILE;
        int groupId = context.groupIdx;
        int tile = groupId / colGroups;
        int colGroup = groupId - tile * colGroups;
        int firstRow = tile * ROW_TILE;
        int firstCol = colGroup * COL_TILE;
        if (firstRow >= activeRows) {
            return;
        }
        int localId = context.localIdx;

        float[] localSums =
                context.allocateFloatLocalArray(localWorkGroupSize * ROW_TILE * COL_TILE);
        int blocksPerRow = (n + QK - 1) / QK;

        // Zeroed explicitly: a private array in generated device code is uninitialized stack,
        // not Java's zero-filled allocation. Leaving it out produced NaN logits, not small errors.
        float[] acc = new float[ROW_TILE * COL_TILE];
        for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
            acc[t] = 0.0f;
        }
        float[] xs = new float[ROW_TILE];

        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;

            // The activations, once, for every output row this workgroup covers. Inactive rows
            // read zero rather than branching: they contribute nothing either way, and they are
            // not written at the end.
            for (int r = 0; r < ROW_TILE; r++) {
                int row = firstRow + r;
                float value = 0.0f;
                if (row < activeRows) {
                    value = xBatch.get(row * n + j);
                }
                xs[r] = value;
            }

            for (int c = 0; c < COL_TILE; c++) {
                int outRow = firstCol + c;
                if (outRow < d) {
                    int blockByteOffset = (outRow * blocksPerRow + blockIdx) * BLOCK_BYTES;
                    float weight = decode(w, blockByteOffset, withinBlock);
                    for (int r = 0; r < ROW_TILE; r++) {
                        acc[c * ROW_TILE + r] += weight * xs[r];
                    }
                }
            }
        }

        for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
            localSums[t * localWorkGroupSize + localId] = acc[t];
        }
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                for (int t = 0; t < ROW_TILE * COL_TILE; t++) {
                    localSums[t * localWorkGroupSize + localId] +=
                            localSums[t * localWorkGroupSize + localId + stride];
                }
            }
            context.localBarrier();
        }

        if (localId == 0) {
            for (int c = 0; c < COL_TILE; c++) {
                int outRow = firstCol + c;
                for (int r = 0; r < ROW_TILE; r++) {
                    int row = firstRow + r;
                    int slot = c * ROW_TILE + r;
                    if (outRow < d && row < activeRows) {
                        outBatch.set(row * d + outRow, localSums[slot * localWorkGroupSize]);
                    }
                }
            }
        }
    }

    /** How many rows a tiled batched projection covers per workgroup. */
    public static int rowTile() {
        return ROW_TILE;
    }

    /**
     * Rows a tiled fused gate/up covers per workgroup. Four, because it holds two accumulators
     * each.
     */
    private static final int FFN_ROW_TILE = 8;

    /** Output rows a tiled fused gate/up covers per workgroup. See {@link #COL_TILE}. */
    private static final int FFN_COL_TILE = 2;

    /** Output rows a tiled fused gate/up covers per workgroup. */
    public static int ffnColTile() {
        return FFN_COL_TILE;
    }

    /** How many rows a tiled fused gate/up covers per workgroup. */
    public static int ffnRowTile() {
        return FFN_ROW_TILE;
    }
}
