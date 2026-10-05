package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

/**
 * Tensor-core GEMMs for CUDA (KernelContext MMA, m16n8k16, compute capability 8.0+), copied from
 * jitllm: {@code gemmMMA} (TransformerBatchPrefillKernels), {@code gemmMMAStrided}
 * (DeepSeek2BatchKernels), {@code gemmMMASplitK} and {@code splitKReduce}
 * (Gemma4BatchPrefillKernels), plus a bfloat16 twin of {@code gemmMMA}. A 256-thread block (8
 * warps, 4 x 2) computes a 128 x 128 tile of C = A B^T with A [M, K] and B [N, K] row-major, K
 * steps of 16, the next step's operands staged in registers while the current step's MMAs run.
 * They need M and N to be multiples of 128 and K of 16 (results are stored by whole fragments).
 * {@code gemmMMAStrideK} and its bfloat16 twin are jitNN's generalisation of {@code gemmMMA}:
 * strided operands for every transpose, and a split along K.
 */
public final class MmaGemmKernels {

    private static final int WARP_SIZE = 32;
    private static final int BM = 128, BN = 128, BK = 16;
    private static final int WARPS_M = 4, WARPS_N = 2;
    private static final int WM = BM / WARPS_M;
    private static final int WN = BN / WARPS_N;
    private static final int B_SUBTILE_BYTES = 256;
    /** Threads per block. */
    public static final int THREADS = 256;
    /** Output tile edge. */
    public static final int TILE = 128;

    private MmaGemmKernels() {
    }

    /** Two consecutive FP16 values packed into one int (lo | hi << 16) for the ldmatrix tiles. */
    private static int packHalves(HalfFloatArray src, int idxLo, int idxHi) {
        int lo = src.get(idxLo).getHalfFloatValue() & 0xFFFF;
        int hi = src.get(idxHi).getHalfFloatValue() & 0xFFFF;
        return lo | (hi << 16);
    }

    /** Two bfloat16 bit patterns packed into one int (lo | hi << 16). */
    private static int packBits(ShortArray src, int idxLo, int idxHi) {
        int lo = src.get(idxLo) & 0xFFFF;
        int hi = src.get(idxHi) & 0xFFFF;
        return lo | (hi << 16);
    }

    /**
     * Tensor-core GEMM: C[M,N] (FP32) = A[M,K] (FP16, row-major) × B[N,K] (FP16, row-major).
     *
     * <p>Software-pipelined: each thread stages the NEXT K-step's A/B elements in registers while
     * the CURRENT step's ldmatrix+MMA execute, so global-memory latency is hidden behind
     * tensor-core compute. Shared memory stays single-buffered; the two block barriers per step
     * preserve correctness (read-complete before overwrite, write-complete before next ldmatrix).
     *
     * <p>Requires M % 128 == 0, N % 128 == 0, K % 16 == 0, SM 8.0+. Worker:
     * WorkerGrid2D((M/128)*256, N/128), local (256,1,1).
     */
    public static void gemmMMA(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray B,
            FloatArray C,
            int M,
            int N,
            int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        // ── Per-thread staging index math (constant across K-steps) ──────────
        // A tile: BM*BK/2 = 1024 ints; layout idx = m_row*(BK/2) + k_pair.
        //   m_row = idx >>> 3, k = (idx & 7)*2; global A element (blockRow+m_row, kBase+k).
        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        // B tile: BK*BN/2 = 1024 ints; subTileId = idx >>> 6, k_row = (idx & 63) >>> 2,
        //   col = subTileId*8 + (idx & 3)*2; B[col, k] at col*K + k (pair at +K).
        int bIdx0 = tid;
        int gB0 = (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 = (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 = (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 = (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K + ((bIdx3 & 63) >>> 2);

        // ── Prologue: stage K-step 0 ─────────────────────────────────────────
        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0 = packHalves(B, gB0, gB0 + K);
        int bReg1 = packHalves(B, gB1, gB1 + K);
        int bReg2 = packHalves(B, gB2, gB2 + K);
        int bReg3 = packHalves(B, gB3, gB3 + K);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            // Issue next step's global loads FIRST: independent of the MMAs below,
            // so their latency overlaps ldmatrix + tensor-core compute.
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packHalves(B, gB0 + kOff, gB0 + kOff + K);
                bReg1 = packHalves(B, gB1 + kOff, gB1 + kOff + K);
                bReg2 = packHalves(B, gB2 + kOff, gB2 + kOff + K);
                bReg3 = packHalves(B, gB3 + kOff, gB3 + kOff + K);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier(); // all shared reads for this K-step complete

            // Overwrite shared tiles for the next step; overlaps the MMAs below.
            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier(); // shared writes visible before next step's ldmatrix
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, N);
    }

    /**
     * {@link #gemmMMA} on bfloat16 operands (raw bits in ShortArrays): C[M,N] (FP32) = A[M,K] x
     * B[N,K]^T with the same tiling, staging and requirements; the MMA reads the bits as bfloat16.
     */
    public static void gemmMMABF16(
            KernelContext ctx,
            ShortArray A,
            ShortArray B,
            FloatArray C,
            int M,
            int N,
            int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        // ── Per-thread staging index math (constant across K-steps) ──────────
        // A tile: BM*BK/2 = 1024 ints; layout idx = m_row*(BK/2) + k_pair.
        //   m_row = idx >>> 3, k = (idx & 7)*2; global A element (blockRow+m_row, kBase+k).
        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        // B tile: BK*BN/2 = 1024 ints; subTileId = idx >>> 6, k_row = (idx & 63) >>> 2,
        //   col = subTileId*8 + (idx & 3)*2; B[col, k] at col*K + k (pair at +K).
        int bIdx0 = tid;
        int gB0 = (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 = (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 = (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 = (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K + ((bIdx3 & 63) >>> 2);

        // ── Prologue: stage K-step 0 ─────────────────────────────────────────
        int aReg0 = packBits(A, gA0, gA0 + 1);
        int aReg1 = packBits(A, gA1, gA1 + 1);
        int aReg2 = packBits(A, gA2, gA2 + 1);
        int aReg3 = packBits(A, gA3, gA3 + 1);
        int bReg0 = packBits(B, gB0, gB0 + K);
        int bReg1 = packBits(B, gB1, gB1 + K);
        int bReg2 = packBits(B, gB2, gB2 + K);
        int bReg3 = packBits(B, gB3, gB3 + K);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            // Issue next step's global loads FIRST: independent of the MMAs below,
            // so their latency overlaps ldmatrix + tensor-core compute.
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packBits(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packBits(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packBits(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packBits(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packBits(B, gB0 + kOff, gB0 + kOff + K);
                bReg1 = packBits(B, gB1 + kOff, gB1 + kOff + K);
                bReg2 = packBits(B, gB2 + kOff, gB2 + kOff + K);
                bReg3 = packBits(B, gB3 + kOff, gB3 + kOff + K);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier(); // all shared reads for this K-step complete

            // Overwrite shared tiles for the next step; overlaps the MMAs below.
            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mmaBF16(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mmaBF16(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mmaBF16(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mmaBF16(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mmaBF16(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mmaBF16(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mmaBF16(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mmaBF16(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mmaBF16(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mmaBF16(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mmaBF16(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mmaBF16(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mmaBF16(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mmaBF16(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mmaBF16(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mmaBF16(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier(); // shared writes visible before next step's ldmatrix
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, N);
    }

    /**
     * {@link #gemmMMA} with leading dimensions and a batch: problem
     * {@code z = groupIdz} computes {@code C[r][colOffset + c] = A[r] . B[c]} for {@code r < M, c
     * < N} over {@code K}, where {@code A[r]} starts at {@code z * aBatch + r * lda}, {@code B[c]}
     * at {@code z * bBatch + c * ldb}, and {@code C}'s rows are {@code ldc} apart with {@code
     * colOffset = z * cBatchCols}. Requires M % 128 == 0, N % 128 == 0, K % 16 == 0. Worker:
     * WorkerGrid3D((M/128)*256, N/128, batches), local (256,1,1).
     */
    public static void gemmMMAStrided(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray B,
            FloatArray C,
            int M,
            int N,
            int K,
            int lda,
            int ldb,
            int ldc,
            int aBatch,
            int bBatch,
            int cBatchCols) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        int z = ctx.groupIdz;
        int aBase = z * aBatch;
        int bBase = z * bBatch;
        int colOffset = z * cBatchCols;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = aBase + (blockRow + (aIdx0 >>> 3)) * lda + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = aBase + (blockRow + (aIdx1 >>> 3)) * lda + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = aBase + (blockRow + (aIdx2 >>> 3)) * lda + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = aBase + (blockRow + (aIdx3 >>> 3)) * lda + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int gB0 =
                bBase
                        + (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * ldb
                        + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 =
                bBase
                        + (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * ldb
                        + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 =
                bBase
                        + (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * ldb
                        + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 =
                bBase
                        + (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * ldb
                        + ((bIdx3 & 63) >>> 2);

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0 = packHalves(B, gB0, gB0 + ldb);
        int bReg1 = packHalves(B, gB1, gB1 + ldb);
        int bReg2 = packHalves(B, gB2, gB2 + ldb);
        int bReg3 = packHalves(B, gB3, gB3 + ldb);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packHalves(B, gB0 + kOff, gB0 + kOff + ldb);
                bReg1 = packHalves(B, gB1 + kOff, gB1 + kOff + ldb);
                bReg2 = packHalves(B, gB2 + kOff, gB2 + kOff + ldb);
                bReg3 = packHalves(B, gB3 + kOff, gB3 + kOff + ldb);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBaseTile = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBaseTile + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = blockRow + warpM * WM;
        int cBase = colOffset + blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, ldc);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, ldc);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, ldc);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, ldc);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, ldc);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, ldc);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, ldc);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, ldc);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, ldc);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, ldc);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, ldc);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, ldc);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, ldc);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, ldc);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, ldc);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, ldc);
    }

    /**
     * {@link #gemmMMA} split along the depth: slice {@code z = groupIdz} of {@code slices} covers
     * {@code K / slices} and writes its partial product to {@code partial[z]}; {@link #splitKReduce}
     * sums them. In jitllm it serves the narrow projections of a model with few output blocks.
     *
     * <p>The two fixes compose and neither subsumes the other. Splitting the depth answers the grid
     * — a projection into {@code dim} gives forty-eight thread blocks on a
     * hundred-and-twenty-eight-SM device. Decoding the weights first answers the pipeline, since a
     * Q8_0 staging converts float to half in software for want of a register-level conversion. This
     * is both.
     *
     * <p>Requires {@code M % 128 == 0}, {@code N % 128 == 0}, {@code (K / slices) % 16 == 0}.
     * Worker: {@code WorkerGrid3D((M/128)*256, N/128, slices)}, local (256,1,1).
     */
    public static void gemmMMASplitK(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray B,
            FloatArray partial,
            int M,
            int N,
            int K,
            int slices) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        int slice = ctx.groupIdz;
        int kSlice = K / slices;
        int kBase = slice * kSlice;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + kBase + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + kBase + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + kBase + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + kBase + ((aIdx3 & 7) << 1);
        int bIdx0 = tid;
        int gB0 =
                (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * K
                        + kBase
                        + ((bIdx0 & 63) >>> 2);
        int bIdx1 = tid + 256;
        int gB1 =
                (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * K
                        + kBase
                        + ((bIdx1 & 63) >>> 2);
        int bIdx2 = tid + 512;
        int gB2 =
                (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * K
                        + kBase
                        + ((bIdx2 & 63) >>> 2);
        int bIdx3 = tid + 768;
        int gB3 =
                (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * K
                        + kBase
                        + ((bIdx3 & 63) >>> 2);

        int aReg0 = packHalves(A, gA0, gA0 + 1);
        int aReg1 = packHalves(A, gA1, gA1 + 1);
        int aReg2 = packHalves(A, gA2, gA2 + 1);
        int aReg3 = packHalves(A, gA3, gA3 + 1);
        int bReg0 = packHalves(B, gB0, gB0 + K);
        int bReg1 = packHalves(B, gB1, gB1 + K);
        int bReg2 = packHalves(B, gB2, gB2 + K);
        int bReg3 = packHalves(B, gB3, gB3 + K);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = kSlice / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalves(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalves(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalves(A, gA3 + kOff, gA3 + kOff + 1);
                bReg0 = packHalves(B, gB0 + kOff, gB0 + kOff + K);
                bReg1 = packHalves(B, gB1 + kOff, gB1 + kOff + K);
                bReg2 = packHalves(B, gB2 + kOff, gB2 + kOff + K);
                bReg3 = packHalves(B, gB3 + kOff, gB3 + kOff + K);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = slice * M + blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, partial, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, partial, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, partial, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, partial, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, partial, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, partial, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, partial, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, partial, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, partial, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, partial, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, partial, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, partial, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, partial, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, partial, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, partial, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, partial, rBase + 16, cBase + 56, N);
    }

    /**
     * Sums the depth slices {@link #gemmMMASplitK} left behind into the output.
     *
     * <p>One thread per output element, slices added in increasing order. The traffic is {@code
     * slices} reads and one write per element — 12.6 MB per call at four slices and a chunk of 512,
     * against the tens of milliseconds the GEMM itself takes, so the pass is not where the time
     * goes.
     *
     * <p>Worker: {@code M*N} threads, local 256.
     */
    public static void splitKReduce(KernelContext context, FloatArray partial, FloatArray out, int elements, int slices) {
        int gid = context.globalIdx;
        if (gid < elements) {
            float sum = 0.0f;
            for (int s = 0; s < slices; s++) {
                sum += partial.get(s * elements + gid);
            }
            out.set(gid, sum);
        }
    }

    /**
     * {@link #gemmMMA} on strided operands, so any transpose works: element (r, k) of A is at
     * {@code r * sAr + k * sAk} and element (c, k) of B at {@code c * sBc + k * sBk} (A [M, K]
     * row-major: sAr = K, sAk = 1; A^T stored [K, M]: sAr = 1, sAk = M; the same for B with N), and
     * C[M, N] = A B^T. Split along K: slice z = groupIdz covers depth z * K .. (z + 1) * K and writes
     * rows z * M .. of C (C holds slices x M rows; one slice computes the product itself). Same
     * tiling and requirements as {@link #gemmMMA}. Grid: (M/128) x (N/128) x slices groups of 256.
     */
    public static void gemmMMAStrideK(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray B,
            FloatArray C,
            int M,
            int N,
            int K,
            int sAr,
            int sAk,
            int sBc,
            int sBk) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        int z = ctx.groupIdz;
        int k0 = z * K;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        // ── Per-thread staging index math (constant across K-steps) ──────────
        // A tile: BM*BK/2 = 1024 ints; layout idx = m_row*(BK/2) + k_pair.
        //   m_row = idx >>> 3, k = (idx & 7)*2; global A element (blockRow+m_row, kBase+k).
        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * sAr + (k0 + ((aIdx0 & 7) << 1)) * sAk;
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * sAr + (k0 + ((aIdx1 & 7) << 1)) * sAk;
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * sAr + (k0 + ((aIdx2 & 7) << 1)) * sAk;
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * sAr + (k0 + ((aIdx3 & 7) << 1)) * sAk;
        // B tile: BK*BN/2 = 1024 ints; subTileId = idx >>> 6, k_row = (idx & 63) >>> 2,
        //   col = subTileId*8 + (idx & 3)*2; B[col, k] at col*K + k (pair at +K).
        int bIdx0 = tid;
        int gB0 = (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * sBc + (k0 + ((bIdx0 & 63) >>> 2)) * sBk;
        int bIdx1 = tid + 256;
        int gB1 = (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * sBc + (k0 + ((bIdx1 & 63) >>> 2)) * sBk;
        int bIdx2 = tid + 512;
        int gB2 = (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * sBc + (k0 + ((bIdx2 & 63) >>> 2)) * sBk;
        int bIdx3 = tid + 768;
        int gB3 = (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * sBc + (k0 + ((bIdx3 & 63) >>> 2)) * sBk;

        // ── Prologue: stage K-step 0 ─────────────────────────────────────────
        int aReg0 = packHalves(A, gA0, gA0 + sAk);
        int aReg1 = packHalves(A, gA1, gA1 + sAk);
        int aReg2 = packHalves(A, gA2, gA2 + sAk);
        int aReg3 = packHalves(A, gA3, gA3 + sAk);
        int bReg0 = packHalves(B, gB0, gB0 + sBc);
        int bReg1 = packHalves(B, gB1, gB1 + sBc);
        int bReg2 = packHalves(B, gB2, gB2 + sBc);
        int bReg3 = packHalves(B, gB3, gB3 + sBc);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            // Issue next step's global loads FIRST: independent of the MMAs below,
            // so their latency overlaps ldmatrix + tensor-core compute.
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packHalves(A, gA0 + kOff * sAk, gA0 + (kOff + 1) * sAk);
                aReg1 = packHalves(A, gA1 + kOff * sAk, gA1 + (kOff + 1) * sAk);
                aReg2 = packHalves(A, gA2 + kOff * sAk, gA2 + (kOff + 1) * sAk);
                aReg3 = packHalves(A, gA3 + kOff * sAk, gA3 + (kOff + 1) * sAk);
                bReg0 = packHalves(B, gB0 + kOff * sBk, gB0 + kOff * sBk + sBc);
                bReg1 = packHalves(B, gB1 + kOff * sBk, gB1 + kOff * sBk + sBc);
                bReg2 = packHalves(B, gB2 + kOff * sBk, gB2 + kOff * sBk + sBc);
                bReg3 = packHalves(B, gB3 + kOff * sBk, gB3 + kOff * sBk + sBc);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier(); // all shared reads for this K-step complete

            // Overwrite shared tiles for the next step; overlaps the MMAs below.
            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier(); // shared writes visible before next step's ldmatrix
        }

        int rBase = z * M + blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, N);
    }

    /**
     * {@link #gemmMMAStrideK} on bfloat16 operands (raw bits): {@link #gemmMMA} on strided operands: element (r, k) of A is at
     * {@code r * sAr + k * sAk} and element (c, k) of B at {@code c * sBc + k * sBk} (A [M, K]
     * row-major: sAr = K, sAk = 1; A^T stored [K, M]: sAr = 1, sAk = M; the same for B with N), and
     * C[M, N] = A B^T. Split along K: slice z = groupIdz covers depth z * K .. (z + 1) * K and writes
     * rows z * M .. of C (C holds slices x M rows; one slice computes the product itself). Same
     * tiling and requirements as {@link #gemmMMA}. Grid: (M/128) x (N/128) x slices groups of 256.
     */
    public static void gemmMMAStrideKBF16(
            KernelContext ctx,
            ShortArray A,
            ShortArray B,
            FloatArray C,
            int M,
            int N,
            int K,
            int sAr,
            int sAk,
            int sBc,
            int sBk) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;
        int z = ctx.groupIdz;
        int k0 = z * K;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        // ── Per-thread staging index math (constant across K-steps) ──────────
        // A tile: BM*BK/2 = 1024 ints; layout idx = m_row*(BK/2) + k_pair.
        //   m_row = idx >>> 3, k = (idx & 7)*2; global A element (blockRow+m_row, kBase+k).
        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * sAr + (k0 + ((aIdx0 & 7) << 1)) * sAk;
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * sAr + (k0 + ((aIdx1 & 7) << 1)) * sAk;
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * sAr + (k0 + ((aIdx2 & 7) << 1)) * sAk;
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * sAr + (k0 + ((aIdx3 & 7) << 1)) * sAk;
        // B tile: BK*BN/2 = 1024 ints; subTileId = idx >>> 6, k_row = (idx & 63) >>> 2,
        //   col = subTileId*8 + (idx & 3)*2; B[col, k] at col*K + k (pair at +K).
        int bIdx0 = tid;
        int gB0 = (blockCol + ((bIdx0 >>> 6) << 3) + ((bIdx0 & 3) << 1)) * sBc + (k0 + ((bIdx0 & 63) >>> 2)) * sBk;
        int bIdx1 = tid + 256;
        int gB1 = (blockCol + ((bIdx1 >>> 6) << 3) + ((bIdx1 & 3) << 1)) * sBc + (k0 + ((bIdx1 & 63) >>> 2)) * sBk;
        int bIdx2 = tid + 512;
        int gB2 = (blockCol + ((bIdx2 >>> 6) << 3) + ((bIdx2 & 3) << 1)) * sBc + (k0 + ((bIdx2 & 63) >>> 2)) * sBk;
        int bIdx3 = tid + 768;
        int gB3 = (blockCol + ((bIdx3 >>> 6) << 3) + ((bIdx3 & 3) << 1)) * sBc + (k0 + ((bIdx3 & 63) >>> 2)) * sBk;

        // ── Prologue: stage K-step 0 ─────────────────────────────────────────
        int aReg0 = packBits(A, gA0, gA0 + sAk);
        int aReg1 = packBits(A, gA1, gA1 + sAk);
        int aReg2 = packBits(A, gA2, gA2 + sAk);
        int aReg3 = packBits(A, gA3, gA3 + sAk);
        int bReg0 = packBits(B, gB0, gB0 + sBc);
        int bReg1 = packBits(B, gB1, gB1 + sBc);
        int bReg2 = packBits(B, gB2, gB2 + sBc);
        int bReg3 = packBits(B, gB3, gB3 + sBc);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        bTile[bIdx0] = bReg0;
        bTile[bIdx1] = bReg1;
        bTile[bIdx2] = bReg2;
        bTile[bIdx3] = bReg3;
        ctx.localBarrier();

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            // Issue next step's global loads FIRST: independent of the MMAs below,
            // so their latency overlaps ldmatrix + tensor-core compute.
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * BK;
                aReg0 = packBits(A, gA0 + kOff * sAk, gA0 + (kOff + 1) * sAk);
                aReg1 = packBits(A, gA1 + kOff * sAk, gA1 + (kOff + 1) * sAk);
                aReg2 = packBits(A, gA2 + kOff * sAk, gA2 + (kOff + 1) * sAk);
                aReg3 = packBits(A, gA3 + kOff * sAk, gA3 + (kOff + 1) * sAk);
                bReg0 = packBits(B, gB0 + kOff * sBk, gB0 + kOff * sBk + sBc);
                bReg1 = packBits(B, gB1 + kOff * sBk, gB1 + kOff * sBk + sBc);
                bReg2 = packBits(B, gB2 + kOff * sBk, gB2 + kOff * sBk + sBc);
                bReg3 = packBits(B, gB3 + kOff * sBk, gB3 + kOff * sBk + sBc);
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier(); // all shared reads for this K-step complete

            // Overwrite shared tiles for the next step; overlaps the MMAs below.
            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
                bTile[bIdx0] = bReg0;
                bTile[bIdx1] = bReg1;
                bTile[bIdx2] = bReg2;
                bTile[bIdx3] = bReg3;
            }

            c00 = ctx.mmaBF16(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mmaBF16(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mmaBF16(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mmaBF16(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mmaBF16(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mmaBF16(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mmaBF16(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mmaBF16(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mmaBF16(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mmaBF16(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mmaBF16(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mmaBF16(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mmaBF16(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mmaBF16(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mmaBF16(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mmaBF16(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier(); // shared writes visible before next step's ldmatrix
        }

        int rBase = z * M + blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, N);
    }
}
