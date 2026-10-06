package org.beehive.jitnn.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;

/**
 * Tensor-core GEMMs for CUDA (KernelContext MMA, m16n8k16, compute capability 8.0+), adapted from
 * jitllm's gemmMMA (TransformerBatchPrefillKernels): a 256-thread block (8 warps, 4 x 2) computes
 * a 128 x 128 tile of C, K steps of 16, the next step's operands staged in registers while the
 * current step's MMAs run. {@code gemmMMAStrideK} and its bfloat16 twin generalise it to strided
 * operands for every transpose and a split along K.
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
