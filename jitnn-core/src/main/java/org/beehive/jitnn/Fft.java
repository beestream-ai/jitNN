package org.beehive.jitnn;

import org.beehive.jitnn.kernels.FftKernels;
import org.beehive.jitnn.ops.Dft;

/**
 * Fourier transforms after torch.fft, and the short-time transforms torch.stft and torch.istft.
 * Complex tensors are real tensors with a trailing axis of 2 (re, im), as torch.view_as_real gives
 * them; {@code dim} counts the complex axes only. All transforms are differentiable.
 *
 * <p>The transforms are direct sums, O(n^2) per row (see {@link FftKernels}): right for the
 * lengths found in models (window sizes, sequence mixing), not for very long signals.
 */
public final class Fft {

    /** Scaling, torch's {@code norm}: BACKWARD scales the inverse by 1/n, ORTHO both by 1/sqrt(n), FORWARD the forward by 1/n. */
    public enum Norm {
        BACKWARD, ORTHO, FORWARD
    }

    private Fft() {
    }

    private static float scale(Norm norm, int n, boolean inverse) {
        return switch (norm) {
            case BACKWARD -> inverse ? 1.0f / n : 1.0f;
            case ORTHO -> (float) (1.0 / Math.sqrt(n));
            case FORWARD -> inverse ? 1.0f : 1.0f / n;
        };
    }

    /** Moves complex axis {@code dim} of x [..., 2] to the last complex position, or back. */
    private static Tensor swapComplex(Tensor x, int dim) {
        int complexRank = x.rank() - 1;
        int ax = dim < 0 ? dim + complexRank : dim;
        return ax == complexRank - 1 ? x : x.transpose(ax, complexRank - 1);
    }

    private static Tensor swapReal(Tensor x, int dim) {
        int ax = x.shape().axis(dim);
        return ax == x.rank() - 1 ? x : x.transpose(ax, x.rank() - 1);
    }

    private static Tensor c2c(Tensor x, int n, int dim, Norm norm, boolean inverse) {
        Tensor t = swapComplex(x, dim);
        int len = t.dim(-2);
        int size = n > 0 ? n : len;
        Tensor y = Dft.apply(t, Math.min(len, size), size, size, inverse ? 1 : -1, scale(norm, size, inverse), false, false, FftKernels.WEIGHT_NONE);
        return swapComplex(y, dim);
    }

    /** torch.fft.fft of complex x [..., 2]; n > 0 crops or zero-pads axis {@code dim} to n. */
    public static Tensor fft(Tensor x, int n, int dim, Norm norm) {
        return c2c(x, n, dim, norm, false);
    }

    /** torch.fft.fft of complex x along the last axis, full length, BACKWARD scaling. */
    public static Tensor fft(Tensor x) {
        return fft(x, -1, -1, Norm.BACKWARD);
    }

    /** torch.fft.ifft. */
    public static Tensor ifft(Tensor x, int n, int dim, Norm norm) {
        return c2c(x, n, dim, norm, true);
    }

    public static Tensor ifft(Tensor x) {
        return ifft(x, -1, -1, Norm.BACKWARD);
    }

    /** torch.fft.rfft of real x: the n / 2 + 1 non-negative frequencies, complex [..., n / 2 + 1, 2]. */
    public static Tensor rfft(Tensor x, int n, int dim, Norm norm) {
        Tensor t = swapReal(x, dim);
        int len = t.dim(-1);
        int size = n > 0 ? n : len;
        Tensor y = Dft.apply(t, Math.min(len, size), size / 2 + 1, size, -1, scale(norm, size, false), true, false, FftKernels.WEIGHT_NONE);
        return swapComplex(y, dim);
    }

    public static Tensor rfft(Tensor x) {
        return rfft(x, -1, -1, Norm.BACKWARD);
    }

    /**
     * torch.fft.irfft: the real signal of length n (default 2 (m - 1)) whose spectrum's non-negative
     * half is x [..., m, 2]; the imaginary parts at frequency 0 and n / 2 are ignored.
     */
    public static Tensor irfft(Tensor x, int n, int dim, Norm norm) {
        Tensor t = swapComplex(x, dim);
        int m = t.dim(-2);
        int size = n > 0 ? n : 2 * (m - 1);
        Tensor y = Dft.apply(t, Math.min(m, size / 2 + 1), size, size, 1, scale(norm, size, true), false, true, FftKernels.WEIGHT_IN);
        return swapReal(y, dim);
    }

    public static Tensor irfft(Tensor x) {
        return irfft(x, -1, -1, Norm.BACKWARD);
    }

    /** torch.fft.fft2: the transform over the last two complex axes. */
    public static Tensor fft2(Tensor x) {
        return fft(fft(x, -1, -2, Norm.BACKWARD), -1, -1, Norm.BACKWARD);
    }

    /** torch.fft.ifft2. */
    public static Tensor ifft2(Tensor x) {
        return ifft(ifft(x, -1, -2, Norm.BACKWARD), -1, -1, Norm.BACKWARD);
    }

    /** torch.fft.rfft2: rfft over the last axis, then fft over the one before. */
    public static Tensor rfft2(Tensor x) {
        return fft(rfft(x), -1, -2, Norm.BACKWARD);
    }

    /** Frame start indices into the reflect-padded signal, as indices into the unpadded one. */
    private static int[] frameIndex(int length, int nFft, int hop, int frames, boolean center) {
        int pad = center ? nFft / 2 : 0;
        int[] idx = new int[frames * nFft];
        for (int f = 0; f < frames; f++) {
            for (int i = 0; i < nFft; i++) {
                int q = f * hop + i - pad;
                if (q < 0) {
                    q = -q;
                }
                if (q >= length) {
                    q = 2 * (length - 1) - q;
                }
                idx[f * nFft + i] = q;
            }
        }
        return idx;
    }

    /**
     * torch.stft(x, nFft, hop, window=window, center, pad_mode="reflect", normalized, onesided=True,
     * return_complex=True) as real pairs: x [L] or [B, L] gives [(B,) nFft / 2 + 1, frames, 2]. The
     * window has nFft entries (null for a rectangular one).
     */
    public static Tensor stft(Tensor x, int nFft, int hop, Tensor window, boolean center, boolean normalized) {
        boolean single = x.rank() == 1;
        Tensor xb = single ? x.unsqueeze(0) : x;
        int length = xb.dim(1);
        if (center && nFft / 2 >= length) {
            throw new IllegalArgumentException("stft: reflect padding of " + nFft / 2 + " needs a longer signal than " + length);
        }
        int padded = length + (center ? 2 * (nFft / 2) : 0);
        int frames = 1 + (padded - nFft) / hop;
        Tensor framed = xb.indexSelect(1, Tensor.of(frameIndex(length, nFft, hop, frames, center), frames * nFft)).reshape(xb.dim(0), frames, nFft);
        if (window != null) {
            framed = framed.mul(window);
        }
        Tensor spec = rfft(framed, -1, -1, normalized ? Norm.ORTHO : Norm.BACKWARD).transpose(1, 2);
        return single ? spec.squeeze(0) : spec;
    }

    /**
     * torch.istft(x, nFft, hop, window=window, center, normalized, onesided=True, length): x
     * [(B,) nFft / 2 + 1, frames, 2] back to [(B,) length] (length <= 0: as long as the frames reach,
     * less the centre padding), by overlap-add divided by the summed squared window.
     */
    public static Tensor istft(Tensor x, int nFft, int hop, Tensor window, boolean center, boolean normalized, int length) {
        boolean single = x.rank() == 3;
        Tensor xb = single ? x.unsqueeze(0) : x;
        int batch = xb.dim(0);
        int frames = xb.dim(2);
        Tensor framed = irfft(xb.transpose(1, 2), nFft, -1, normalized ? Norm.ORTHO : Norm.BACKWARD);
        Tensor win = window != null ? window : Tensor.ones(nFft);
        framed = framed.mul(win);
        int full = nFft + hop * (frames - 1);
        int[] idx = new int[batch * frames * nFft];
        for (int b = 0; b < batch; b++) {
            for (int f = 0; f < frames; f++) {
                for (int i = 0; i < nFft; i++) {
                    idx[(b * frames + f) * nFft + i] = f * hop + i;
                }
            }
        }
        Tensor index = Tensor.of(idx, batch, frames * nFft);
        Tensor y = Tensor.zeros(batch, full).scatterAdd(1, index, framed.reshape(batch, frames * nFft));
        Tensor env = Tensor.zeros(batch, full).scatterAdd(1, index, win.square().detach().unsqueeze(0).expand(Shape.of(batch * frames, nFft)).reshape(batch, frames * nFft));
        int start = center ? nFft / 2 : 0;
        int end = length > 0 ? start + length : full - (center ? nFft / 2 : 0);
        if (end > full) {
            throw new IllegalArgumentException("istft: length " + length + " is longer than the " + (full - start) + " samples the frames cover");
        }
        Tensor out = y.slice(1, start, end).div(env.slice(1, start, end));
        return single ? out.squeeze(0) : out;
    }
}
