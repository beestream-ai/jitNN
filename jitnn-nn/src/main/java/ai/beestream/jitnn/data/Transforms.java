package ai.beestream.jitnn.data;

import java.util.SplittableRandom;

/**
 * Per-sample transforms for {@link DataLoader}, applied in place to one CHW sample on the host:
 * the usual image augmentations and normalisation. A transform draws from the random stream it is
 * given, which the loader derives from its seed, the epoch and the sample, so batches do not depend
 * on the number of workers.
 */
public final class Transforms {

    private Transforms() {
    }

    /** Transforms one sample of {@code shape} at {@code x[at ..]} in place. */
    @FunctionalInterface
    public interface Transform {
        void apply(float[] x, int at, int[] shape, SplittableRandom random);

        /** This transform, then {@code next}. */
        default Transform then(Transform next) {
            return (x, at, shape, random) -> {
                apply(x, at, shape, random);
                next.apply(x, at, shape, random);
            };
        }
    }

    /** Mirrors the image left to right with probability 1/2 (torchvision's RandomHorizontalFlip). */
    public static Transform randomHorizontalFlip() {
        return (x, at, shape, random) -> {
            if (!random.nextBoolean()) {
                return;
            }
            int h = shape[shape.length - 2];
            int w = shape[shape.length - 1];
            int planes = planes(shape);
            for (int p = 0; p < planes; p++) {
                for (int r = 0; r < h; r++) {
                    int row = at + (p * h + r) * w;
                    for (int c = 0; c < w / 2; c++) {
                        float t = x[row + c];
                        x[row + c] = x[row + w - 1 - c];
                        x[row + w - 1 - c] = t;
                    }
                }
            }
        };
    }

    /**
     * Pads the image with {@code pad} zeros on every side and crops the original size back at a
     * random offset (torchvision's RandomCrop(size, padding)).
     */
    public static Transform randomCrop(int pad) {
        return (x, at, shape, random) -> {
            int h = shape[shape.length - 2];
            int w = shape[shape.length - 1];
            int dy = random.nextInt(2 * pad + 1) - pad;
            int dx = random.nextInt(2 * pad + 1) - pad;
            if (dy == 0 && dx == 0) {
                return;
            }
            float[] plane = new float[h * w];
            for (int p = 0; p < planes(shape); p++) {
                int base = at + p * h * w;
                System.arraycopy(x, base, plane, 0, h * w);
                for (int r = 0; r < h; r++) {
                    for (int c = 0; c < w; c++) {
                        int sr = r + dy;
                        int sc = c + dx;
                        x[base + r * w + c] = sr >= 0 && sr < h && sc >= 0 && sc < w ? plane[sr * w + sc] : 0.0f;
                    }
                }
            }
        };
    }

    /** (x - mean[c]) / std[c] per channel (torchvision's Normalize). */
    public static Transform normalize(float[] mean, float[] std) {
        return (x, at, shape, random) -> {
            int channels = shape[0];
            int plane = 1;
            for (int d = 1; d < shape.length; d++) {
                plane *= shape[d];
            }
            if (mean.length != channels || std.length != channels) {
                throw new IllegalArgumentException("normalize: " + mean.length + " means for " + channels + " channels");
            }
            for (int c = 0; c < channels; c++) {
                float inv = 1.0f / std[c];
                for (int i = 0; i < plane; i++) {
                    int k = at + c * plane + i;
                    x[k] = (x[k] - mean[c]) * inv;
                }
            }
        };
    }

    private static int planes(int[] shape) {
        int p = 1;
        for (int d = 0; d < shape.length - 2; d++) {
            p *= shape[d];
        }
        return p;
    }
}
