/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.visrec;

import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

/** Images to network inputs, as Deep Netts prepares them: resized, RGB (or grey) in [0, 1], channels first. */
final class Images {

    private Images() {
    }

    /** {@code image} scaled to {@code width} x {@code height}, as [channels, height, width] in [0, 1] (3 channels: R, G, B; 1: luminance). */
    static float[] pixels(BufferedImage image, int width, int height, int channels) {
        BufferedImage scaled = image;
        if (image.getWidth() != width || image.getHeight() != height || image.getType() != BufferedImage.TYPE_INT_RGB) {
            scaled = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            var g = scaled.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(image, 0, 0, width, height, null);
            g.dispose();
        }
        float[] out = new float[channels * height * width];
        int plane = height * width;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = scaled.getRGB(x, y);
                float r = ((rgb >> 16) & 0xff) / 255f;
                float gr = ((rgb >> 8) & 0xff) / 255f;
                float b = (rgb & 0xff) / 255f;
                int i = y * width + x;
                if (channels == 1) {
                    out[i] = 0.299f * r + 0.587f * gr + 0.114f * b;
                } else {
                    out[i] = r;
                    out[plane + i] = gr;
                    out[2 * plane + i] = b;
                }
            }
        }
        return out;
    }
}
