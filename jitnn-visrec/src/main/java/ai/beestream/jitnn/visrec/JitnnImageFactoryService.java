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

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Path;
import java.util.Optional;

import javax.imageio.ImageIO;
import javax.visrec.ImageFactory;
import javax.visrec.spi.ImageFactoryService;

/** Reads images as {@link BufferedImage} through {@link ImageIO}, the only image type this implementation classifies. */
final class JitnnImageFactoryService implements ImageFactoryService {

    static final ImageFactory<BufferedImage> BUFFERED_IMAGES = new ImageFactory<>() {
        @Override
        public BufferedImage getImage(Path path) throws IOException {
            return check(ImageIO.read(path.toFile()), path);
        }

        @Override
        public BufferedImage getImage(URL file) throws IOException {
            return check(ImageIO.read(file), file);
        }

        @Override
        public BufferedImage getImage(InputStream file) throws IOException {
            return check(ImageIO.read(file), "stream");
        }

        private BufferedImage check(BufferedImage image, Object source) throws IOException {
            if (image == null) {
                throw new IOException("not an image ImageIO can read: " + source);
            }
            return image;
        }
    };

    @Override
    @SuppressWarnings("unchecked")
    public <T> Optional<ImageFactory<T>> getByImageType(Class<T> imageCls) {
        return imageCls == BufferedImage.class ? Optional.of((ImageFactory<T>) BUFFERED_IMAGES) : Optional.empty();
    }
}
