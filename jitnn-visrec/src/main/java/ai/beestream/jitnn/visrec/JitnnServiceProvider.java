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

import java.io.InputStream;
import java.util.Properties;

import javax.visrec.spi.ImageFactoryService;
import javax.visrec.spi.ImplementationService;
import javax.visrec.spi.ServiceProvider;

/**
 * jitNN's JSR 381 service provider, found by {@code ServiceProvider.current()} through the
 * {@code ServiceLoader}. Its priority (100) puts it ahead of the reference implementation (0) when
 * both are on the class path.
 */
public final class JitnnServiceProvider extends ServiceProvider {

    private static final ImageFactoryService IMAGES = new JitnnImageFactoryService();

    private static final ImplementationService IMPLEMENTATION = new ImplementationService() {
        @Override
        public String getName() {
            return "jitNN";
        }

        @Override
        public String getVersion() {
            return VERSION;
        }
    };

    private static final String VERSION = readVersion();

    @Override
    public int getPriority() {
        return 100;
    }

    @Override
    public ImageFactoryService getImageFactoryService() {
        return IMAGES;
    }

    @Override
    public ImplementationService getImplementationService() {
        return IMPLEMENTATION;
    }

    private static String readVersion() {
        try (InputStream in = JitnnServiceProvider.class.getResourceAsStream("/META-INF/maven/ai.beestream/jitnn-visrec/pom.properties")) {
            if (in != null) {
                Properties p = new Properties();
                p.load(in);
                return p.getProperty("version", "dev");
            }
        } catch (java.io.IOException e) {
            // fall through
        }
        return "dev";
    }
}
