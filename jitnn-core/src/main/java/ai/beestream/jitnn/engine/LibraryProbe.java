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
package ai.beestream.jitnn.engine;

import java.util.ServiceLoader;

import ai.beestream.jitnn.engine.Native.Lib;

/**
 * Checks that a vendor library works on the current device. The implementation lives with the
 * library bindings and is found with {@link ServiceLoader}
 * ({@code META-INF/services/ai.beestream.jitnn.engine.LibraryProbe}); without one every library
 * counts as missing.
 */
public interface LibraryProbe {

    /** Null when {@code lib} works on the current device, else why not. */
    String probe(Lib lib);

    /** Null when the library function works (or has no probe of its own), else why not. */
    String probeFunction(Lib lib, String function);

    /** The registered probe. */
    static LibraryProbe get() {
        return Registered.PROBE;
    }

    /** The registered probe, loaded once. */
    final class Registered {
        static final LibraryProbe PROBE = ServiceLoader.load(LibraryProbe.class, LibraryProbe.class.getClassLoader()).findFirst().orElse(new LibraryProbe() {
            @Override
            public String probe(Lib lib) {
                return "no library bindings on the class path";
            }

            @Override
            public String probeFunction(Lib lib, String function) {
                return null;
            }
        });

        private Registered() {
        }
    }
}
