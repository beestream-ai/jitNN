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
package ai.beestream.jitnn;

/**
 * A failure of the device or the runtime: a plan that TornadoVM could not compile or run, a
 * vendor library call that failed, a native library of the CPU backend that could not be called.
 * The cause is the underlying exception. Misuse (a released tensor, mismatched shapes) is reported
 * with {@link IllegalStateException} and {@link IllegalArgumentException}; files that cannot be read
 * with {@link java.io.IOException}.
 */
public class JitnnException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    /** A failure described by {@code message}, caused by {@code cause}. */
    public JitnnException(String message, Throwable cause) {
        super(message, cause);
    }

    /** A failure described by {@code message}. */
    public JitnnException(String message) {
        super(message);
    }

    /** True when {@code e}, or one of its causes, comes from TornadoVM. */
    static boolean fromTornadoVM(Throwable e) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t.getClass().getName().startsWith("uk.ac.manchester.tornado.")) {
                return true;
            }
            if (t.getCause() == t) {
                break;
            }
        }
        return false;
    }

    /**
     * {@code e} as a {@code JitnnException} when it is a device or runtime failure (one from
     * TornadoVM), else {@code e} itself.
     */
    public static RuntimeException wrap(String what, RuntimeException e) {
        if (e instanceof JitnnException || !fromTornadoVM(e)) {
            return e;
        }
        String m = e.getMessage();
        return new JitnnException(what + ": " + e.getClass().getSimpleName() + (m == null ? "" : ": " + m.lines().findFirst().orElse("")), e);
    }
}
