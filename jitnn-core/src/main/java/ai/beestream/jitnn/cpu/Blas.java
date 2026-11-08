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
package ai.beestream.jitnn.cpu;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import ai.beestream.jitnn.JitnnConfig;

/**
 * A native BLAS's {@code sgemm} (Intel MKL, OpenBLAS, Arm Performance Libraries, Apple Accelerate),
 * called through the foreign function API: {@code -Djitnn.cpu.blas=<library file>} names the
 * library, {@code auto} (the default) looks for one on the system's library path, {@code off} uses
 * none. The library runs {@link Cpu#threads()} threads of its own; a call from inside a parallel
 * loop does not use it (its threads would oversubscribe the cores).
 */
public final class Blas {

    private static final int ROW_MAJOR = 101;
    private static final int NO_TRANS = 111;
    private static final int TRANS = 112;

    /** Libraries {@code auto} tries, in order. */
    private static final String[] AUTO = { "libmkl_rt.so.3", "libmkl_rt.so.2", "libmkl_rt.so", "libopenblas.so.0", "libopenblas.so", "libarmpl_lp64_mp.so", "libarmpl_lp64.so",
            "/System/Library/Frameworks/Accelerate.framework/Accelerate" };

    private static final String[] OPENMP = { "libgomp.so.1", "libomp.so", "libiomp5.so" };
    /** omp_set_num_threads of an OpenMP library (ArmPL), or null. */
    private static MethodHandle ompSetThreads;
    private static final ThreadLocal<Boolean> THREADS_SET = ThreadLocal.withInitial(() -> Boolean.FALSE);

    private static final MethodHandle SGEMM;
    private static final String NAME;

    static {
        MethodHandle sgemm = null;
        String name = null;
        String p = JitnnConfig.current().get(JitnnConfig.CPU_BLAS);
        if (!p.equals("off")) {
            for (String candidate : p.equals("auto") ? AUTO : new String[] { p }) {
                try {
                    sgemm = bind(candidate);
                    if (sgemm != null) {
                        name = candidate;
                        break;
                    }
                } catch (Throwable e) {
                    if (!p.equals("auto")) {
                        System.err.println("[jitnn] cannot use the BLAS library " + candidate + ": " + e);
                    }
                }
            }
        }
        SGEMM = sgemm;
        NAME = name;
        if (NAME != null && JitnnConfig.current().get(JitnnConfig.VERBOSE)) {
            System.err.println("[jitnn] the CPU backend calls sgemm of " + NAME + " (" + Cpu.threads() + " threads)");
        }
    }

    private Blas() {
    }

    private static MethodHandle bind(String library) {
        Linker linker = Linker.nativeLinker();
        if (library.contains("armpl")) {
            // Arm Performance Libraries call OpenMP (omp_get_max_threads ...) without depending on a
            // runtime: the application links one. Load GCC's where the library can see it
            openMpGlobal(linker);
        }
        SymbolLookup lib = library.contains("/") && Files.exists(Path.of(library)) ? SymbolLookup.libraryLookup(Path.of(library), Arena.global())
                : SymbolLookup.libraryLookup(library, Arena.global());
        Optional<MemorySegment> fn = lib.find("cblas_sgemm").or(() -> lib.find("scipy_cblas_sgemm"));
        if (fn.isEmpty()) {
            return null;
        }
        ValueLayout.OfInt i = ValueLayout.JAVA_INT;
        ValueLayout.OfFloat f = ValueLayout.JAVA_FLOAT;
        ValueLayout a = ValueLayout.ADDRESS;
        MethodHandle h = linker.downcallHandle(fn.get(), FunctionDescriptor.ofVoid(i, i, i, i, i, i, f, a, i, a, i, f, a, i));
        setThreads(linker, lib, Cpu.threads());
        return h;
    }

    private static void setThreads(Linker linker, SymbolLookup lib, int threads) {
        for (String s : new String[] { "MKL_Set_Num_Threads", "openblas_set_num_threads", "scipy_openblas_set_num_threads" }) {
            Optional<MemorySegment> fn = lib.find(s);
            if (fn.isPresent()) {
                try {
                    linker.downcallHandle(fn.get(), FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT)).invokeExact(threads);
                } catch (Throwable e) {
                    throw new IllegalStateException(e);
                }
                return;
            }
        }
        // an OpenMP library (ArmPL): the thread count is per calling thread, set before its first call
        Optional<MemorySegment> fn = lib.find("omp_set_num_threads").or(() -> openMp().flatMap(o -> o.find("omp_set_num_threads")));
        if (fn.isPresent()) {
            ompSetThreads = linker.downcallHandle(fn.get(), FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
        }
    }

    /** dlopen(RTLD_NOW | RTLD_GLOBAL) of the first OpenMP runtime found, so later libraries resolve against it (Linux). */
    private static void openMpGlobal(Linker linker) {
        Optional<MemorySegment> dlopen = linker.defaultLookup().find("dlopen");
        if (dlopen.isEmpty()) {
            return;
        }
        MethodHandle h = linker.downcallHandle(dlopen.get(), FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        try (Arena arena = Arena.ofConfined()) {
            for (String name : OPENMP) {
                MemorySegment handle = (MemorySegment) h.invokeExact(arena.allocateFrom(name), 0x2 | 0x100);
                if (!handle.equals(MemorySegment.NULL)) {
                    return;
                }
            }
        } catch (Throwable e) {
            throw new IllegalStateException(e);
        }
    }

    private static Optional<SymbolLookup> openMp() {
        for (String name : OPENMP) {
            try {
                return Optional.of(SymbolLookup.libraryLookup(name, Arena.global()));
            } catch (IllegalArgumentException e) {
                // not this runtime
            }
        }
        return Optional.empty();
    }

    /** True when a BLAS library was found and this call is not inside a parallel loop. */
    public static boolean usable() {
        return SGEMM != null && !Cpu.inside();
    }

    /** The library's file, or null. */
    public static String library() {
        return NAME;
    }

    /**
     * True when {@code m} (a {@link Gemm.Matrix} of rows x depth) can be passed as a BLAS operand:
     * one of its strides is 1, the other fits in an int, and it lies in native memory.
     */
    public static boolean operand(Gemm.Matrix m) {
        return m.seg().isNative() && (m.qs() == 1 || m.xs() == 1) && Math.max(m.xs(), m.qs()) <= Integer.MAX_VALUE;
    }

    /**
     * C = A B (+ beta C): A is m x k and B k x n as {@link Gemm.Matrix} operands (B read by columns:
     * its rows are B's columns), C row-major with leading dimension {@code ldc}.
     */
    public static void sgemm(Gemm.Matrix a, Gemm.Matrix b, int m, int n, int k, MemorySegment c, long cOff, long ldc, float beta) {
        if (ompSetThreads != null && !THREADS_SET.get()) {
            try {
                ompSetThreads.invokeExact(Cpu.threads());
            } catch (Throwable e) {
                throw new IllegalStateException(e);
            }
            THREADS_SET.set(Boolean.TRUE);
        }
        // A: row x, depth q at x * xs + q * qs. Row-major [m, k] when qs == 1 (lda = xs), else A^T
        boolean ta = a.qs() != 1;
        long lda = ta ? a.qs() : a.xs();
        // B as k x n: depth q, column x at x * xs + q * qs. Row-major [k, n] when xs == 1 (ldb = qs)
        boolean tb = b.xs() != 1;
        long ldb = tb ? b.xs() : b.qs();
        // a dimension of 1 leaves its stride arbitrary; BLAS still wants it at least the row length
        lda = Math.max(lda, Math.max(1, ta ? m : k));
        ldb = Math.max(ldb, Math.max(1, tb ? k : n));
        try {
            SGEMM.invokeExact(ROW_MAJOR, ta ? TRANS : NO_TRANS, tb ? TRANS : NO_TRANS, m, n, k, 1f, a.seg().asSlice(a.off() * 4), (int) lda, b.seg().asSlice(b.off() * 4), (int) ldb, beta,
                    c.asSlice(cOff * 4), (int) ldc);
        } catch (Throwable e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * C = A B (+ {@code epilogue}) on the BLAS library or on {@code kernel} (the CPU backend's own
     * GEMM for the same product), whichever the {@link Race} of {@code key} picks; {@code kernel}
     * alone when there is no library or an operand does not suit it.
     */
    public static void multiply(String key, Gemm.Matrix a, Gemm.Matrix b, int m, int n, int k, MemorySegment c, long cOff, long ldc, MemorySegment bias, long biasOff, int epilogue,
            Runnable kernel) {
        if (!operand(a) || !operand(b) || !c.isNative() || ldc > Integer.MAX_VALUE || m == 0 || n == 0) {
            kernel.run();
            return;
        }
        Race.run(key, kernel, usable() ? () -> {
            sgemm(a, b, m, n, k, c, cOff, ldc, (epilogue & Gemm.ACCUMULATE) != 0 ? 1f : 0f);
            Gemm.epilogue(c, cOff, ldc, m, n, bias, biasOff, epilogue);
        } : null, null);
    }

    private static final ThreadLocal<MemorySegment[]> SCRATCH = ThreadLocal.withInitial(() -> new MemorySegment[2]);

    /** A native scratch segment of at least {@code floats} floats for this thread, slot 0 or 1 (contents undefined). */
    public static MemorySegment scratch(int slot, long floats) {
        MemorySegment[] s = SCRATCH.get();
        if (s[slot] == null || s[slot].byteSize() < floats * 4) {
            s[slot] = Arena.ofAuto().allocate(Math.max(floats, 1) * 4, 64);
        }
        return s[slot];
    }
}
