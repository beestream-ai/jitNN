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
package ai.beestream.jitnn.nativeimage;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Stream;

import org.graalvm.nativeimage.hosted.Feature;
import org.graalvm.nativeimage.hosted.RuntimeReflection;
import org.graalvm.nativeimage.hosted.RuntimeSerialization;

/**
 * Registers what TornadoVM needs to compile jitNN's kernels inside a GraalVM native image. It is
 * named in {@code META-INF/native-image/ai.beestream/jitnn/native-image.properties}, so any image
 * built with this jar on its class path picks it up; on a JVM the class is never loaded.
 *
 * <p>An operator adds a kernel with {@code TaskGraph.task(id, Kernels::method, ...)}. TornadoVM
 * reads the method reference through the lambda's {@code writeReplace}, loads the class it names,
 * finds the method with {@code getDeclaredMethod} and parses the class's bytecode. The closed-world
 * analysis sees none of this, so for every jitNN class on the class path (application code using
 * jitNN's {@code CustomOp} included) this feature:
 *
 * <ul>
 *   <li>registers the class and its declared methods for reflection, except the {@code cpu}
 *       backend's package and other classes that use the Vector API, which hold no TornadoVM
 *       kernels (their code stays reachable only through the cpu backend itself);
 *   <li>registers every class that javac gave a {@code $deserializeLambda$} method as
 *       lambda-capturing, so its serializable lambdas keep {@code writeReplace};
 *   <li>registers the types of TornadoVM's API that kernels use (arrays, vectors, half floats,
 *       math), which TornadoVM looks up by name.
 * </ul>
 *
 * The class files themselves are included as resources by the {@code reachability-metadata.json}
 * next to the properties file.
 */
public final class JitnnFeature implements Feature {

    private static final String[] PACKAGES = { "ai/beestream/jitnn/" };
    private static final String SELF = "ai/beestream/jitnn/nativeimage/";
    /** The {@code cpu} backend: Vector API and FFM code, no TornadoVM kernels. */
    private static final String CPU = "ai/beestream/jitnn/cpu/";
    private static final byte[] VECTOR_API = "jdk/incubator/vector".getBytes(StandardCharsets.UTF_8);
    private static final byte[] DESERIALIZE_LAMBDA = "$deserializeLambda$".getBytes(StandardCharsets.UTF_8);

    @Override
    public String getDescription() {
        return "Registers jitNN's TornadoVM kernels and task lambdas for reflection and serialization";
    }

    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        int classes = 0;
        int capturing = 0;
        for (Path entry : access.getApplicationClassPath()) {
            for (ClassFile file : classFiles(entry)) {
                Class<?> type = access.findClassByName(file.name());
                if (type == null) {
                    continue;
                }
                if (file.name().startsWith(CPU.replace('/', '.'))) {
                    continue;
                }
                RuntimeReflection.register(type);
                if (indexOf(file.bytes(), VECTOR_API) < 0) {
                    try {
                        RuntimeReflection.register(type.getDeclaredMethods());
                    } catch (LinkageError e) {
                        // a signature names a class that is not on the image's class path
                    }
                }
                classes++;
                if (indexOf(file.bytes(), DESERIALIZE_LAMBDA) >= 0) {
                    RuntimeSerialization.registerLambdaCapturingClass(type);
                    capturing++;
                }
            }
        }
        if (classes == 0) {
            throw new IllegalStateException("the jitNN native-image feature found no jitNN classes on the application class path");
        }
        int api = registerTornadoApi(access);
        if (Boolean.getBoolean("jitnn.nativeimage.verbose")) {
            System.out.printf("jitNN feature: %d classes, %d lambda-capturing, %d TornadoVM API types%n", classes, capturing, api);
        }
    }

    /** Packages of TornadoVM's API whose types kernels take or call, which TornadoVM looks up by name. */
    private static final String[] TORNADO_API = { "uk/ac/manchester/tornado/api/types/", "uk/ac/manchester/tornado/api/math/", "uk/ac/manchester/tornado/api/enums/",
            "uk/ac/manchester/tornado/api/utils/", "uk/ac/manchester/tornado/api/KernelContext.class", "uk/ac/manchester/tornado/api/TileContext.class" };

    /**
     * Registers the types kernels use from TornadoVM's API module (arrays, vectors, half floats,
     * math), which TornadoVM resolves by name while it compiles a kernel. Returns how many.
     */
    private static int registerTornadoApi(BeforeAnalysisAccess access) {
        Class<?> context = access.findClassByName("uk.ac.manchester.tornado.api.KernelContext");
        if (context == null || context.getModule().getLayer() == null) {
            return 0;
        }
        Module module = context.getModule();
        int count = 0;
        try (java.lang.module.ModuleReader reader = module.getLayer().configuration().findModule(module.getName()).orElseThrow().reference().open();
                Stream<String> names = reader.list()) {
            for (String path : (Iterable<String>) names::iterator) {
                if (!path.endsWith(".class") || path.endsWith("module-info.class")) {
                    continue;
                }
                boolean wanted = false;
                for (String p : TORNADO_API) {
                    wanted |= path.startsWith(p);
                }
                if (wanted) {
                    Class<?> type = access.findClassByName(className(path));
                    if (type != null) {
                        RuntimeReflection.register(type);
                        count++;
                    }
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot list TornadoVM's API module", e);
        }
        return count;
    }

    private record ClassFile(String name, byte[] bytes) {
    }

    private static List<ClassFile> classFiles(Path entry) {
        List<ClassFile> files = new ArrayList<>();
        try {
            if (Files.isDirectory(entry)) {
                try (Stream<Path> walk = Files.walk(entry)) {
                    for (Path p : (Iterable<Path>) walk::iterator) {
                        String relative = entry.relativize(p).toString().replace('\\', '/');
                        if (isClassFile(relative)) {
                            files.add(new ClassFile(className(relative), Files.readAllBytes(p)));
                        }
                    }
                }
            } else if (entry.toString().endsWith(".jar") && Files.isRegularFile(entry)) {
                try (JarFile jar = new JarFile(entry.toFile())) {
                    for (JarEntry e : (Iterable<JarEntry>) jar.stream()::iterator) {
                        if (isClassFile(e.getName())) {
                            try (InputStream in = jar.getInputStream(e)) {
                                files.add(new ClassFile(className(e.getName()), in.readAllBytes()));
                            }
                        }
                    }
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot scan " + entry + " for jitNN classes", e);
        }
        return files;
    }

    private static boolean isClassFile(String path) {
        if (!path.endsWith(".class") || path.endsWith("module-info.class") || path.startsWith(SELF)) {
            return false;
        }
        for (String p : PACKAGES) {
            if (path.startsWith(p)) {
                return true;
            }
        }
        return false;
    }

    private static String className(String path) {
        return path.substring(0, path.length() - ".class".length()).replace('/', '.');
    }

    private static int indexOf(byte[] haystack, byte[] needle) {
        outer:
        for (int i = 0; i <= haystack.length - needle.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (haystack[i + j] != needle[j]) {
                    continue outer;
                }
            }
            return i;
        }
        return -1;
    }
}
