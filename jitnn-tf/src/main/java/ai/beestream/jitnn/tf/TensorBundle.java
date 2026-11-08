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
package ai.beestream.jitnn.tf;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import ai.beestream.jitnn.onnx.Proto;

/**
 * A TensorFlow checkpoint ({@code variables.index} + {@code variables.data-NNNNN-of-NNNNN}): the
 * index is an SSTable mapping each tensor name to a BundleEntryProto (type, shape, shard, offset,
 * size); the data shards hold the raw little-endian bytes.
 */
final class TensorBundle {

    private static final long MAGIC = 0xdb4775248b80fb57L;

    private record Entry(int dtype, long[] dims, int shard, long offset, long size) {
    }

    private final Map<String, Entry> entries = new LinkedHashMap<>();
    private final ByteBuffer[] shards;

    /** {@code prefix} is the path without suffix, e.g. {@code <model>/variables/variables}. */
    TensorBundle(Path prefix) throws IOException {
        ByteBuffer index = map(Path.of(prefix + ".index"));
        readTable(index);
        // the shard count is in the data files' names: <prefix>.data-00000-of-NNNNN
        int shardCount = 1;
        String base = prefix.getFileName() + ".data-";
        try (java.util.stream.Stream<Path> files = Files.list(prefix.toAbsolutePath().getParent())) {
            for (Path f : (Iterable<Path>) files::iterator) {
                String n = f.getFileName().toString();
                if (n.startsWith(base) && n.contains("-of-")) {
                    shardCount = Math.max(shardCount, Integer.parseInt(n.substring(n.lastIndexOf("-of-") + 4)));
                }
            }
        }
        shards = new ByteBuffer[shardCount];
        for (int s = 0; s < shardCount; s++) {
            Path data = Path.of(String.format("%s.data-%05d-of-%05d", prefix, s, shardCount));
            if (Files.exists(data)) {
                shards[s] = map(data);
            }
        }
    }

    Iterable<String> names() {
        return entries.keySet();
    }

    boolean contains(String name) {
        return entries.containsKey(name);
    }

    TfGraph.Tensor get(String name) {
        Entry e = entries.get(name);
        if (e == null) {
            throw new IllegalArgumentException("no tensor '" + name + "' in the checkpoint");
        }
        ByteBuffer shard = shards[e.shard];
        if (shard == null) {
            throw new IllegalStateException("checkpoint shard " + e.shard + " of '" + name + "' is missing");
        }
        ByteBuffer raw = shard.slice((int) e.offset, (int) e.size).order(ByteOrder.LITTLE_ENDIAN);
        return new TfGraph.Tensor(e.dtype, e.dims, raw);
    }

    private static ByteBuffer map(Path file) throws IOException {
        try (RandomAccessFile f = new RandomAccessFile(file.toFile(), "r"); FileChannel ch = f.getChannel()) {
            return ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()).order(ByteOrder.LITTLE_ENDIAN);
        }
    }

    // ---------------------------------------------------------------- SSTable

    private static long[] varint(ByteBuffer b) {
        long result = 0;
        int shift = 0;
        while (true) {
            byte x = b.get();
            result |= (long) (x & 0x7F) << shift;
            if ((x & 0x80) == 0) {
                return new long[] { result };
            }
            shift += 7;
        }
    }

    private static long v(ByteBuffer b) {
        return varint(b)[0];
    }

    private void readTable(ByteBuffer file) throws IOException {
        int len = file.limit();
        // footer: metaindex handle, index handle (varints, padded to 40 bytes), then the 8-byte magic
        ByteBuffer footer = file.slice(len - 48, 48).order(ByteOrder.LITTLE_ENDIAN);
        if (footer.getLong(40) != MAGIC) {
            throw new IOException("not a TensorFlow checkpoint index: bad table magic");
        }
        v(footer);
        v(footer);
        long indexOffset = v(footer);
        long indexSize = v(footer);
        for (Map.Entry<String, ByteBuffer> ie : block(file, indexOffset, indexSize).entrySet()) {
            ByteBuffer handle = ie.getValue();
            long off = v(handle);
            long size = v(handle);
            for (Map.Entry<String, ByteBuffer> de : block(file, off, size).entrySet()) {
                // the empty key holds the BundleHeaderProto, not a tensor
                if (!de.getKey().isEmpty()) {
                    entries.put(de.getKey(), entry(de.getValue()));
                }
            }
        }
    }

    /** The key/value pairs of one uncompressed block (prefix-compressed keys, restart array at the end). */
    private static Map<String, ByteBuffer> block(ByteBuffer file, long offset, long size) throws IOException {
        int type = file.get((int) (offset + size)) & 0xFF;
        if (type != 0) {
            throw new IOException("compressed checkpoint blocks (type " + type + ") are not supported");
        }
        ByteBuffer b = file.slice((int) offset, (int) size).order(ByteOrder.LITTLE_ENDIAN);
        int restarts = b.getInt((int) size - 4);
        int limit = (int) size - 4 - 4 * restarts;
        Map<String, ByteBuffer> out = new LinkedHashMap<>();
        byte[] key = new byte[0];
        b.position(0);
        while (b.position() < limit) {
            int shared = (int) v(b);
            int nonShared = (int) v(b);
            int valueLen = (int) v(b);
            byte[] k = new byte[shared + nonShared];
            System.arraycopy(key, 0, k, 0, shared);
            b.get(k, shared, nonShared);
            key = k;
            ByteBuffer value = b.slice(b.position(), valueLen).order(ByteOrder.LITTLE_ENDIAN);
            b.position(b.position() + valueLen);
            out.put(new String(k, StandardCharsets.UTF_8), value);
        }
        return out;
    }

    /** A BundleEntryProto: dtype 1, shape 2, shard_id 3, offset 4, size 5. */
    private static Entry entry(ByteBuffer value) {
        Proto p = new Proto(value);
        int dtype = 0;
        long[] dims = new long[0];
        int shard = 0;
        long off = 0;
        long size = 0;
        while (p.next()) {
            switch (p.field()) {
                case 1 -> dtype = (int) p.varint();
                case 2 -> {
                    long[] d = TfGraph.shape(p.message());
                    dims = d == null ? new long[0] : d;
                }
                case 3 -> shard = (int) p.varint();
                case 4 -> off = p.varint();
                case 5 -> size = p.varint();
                default -> p.skip();
            }
        }
        return new Entry(dtype, dims, shard, off, size);
    }
}
