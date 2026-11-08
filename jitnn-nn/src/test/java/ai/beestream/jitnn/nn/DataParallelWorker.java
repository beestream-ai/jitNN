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
package ai.beestream.jitnn.nn;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.dist.NcclCollective;
import ai.beestream.jitnn.dist.ProcessGroup;

/**
 * One rank of {@link DataParallelMultiProcessAccelTest} in its own JVM: joins the NCCL
 * communicator, runs {@link DataParallelAccelTest#train}, and writes its final parameters to a file.
 * Arguments: unique id, world size, rank, device, output file.
 */
public final class DataParallelWorker {

    private DataParallelWorker() {
    }

    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws IOException {
        String id = args[0];
        int world = Integer.parseInt(args[1]);
        int rank = Integer.parseInt(args[2]);
        String device = args[3];
        List<Object[]> out;
        try (NcclCollective nccl = new NcclCollective(id, world, rank, device)) {
            out = ProcessGroup.run(new String[] { device }, nccl, (r, c) -> DataParallelAccelTest.train(r, c, world));
            JitnnRuntime.reset();
        }
        List<float[]> params = (List<float[]>) out.get(0)[1];
        try (DataOutputStream o = new DataOutputStream(new FileOutputStream(args[4]))) {
            o.writeInt(params.size());
            for (float[] p : params) {
                o.writeInt(p.length);
                for (float v : p) {
                    o.writeFloat(v);
                }
            }
        }
        System.exit(0);
    }
}
