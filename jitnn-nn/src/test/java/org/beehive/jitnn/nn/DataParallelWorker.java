package org.beehive.jitnn.nn;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

import org.beehive.jitnn.dist.NcclCollective;
import org.beehive.jitnn.dist.ProcessGroup;
import org.beehive.jitnn.engine.Engine;

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
            Engine.reset();
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
