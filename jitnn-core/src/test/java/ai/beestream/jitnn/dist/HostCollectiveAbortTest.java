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
package ai.beestream.jitnn.dist;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;

/** A rank that reaches the collective after another rank failed fails too, instead of waiting forever. */
class HostCollectiveAbortTest {

    @Test
    void aRankArrivingAfterTheAbortFails() throws Exception {
        HostCollective c = new HostCollective(2);
        c.abort(); // rank 0 failed before rank 1 got here
        CompletableFuture<Void> rank1 = CompletableFuture.runAsync(() -> c.barrier(1));
        ExecutionException e = assertThrows(ExecutionException.class, () -> rank1.get(10, TimeUnit.SECONDS));
        assertTrue(e.getCause() instanceof IllegalStateException, String.valueOf(e.getCause()));
    }

    @Test
    void aWaitingRankIsReleasedByTheAbort() throws Exception {
        HostCollective c = new HostCollective(2);
        CompletableFuture<Void> rank1 = CompletableFuture.runAsync(() -> c.barrier(1));
        Thread.sleep(100);
        c.abort();
        assertThrows(ExecutionException.class, () -> rank1.get(10, TimeUnit.SECONDS));
    }

    @Test
    void ranksMeetWithoutAnAbort() throws Exception {
        HostCollective c = new HostCollective(2);
        CompletableFuture<Void> rank1 = CompletableFuture.runAsync(() -> c.barrier(1));
        c.barrier(0);
        rank1.get(10, TimeUnit.SECONDS);
    }
}
