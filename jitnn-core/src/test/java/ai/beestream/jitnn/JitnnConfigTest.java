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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

import org.junit.jupiter.api.Test;

class JitnnConfigTest {

    @Test
    void readsTypedValuesFromProperties() {
        Properties p = new Properties();
        p.setProperty("jitnn.plan.cache", "64");
        p.setProperty("jitnn.cudaGraph", "FALSE");
        p.setProperty("jitnn.autotune.margin", "0.1");
        p.setProperty("jitnn.native", "cublas,cudnn");
        p.setProperty("jitnn.test.device", "0:1"); // not an option: ignored
        JitnnConfig c = JitnnConfig.fromProperties(p);
        assertEquals(64, c.get(JitnnConfig.PLAN_CACHE));
        assertFalse(c.get(JitnnConfig.CUDA_GRAPH));
        assertEquals(0.1, c.get(JitnnConfig.AUTOTUNE_MARGIN));
        assertEquals("cublas,cudnn", c.get(JitnnConfig.NATIVE));
        assertTrue(c.isSet(JitnnConfig.PLAN_CACHE));
        assertFalse(c.isSet(JitnnConfig.CONCURRENCY));
        assertEquals(1, c.get(JitnnConfig.CONCURRENCY));
        assertNull(c.get(JitnnConfig.CPU_THREADS));
    }

    @Test
    void invalidValuesNameTheKey() {
        Properties p = new Properties();
        p.setProperty("jitnn.plan.cache", "lots");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> JitnnConfig.fromProperties(p));
        assertTrue(e.getMessage().startsWith("jitnn.plan.cache:"), e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> JitnnConfig.builder().set("jitnn.conv", "fast"));
        assertThrows(IllegalArgumentException.class, () -> JitnnConfig.builder().set("jitnn.fuse", "yes"));
        assertThrows(IllegalArgumentException.class, () -> JitnnConfig.builder().set("jitnn.noSuchOption", "1"));
    }

    @Test
    void builderAndCopiesAreIndependent() {
        JitnnConfig a = JitnnConfig.builder().set(JitnnConfig.PLAN_CACHE, 32).set("jitnn.winograd", "2").build();
        JitnnConfig b = a.with(JitnnConfig.PLAN_CACHE, null);
        assertEquals(32, a.get(JitnnConfig.PLAN_CACHE));
        assertEquals(16, b.get(JitnnConfig.PLAN_CACHE));
        assertEquals("2", b.get(JitnnConfig.WINOGRAD));
        assertEquals(a, JitnnConfig.fromProperties(a.toProperties()));
    }

    @Test
    void onlyRunTimeOptionsChangeAfterStart() {
        JitnnConfig.current();
        assertThrows(IllegalArgumentException.class, () -> JitnnConfig.change(JitnnConfig.PLAN_CACHE, 8));
        assertFalse(JitnnConfig.defaultIfUnset(JitnnConfig.PLAN_CACHE, 8));
        JitnnConfig started = JitnnConfig.current();
        assertThrows(IllegalStateException.class, () -> JitnnConfig.install(started.with(JitnnConfig.PLAN_CACHE, started.get(JitnnConfig.PLAN_CACHE) + 1)));
        JitnnConfig.install(started); // the same configuration: allowed
        String before = JitnnConfig.current().get(JitnnConfig.TRANSPOSE);
        try {
            JitnnConfig.change(JitnnConfig.TRANSPOSE, "16");
            assertEquals("16", JitnnConfig.current().get(JitnnConfig.TRANSPOSE));
        } finally {
            JitnnConfig.change(JitnnConfig.TRANSPOSE, before);
        }
    }

    @Test
    void keysAreUniqueAndDocumented() {
        Set<String> keys = new HashSet<>();
        for (JitnnConfig.Option<?> o : JitnnConfig.options()) {
            assertTrue(keys.add(o.key()), "duplicate " + o.key());
            assertTrue(o.key().startsWith("jitnn."), o.key());
            assertFalse(o.description().isBlank(), o.key());
            assertFalse(o.defaultText().isBlank(), o.key());
            if (o.defaultValue() != null && !o.choices().isEmpty()) {
                assertTrue(o.choices().contains((String) o.defaultValue()), o.key());
            }
        }
    }
}
