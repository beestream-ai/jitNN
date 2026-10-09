package org.beehive.jitnn.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TuningTest {

    @Test
    void shapeClasses() {
        assertEquals("m11n10k12t01", Tuning.gemmClass(2048, 768, 3072, 0, 1));
        assertEquals("m11n10k12t01", Tuning.gemmClass(1100, 1000, 2100, 0, 1));
        assertEquals("m0n0k0t10", Tuning.gemmClass(1, 1, 1, 1, 0));
        assertEquals("768x768x1024t10", Tuning.gemmShape(768, 768, 1024, 1, 0));
    }

    @Test
    void parsesTheGemmSection() {
        String json = "{\"device\": \"x\", \"measured\": \"2026-10-09\",\n \"gemm\": {\n  \"m11n10k12t01\": \"gemm128pd\",\n  \"m6n4k9t00\": \"auto\"\n }\n}\n";
        Map<String, String> t = Tuning.parse(json, "gemm");
        assertEquals(2, t.size());
        assertEquals("gemm128pd", t.get("m11n10k12t01"));
        assertTrue(Tuning.parse(json, "conv").isEmpty());
        assertTrue(Tuning.parse("", "gemm").isEmpty());
    }
}
