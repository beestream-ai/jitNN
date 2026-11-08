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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.jupiter.api.Test;

/**
 * The option tables of {@code docs/CONFIGURATION.md} are generated from {@link JitnnConfig}: this
 * test fails when they differ, and with {@code -Djitnn.docs.write=true} rewrites them.
 */
class ConfigurationDocTest {

    private static final Pattern GENERATED = Pattern.compile("<!-- generated: (\\w+) -->\\n(.*?)<!-- end generated -->", Pattern.DOTALL);

    /** The document with every generated section replaced by the tables of the current options. */
    static String generate(String doc) {
        Matcher m = GENERATED.matcher(doc);
        StringBuilder out = new StringBuilder();
        while (m.find()) {
            String table = JitnnConfig.markdownTable(JitnnConfig.Section.valueOf(m.group(1)));
            m.appendReplacement(out, Matcher.quoteReplacement("<!-- generated: " + m.group(1) + " -->\n" + table + "<!-- end generated -->"));
        }
        m.appendTail(out);
        return out.toString();
    }

    @Test
    void optionTablesMatchTheConfiguration() throws IOException {
        Path file = Path.of("..", "docs", "CONFIGURATION.md");
        String doc = Files.readString(file);
        String expected = generate(doc);
        if (Boolean.getBoolean("jitnn.docs.write")) {
            Files.writeString(file, expected);
            return;
        }
        assertEquals(expected, doc, "docs/CONFIGURATION.md is out of date: run ConfigurationDocTest with -Djitnn.docs.write=true");
    }
}
