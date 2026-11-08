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
package ai.beestream.jitnn.nn.text;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Small hand-written tokenizer.json files; HfAccelTest checks the real ones against Hugging Face. */
class TokenizerTest {

    @TempDir
    Path dir;

    private Tokenizer load(String json) throws IOException {
        Path f = dir.resolve("tokenizer.json");
        Files.writeString(f, json);
        return Tokenizer.load(dir);
    }

    @Test
    void wordPiece() throws IOException {
        Tokenizer t = load("""
                {"added_tokens":[{"id":0,"content":"[PAD]","special":true},{"id":1,"content":"[UNK]","special":true},
                  {"id":2,"content":"[CLS]","special":true},{"id":3,"content":"[SEP]","special":true},{"id":4,"content":"[MASK]","special":true}],
                 "normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},
                 "pre_tokenizer":{"type":"BertPreTokenizer"},
                 "post_processor":{"type":"TemplateProcessing",
                   "single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],
                   "pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},
                           {"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],
                   "special_tokens":{"[CLS]":{"id":"[CLS]","ids":[2],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[3],"tokens":["[SEP]"]}}},
                 "model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,
                   "vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"un":5,"##aff":6,"##able":7,"cafe":8,"!":9,"is":10}}}
                """);
        assertArrayEquals(new int[] { 2, 5, 6, 7, 9, 1, 3 }, t.encode("UNaffable! xyz").ids());
        assertArrayEquals(new int[] { 2, 8, 10, 4, 3 }, t.encode("Café is[MASK]").ids());
        Tokenizer.Encoding p = t.encode("cafe", "is");
        assertArrayEquals(new int[] { 2, 8, 3, 10, 3 }, p.ids());
        assertArrayEquals(new int[] { 0, 0, 0, 1, 1 }, p.typeIds());
        assertEquals("unaffable!", t.decode(t.encode("unaffable !").ids(), true));
        Tokenizer.Batch b = t.encodeBatch(List.of("cafe", "unaffable"), 4);
        assertArrayEquals(new int[] { 2, 8, 3, 0, 2, 5, 6, 3 }, b.idArray());
        assertArrayEquals(new int[] { 1, 1, 1, 0, 1, 1, 1, 1 }, b.attention());
    }

    @Test
    void byteLevelBpe() throws IOException {
        // bytes_to_unicode maps the space to U+0120
        Tokenizer t = load("""
                {"added_tokens":[{"id":7,"content":"<|endoftext|>","special":true}],
                 "normalizer":null,"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"use_regex":true},
                 "post_processor":{"type":"ByteLevel"},
                 "model":{"type":"BPE","vocab":{"h":0,"i":1,"\\u0120":2,"hi":3,"\\u0120hi":4,"!":5,"\\u0120h":6},
                   "merges":["\\u0120 h","\\u0120h i","h i"]}}
                """);
        assertArrayEquals(new int[] { 3, 4, 5, 7 }, t.encode("hi hi!<|endoftext|>").ids());
        assertEquals("hi hi!", t.decode(new int[] { 3, 4, 5, 7 }, true));
        assertEquals("hi hi!<|endoftext|>", t.decode(new int[] { 3, 4, 5, 7 }, false));
        assertThrows(IllegalArgumentException.class, () -> t.encode("x"));
    }

    @Test
    void unsupportedModelsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> load("{\"model\":{\"type\":\"Unigram\",\"vocab\":[]}}"));
    }
}
