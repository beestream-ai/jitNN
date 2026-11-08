# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Downloads two small Llama-family checkpoints and writes the references HfAccelTest's
llamaLogits compares with: FP32 CPU logits of the transformers models for a fixed prompt, and the
token ids of the Rust tokenizers library for the texts of hf_reference.py (llamaTokenizers).

    SmolLM2-135M   model_type llama, grouped-query attention (9 query, 3 key/value heads), tied output
    Qwen2.5-0.5B   model_type qwen2, grouped-query attention (14 / 2), q/k/v biases, tied output
    tiny-llama3    a random 2-layer Llama with Llama 3.1's rope_scaling (made here, no tokenizer),
                   on 200 tokens so the rescaled frequencies matter

    pip install torch transformers tokenizers huggingface_hub safetensors
    python scripts/hf/hf_llama_reference.py <dir>
    mvn ... -Djitnn.hf=<dir>
"""
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

out = Path(sys.argv[1])
MODELS = {"smollm2-135m": "HuggingFaceTB/SmolLM2-135M", "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B"}
PROMPT = "The capital of France is Paris. The capital of Germany is"
TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I can't believe it's not butter; we've tried, they'll see, I'm sure you'd agree.",
    "  leading and   multiple   spaces\tand tabs\nand newlines  ",
    "Numbers: 3.14159, 1,000,000 and 2026-10-24.",
    "Café naïve résumé Ångström Straße",
    "日本語のテキストと中文字符",
    "Emoji 😀👍🏽 and symbols © ® ™ € £ ¥",
    "URLs like https://example.com/path?q=1&r=2 and e-mail a.b@c.org",
    "end of text <|endoftext|> and <|im_start|>user\nhi<|im_end|>",
    "",
    "a" * 120,
    "Line one\r\nline two\n\n\nthree   \n",
    "Ｆｕｌｌｗｉｄｔｈ ﬁ ligature and e\u0301 combining",
]

torch.backends.cuda.matmul.allow_tf32 = False
meta = {}
for name, repo in MODELS.items():
    snapshot_download(repo, local_dir=out / name, allow_patterns=["config.json", "*.safetensors", "model.safetensors.index.json", "tokenizer.json"])
    tok = Tokenizer.from_file(str(out / name / "tokenizer.json"))
    ids = tok.encode(PROMPT).ids
    model = AutoModelForCausalLM.from_pretrained(out / name, torch_dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
    logits.numpy().astype("<f4").tofile(out / f"{name}_logits.f32")
    meta[name] = {"prompt": PROMPT, "ids": ids, "shape": list(logits.shape), "next": tok.id_to_token(int(logits[-1].argmax())),
                  "texts": [{"text": t, "ids": tok.encode(t).ids, "decoded": tok.decode(tok.encode(t).ids, skip_special_tokens=True)} for t in TEXTS]}
torch.manual_seed(0)
config = LlamaConfig(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
                     rope_theta=500000.0, rms_norm_eps=1e-5, tie_word_embeddings=False, max_position_embeddings=1024,
                     rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                   "original_max_position_embeddings": 64})
tiny = LlamaForCausalLM(config).eval()
with torch.no_grad():
    for p in tiny.parameters():
        p.normal_(0, 0.2)
tiny.save_pretrained(out / "tiny-llama3", safe_serialization=True)
ids = torch.randint(0, 1000, (200,), generator=torch.Generator().manual_seed(1)).tolist()
with torch.no_grad():
    logits = tiny(torch.tensor([ids])).logits[0]
logits.numpy().astype("<f4").tofile(out / "tiny-llama3_logits.f32")
unscaled = LlamaForCausalLM(LlamaConfig(**{**config.to_dict(), "rope_scaling": None, "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}})).eval()
unscaled.load_state_dict(tiny.state_dict())
with torch.no_grad():
    print("tiny-llama3: logits without the rope scaling differ by", float((unscaled(torch.tensor([ids])).logits[0] - logits).abs().max()), "of", float(logits.abs().max()))
meta["tiny-llama3"] = {"ids": ids, "shape": list(logits.shape), "next": int(logits[-1].argmax())}
(out / "llama_reference.json").write_text(json.dumps(meta, indent=1))
print("wrote", out, {k: v["next"] for k, v in meta.items()})
