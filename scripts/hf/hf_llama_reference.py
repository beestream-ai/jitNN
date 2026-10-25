"""Downloads two small Llama-family checkpoints and writes the references HfAccelTest's
llamaLogits compares with: FP32 CPU logits of the transformers models for a fixed prompt.

    SmolLM2-135M   model_type llama, grouped-query attention (9 query, 3 key/value heads), tied output
    Qwen2.5-0.5B   model_type qwen2, grouped-query attention (14 / 2), q/k/v biases, tied output

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
from transformers import AutoModelForCausalLM

out = Path(sys.argv[1])
MODELS = {"smollm2-135m": "HuggingFaceTB/SmolLM2-135M", "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B"}
PROMPT = "The capital of France is Paris. The capital of Germany is"

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
    meta[name] = {"ids": ids, "shape": list(logits.shape), "next": tok.id_to_token(int(logits[-1].argmax()))}
(out / "llama_reference.json").write_text(json.dumps(meta, indent=1))
print("wrote", out, {k: v["next"] for k, v in meta.items()})
