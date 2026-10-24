"""Downloads gpt2 and bert-base-uncased and writes the references HfAccelTest and
HfTokenizerTest compare with: token ids from the Rust tokenizers library, and FP32 CPU outputs
of the transformers models.

    pip install torch transformers tokenizers huggingface_hub safetensors
    python scripts/hf/hf_reference.py <dir>
    mvn ... -Djitnn.hf=<dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from transformers import BertForMaskedLM, BertModel, GPT2LMHeadModel

out = Path(sys.argv[1])
FILES = ["config.json", "model.safetensors", "tokenizer.json"]
MODELS = {"gpt2": "openai-community/gpt2", "bert-base-uncased": "google-bert/bert-base-uncased"}
for name, repo in MODELS.items():
    snapshot_download(repo, local_dir=out / name, allow_patterns=FILES)

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
    "The capital of France is [MASK].",
    "end of text <|endoftext|> and more",
    "",
    "a" * 120,
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
]
PAIRS = [("How old are you?", "I am six years old."), ("Short.", "A somewhat longer second segment, with punctuation!")]

tokens = {}
for name in MODELS:
    tok = Tokenizer.from_file(str(out / name / "tokenizer.json"))
    single = [{"text": t, "ids": tok.encode(t).ids, "type_ids": tok.encode(t).type_ids} for t in TEXTS]
    pairs = [{"a": a, "b": b, "ids": tok.encode(a, b).ids, "type_ids": tok.encode(a, b).type_ids} for a, b in PAIRS]
    decoded = [{"ids": tok.encode(t).ids, "text": tok.decode(tok.encode(t).ids, skip_special_tokens=True)} for t in TEXTS]
    tokens[name] = {"single": single, "pairs": pairs, "decoded": decoded}
(out / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False, indent=1))

torch.backends.cuda.matmul.allow_tf32 = False
meta = {}

gpt2_tok = Tokenizer.from_file(str(out / "gpt2" / "tokenizer.json"))
ids = gpt2_tok.encode("Hello, my dog is cute and it likes to play with").ids
gpt2 = GPT2LMHeadModel.from_pretrained(out / "gpt2", torch_dtype=torch.float32).eval()
with torch.no_grad():
    logits = gpt2(torch.tensor([ids])).logits[0]
logits.numpy().astype("<f4").tofile(out / "gpt2_logits.f32")
meta["gpt2"] = {"ids": ids, "shape": list(logits.shape)}

bert_tok = Tokenizer.from_file(str(out / "bert-base-uncased" / "tokenizer.json"))
texts = ["The capital of France is [MASK].", "Hello world!"]
encs = [bert_tok.encode(t) for t in texts]
t = max(len(e.ids) for e in encs)
pad = bert_tok.token_to_id("[PAD]")
ids = [e.ids + [pad] * (t - len(e.ids)) for e in encs]
att = [[1] * len(e.ids) + [0] * (t - len(e.ids)) for e in encs]
mlm = BertForMaskedLM.from_pretrained(out / "bert-base-uncased", torch_dtype=torch.float32).eval()
base = BertModel.from_pretrained(out / "bert-base-uncased", torch_dtype=torch.float32).eval()
with torch.no_grad():
    o = base(torch.tensor(ids), attention_mask=torch.tensor(att))
    m = mlm(torch.tensor(ids), attention_mask=torch.tensor(att)).logits
mask_pos = encs[0].ids.index(bert_tok.token_to_id("[MASK]"))
o.last_hidden_state.numpy().astype("<f4").tofile(out / "bert_hidden.f32")
o.pooler_output.numpy().astype("<f4").tofile(out / "bert_pooled.f32")
m[0, mask_pos].numpy().astype("<f4").tofile(out / "bert_mask_logits.f32")
meta["bert"] = {"texts": texts, "ids": ids, "attention": att, "mask_row": mask_pos,
                "hidden_shape": list(o.last_hidden_state.shape), "prediction": bert_tok.id_to_token(int(m[0, mask_pos].argmax()))}
(out / "reference.json").write_text(json.dumps(meta, indent=1))
print("wrote", out, meta["bert"]["prediction"])
