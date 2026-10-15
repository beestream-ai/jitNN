"""Exports two small models with symbolic dimensions for OnnxBucketAccelTest: an encoder taking
input_ids and attention_mask, both [batch, sequence], and a CNN taking images [batch, 3, 32, 32].

    python scripts/onnx/bucket_models.py <dir>
"""
import os
import sys

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab=1000, dim=128, heads=4, layers=2):
        super().__init__()
        self.tokens = nn.Embedding(vocab, dim)
        self.positions = nn.Embedding(512, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, input_ids, attention_mask):
        pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = self.tokens(input_ids) + self.positions(pos)
        # an additive mask: padded keys (mask 0) get a large negative score
        bias = (1.0 - attention_mask.to(x.dtype))[:, None, None, :] * -10000.0
        bias = bias.expand(-1, self.encoder.layers[0].self_attn.num_heads, input_ids.shape[1], -1).reshape(-1, input_ids.shape[1], input_ids.shape[1])
        return self.encoder(x, mask=bias)


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)


def main(out):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    enc = Encoder().eval()
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.int64)
    batch = torch.export.Dim("batch", min=1, max=64)
    seq = torch.export.Dim("sequence", min=2, max=512)
    torch.onnx.export(enc, (ids, mask), os.path.join(out, "encoder.onnx"), input_names=["input_ids", "attention_mask"], output_names=["hidden"],
                      dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}}, opset_version=18, dynamo=True)
    cnn = Cnn().eval()
    torch.onnx.export(cnn, (torch.randn(2, 3, 32, 32),), os.path.join(out, "cnn.onnx"), input_names=["images"], output_names=["logits"],
                      dynamic_shapes={"x": {0: batch}}, opset_version=18, dynamo=True)
    print("exported to", out)


if __name__ == "__main__":
    main(sys.argv[1])
