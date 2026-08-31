"""Independent dense reference for MiniMax H3's Qwen3-VL text path.

This deliberately uses the Python ``gguf`` package for K-quant decoding and
PyTorch for the transformer math. It compares the Swift implementation's
selected language layers without loading the visual tower or the remaining
checkpoint layers.

Usage:
    python3 qwen_text_ref.py <qwen3vl.gguf> --ids 14777,110048,... \
        --layers 2 --out /tmp/qwen-text-reference.npy
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path = [
    path for path in sys.path
    if os.path.abspath(path or os.curdir) != _SCRIPT_DIRECTORY
]
from gguf import GGUFReader, dequantize


HIDDEN = 5120
HEADS = 64
KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 25600
VOCAB = 151936
ROPE_THETA = 5_000_000.0
RMS_EPS = 1e-6

DEFAULT_IDS = [
    14777, 110048, 114647, 18493, 100167, 29490, 100501,
    111484, 3837, 64, 64665, 6303, 6460, 13,
]


def parse_ids(value):
    try:
        ids = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("ids must be comma-separated integers") from error
    if not ids or any(token < 0 or token >= VOCAB for token in ids):
        raise argparse.ArgumentTypeError("ids must be non-empty and within the Qwen vocabulary")
    return ids


class WeightStore:
    def __init__(self, path):
        self.reader = GGUFReader(path)
        self.by_name = {tensor.name: tensor for tensor in self.reader.tensors}
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            tensor = self.by_name[name]
            decoded = dequantize(tensor.data, tensor.tensor_type)
            self.cache[name] = torch.from_numpy(
                np.asarray(decoded, dtype=np.float32).copy()
            )
        return self.cache[name]


def rms_norm(value, weight):
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + RMS_EPS) * weight


def apply_rope(value):
    sequence = value.shape[-2]
    positions = torch.arange(sequence, dtype=torch.float32)
    index = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)
    inverse_frequency = 1.0 / (ROPE_THETA ** (index / HEAD_DIM))
    frequencies = positions[:, None] * inverse_frequency[None, :]
    cosine = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1)[None, None]
    sine = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1)[None, None]
    half = HEAD_DIM // 2
    first = value[..., :half] * cosine[..., :half] - value[..., half:] * sine[..., :half]
    second = value[..., half:] * cosine[..., half:] + value[..., :half] * sine[..., half:]
    return torch.cat((first, second), dim=-1)


def linear(value, weight):
    return F.linear(value, weight)


def forward(store, token_ids, layers):
    embedding = store.get("model.embed_tokens.weight")
    hidden = embedding[torch.tensor(token_ids, dtype=torch.long)].unsqueeze(0)
    causal = torch.triu(torch.ones((len(token_ids), len(token_ids)), dtype=torch.bool), diagonal=1)

    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        normalized = rms_norm(hidden, store.get(f"{prefix}.input_layernorm.weight"))
        query = linear(normalized, store.get(f"{prefix}.self_attn.q_proj.weight"))
        key = linear(normalized, store.get(f"{prefix}.self_attn.k_proj.weight"))
        value = linear(normalized, store.get(f"{prefix}.self_attn.v_proj.weight"))

        query = query.view(1, len(token_ids), HEADS, HEAD_DIM).transpose(1, 2)
        key = key.view(1, len(token_ids), KV_HEADS, HEAD_DIM).transpose(1, 2)
        value = value.view(1, len(token_ids), KV_HEADS, HEAD_DIM).transpose(1, 2)
        query = rms_norm(query, store.get(f"{prefix}.self_attn.q_norm.weight"))
        key = rms_norm(key, store.get(f"{prefix}.self_attn.k_norm.weight"))
        query = apply_rope(query)
        key = apply_rope(key)

        key = key.repeat_interleave(HEADS // KV_HEADS, dim=1)
        value = value.repeat_interleave(HEADS // KV_HEADS, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / HEAD_DIM**0.5
        scores = scores.masked_fill(causal[None, None], -torch.inf)
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        attended = attended.transpose(1, 2).reshape(1, len(token_ids), HEADS * HEAD_DIM)
        hidden = hidden + linear(attended, store.get(f"{prefix}.self_attn.o_proj.weight"))

        post_norm = rms_norm(
            hidden, store.get(f"{prefix}.post_attention_layernorm.weight")
        )
        gate = linear(post_norm, store.get(f"{prefix}.mlp.gate_proj.weight"))
        up = linear(post_norm, store.get(f"{prefix}.mlp.up_proj.weight"))
        feed_forward = linear(
            F.silu(gate) * up, store.get(f"{prefix}.mlp.down_proj.weight")
        )
        hidden = hidden + feed_forward

    return hidden[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("--ids", type=parse_ids, default=DEFAULT_IDS)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.layers < 1 or args.layers > 50:
        parser.error("layers must be between 1 and 50")

    with torch.no_grad():
        store = WeightStore(args.gguf)
        result = forward(store, args.ids, args.layers).numpy()
    np.save(args.out, result)
    print(f"gguf tensors        : {len(store.reader.tensors)}")
    print(f"token ids ({len(args.ids)})     : {args.ids}")
    print(f"layers              : {args.layers}")
    print(f"hidden              : {result.shape}")
    print(
        "stats               : min %.6f max %.6f mean %.6f std %.6f"
        % (result.min(), result.max(), result.mean(), result.std())
    )
    print(f"wrote               : {args.out}")


if __name__ == "__main__":
    main()
