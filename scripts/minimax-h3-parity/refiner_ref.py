"""Reference run of H3's condition_proj + token refiner.

The refiner is only two blocks, so it can run with fully decoded weights. That
keeps the comparison focused on model logic — attention, QK-RMSNorm, SwiGLU and
the residual structure — with no quantization error in the way.

Usage:
    python3 refiner_ref.py <transformer.gguf> [--ref DIR] [--out DIR]
"""
import argparse
import json
import math
import os
import struct
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gguf import GGUFFile

HIDDEN, HEADS, HEAD_DIM, FFN, TEXT_DIM = 5376, 56, 128, 14336, 5120


def install_comfy_shims():
    """Stub the comfy modules `model.py` imports, keeping the maths real."""
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    for name in ["ops", "rmsnorm", "quant_ops", "model_management",
                 "patcher_extension", "model_prefetch"]:
        module = types.ModuleType(f"comfy.{name}")
        setattr(comfy, name, module)
        sys.modules[f"comfy.{name}"] = module

    comfy.ops.cast_to_input = lambda w, x: w.to(dtype=x.dtype, device=x.device)
    comfy.ops.disable_weight_init = nn

    def linear_input_act(fc2, x, kind):
        assert kind == "swiglu", kind
        gate, value = x.chunk(2, dim=-1)
        return fc2(F.silu(gate) * value)

    comfy.ops.linear_input_act = linear_input_act
    comfy.model_management.cast_to = lambda w, **kwargs: w
    comfy.model_management.in_training = False

    def _rms_rope_split_half(q, k, table, q_weight, k_weight, epsilon, rot_dim):
        """Fused per-head RMSNorm plus partial split-half rope.

        `rope_rotation_table` stacks [cos, -sin, sin, cos] into 2x2 blocks, so
        cos sits at [..., 0, 0] and sin at [..., 1, 0]. The rotation pairs
        channel i with channel i + rot_dim/2 (the "split half" convention), and
        leaves channels past rot_dim untouched.
        """
        cos = table[..., 0, 0]
        sin = table[..., 1, 0]
        half = rot_dim // 2
        results = []
        for value, weight in ((q, q_weight), (k, k_weight)):
            x = value.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
            x = x * weight.float()
            first, second = x[..., :half], x[..., half:rot_dim]
            rotated = torch.cat([
                first * cos - second * sin,
                first * sin + second * cos,
                x[..., rot_dim:],
            ], dim=-1)
            results.append(rotated.to(value.dtype))
        return results

    def rms_rope_split_half(q, k, table, qw, kw, epsilon=1e-5, rot_dim=0):
        return tuple(_rms_rope_split_half(q, k, table, qw, kw, epsilon, rot_dim))

    def rms_rope_split_half_(q, k, table, qw, kw, epsilon=1e-5, rot_dim=0):
        new_q, new_k = _rms_rope_split_half(q, k, table, qw, kw, epsilon, rot_dim)
        q.copy_(new_q)
        k.copy_(new_k)

    comfy.quant_ops.ck = types.SimpleNamespace(
        rms_rope_split_half=rms_rope_split_half,
        rms_rope_split_half_=rms_rope_split_half_,
    )

    ldm = types.ModuleType("comfy.ldm"); ldm.__path__ = []
    modules = types.ModuleType("comfy.ldm.modules"); modules.__path__ = []
    attention = types.ModuleType("comfy.ldm.modules.attention")
    common_dit = types.ModuleType("comfy.ldm.common_dit")
    comfy.ldm = ldm
    ldm.modules = modules
    modules.attention = attention
    ldm.common_dit = common_dit

    def optimized_attention(q, k, v, heads, mask=None, skip_reshape=False,
                            transformer_options=None):
        out = F.scaled_dot_product_attention(q, k, v)
        batch, head_count, seq, dim = out.shape
        return out.transpose(1, 2).reshape(batch, seq, head_count * dim)

    attention.optimized_attention = optimized_attention
    attention.AttentionTensorContainer = lambda x: x
    common_dit.pad_to_patch_size = lambda x, patch: x

    sys.modules.update({
        "comfy": comfy, "comfy.ldm": ldm, "comfy.ldm.modules": modules,
        "comfy.ldm.modules.attention": attention,
        "comfy.ldm.common_dit": common_dit,
    })
    return comfy, optimized_attention


def load_reference(ref_dir, comfy, optimized_attention):
    source = open(os.path.join(ref_dir, "model.py")).read()
    module = types.ModuleType("h3model")
    module.__dict__.update({
        "torch": torch, "nn": nn, "F": F, "math": math, "comfy": comfy,
        "ops": nn, "optimized_attention": optimized_attention,
    })
    exec(compile(source, "model.py", "exec"), module.__dict__)
    return module


def write_safetensors(path, arrays):
    header, offset, blobs = {}, 0, []
    for name, array in arrays.items():
        array = np.ascontiguousarray(array, dtype=np.float32)
        header[name] = {"dtype": "F32", "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
        blobs.append(array.tobytes())
    encoded = json.dumps(header).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("--ref", default=os.environ.get(
        "H3_REF", os.path.join(os.environ.get("TMPDIR", "/tmp"), "minimax-h3-ref")))
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures"))
    parser.add_argument("--tokens", type=int, default=7)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    comfy, optimized_attention = install_comfy_shims()
    reference = load_reference(args.ref, comfy, optimized_attention)

    gguf = GGUFFile(args.gguf)
    print(f"gguf: {len(gguf.tensors)} tensors, arch {gguf.architecture}")

    refiner = reference.TokenRefiner(
        2, HIDDEN, HEADS, HEAD_DIM, FFN, 1e-5, 1e-5, 1e-5, operations=nn
    )
    condition = nn.Linear(TEXT_DIM, HIDDEN, bias=True)
    condition.load_state_dict({
        "weight": torch.from_numpy(gguf.decode("condition_proj.weight")),
        "bias": torch.from_numpy(gguf.decode("condition_proj.bias")),
    })
    state = {
        name[len("token_refiner."):]: torch.from_numpy(gguf.decode(name))
        for name in gguf.tensors if name.startswith("token_refiner.")
    }
    missing, unexpected = refiner.load_state_dict(state, strict=False)
    print(f"refiner missing {list(missing)}  unexpected {list(unexpected)}")
    refiner = refiner.float().eval()
    condition = condition.float().eval()

    generator = torch.Generator().manual_seed(args.seed)
    text = torch.randn(args.tokens, TEXT_DIM, generator=generator, dtype=torch.float32)
    with torch.no_grad():
        out = refiner(condition(text))

    print(f"out {tuple(out.shape)}")
    print("stats: min %.6f max %.6f mean %.6f std %.6f"
          % (out.min(), out.max(), out.mean(), out.std()))

    np.save(os.path.join(args.out, "refiner_out.npy"), out.numpy())
    write_safetensors(os.path.join(args.out, "refiner_in.safetensors"),
                      {"text": text.numpy()})
    print(f"wrote fixtures to {args.out}")


if __name__ == "__main__":
    main()
