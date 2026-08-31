"""Reference run of the H3 video VAE decode path.

Covers `_decode_pixels` — `decoder(post_quant_conv(z))` — not just the ViT
decoder, so the comparison includes the channel mix that sits between the
latent and the transformer.

Usage:
    python3 vae_ref.py <video_vae.safetensors> [--ref DIR] [--out DIR]
"""
import argparse
import json
import os
import struct
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from refiner_ref import write_safetensors


def install_vae_shims():
    """`vae.py` needs a smaller shim set than `model.py`."""
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    for name in ["ops", "rmsnorm", "quant_ops", "model_management"]:
        module = types.ModuleType(f"comfy.{name}")
        setattr(comfy, name, module)
        sys.modules[f"comfy.{name}"] = module

    def rms_norm(x, weight=None, eps=1e-5):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return out if weight is None else out * weight

    def apply_rope_split_half(q, k, table):
        # table is [B, S, 1, pairs, 2, 2] built from [cos, -sin, sin, cos].
        cos = table[..., 0, 0]
        sin = table[..., 1, 0]

        def rotate(x):
            pairs = cos.shape[-1]
            first, second = x[..., :pairs], x[..., pairs:2 * pairs]
            return torch.cat([first * cos - second * sin,
                              first * sin + second * cos], dim=-1)

        return rotate(q), rotate(k)

    class AutopadConv3d(nn.Conv3d):
        """`nn.Conv3d` plus the `autopad` kwarg ComfyUI's ops layer adds.

        `CausalConv3d` passes `autopad="causal_zero"` for a single frame,
        where the causal front padding would be all zeros. Its in-source
        comment says to "truncate the temporal taps instead of convolving zero
        frames", i.e. keep only the taps that see real data — the last
        `min(T, KT)` of them.
        """

        def forward(self, x, autopad=None):
            if autopad is None:
                return super().forward(x)
            assert autopad == "causal_zero", autopad
            taps = min(x.shape[2], self.weight.shape[2])
            return F.conv3d(
                x, self.weight[:, :, -taps:], self.bias,
                stride=self.stride, padding=0,
                dilation=self.dilation, groups=self.groups,
            )

    ops_module = types.SimpleNamespace(
        Conv3d=AutopadConv3d, GroupNorm=nn.GroupNorm, Linear=nn.Linear,
        LayerNorm=nn.LayerNorm, RMSNorm=nn.RMSNorm, Conv1d=nn.Conv1d,
        ConvTranspose1d=nn.ConvTranspose1d,
    )

    comfy.rmsnorm.rms_norm = rms_norm
    comfy.ops.cast_to_input = lambda w, x: w.to(dtype=x.dtype, device=x.device)
    comfy.ops.disable_weight_init = ops_module
    comfy.quant_ops.ck = types.SimpleNamespace(
        apply_rope_split_half=apply_rope_split_half
    )
    comfy.model_management.cast_to = lambda w, **kwargs: w

    ldm = types.ModuleType("comfy.ldm"); ldm.__path__ = []
    modules = types.ModuleType("comfy.ldm.modules"); modules.__path__ = []
    attention = types.ModuleType("comfy.ldm.modules.attention")
    comfy.ldm = ldm; ldm.modules = modules; modules.attention = attention

    def optimized_attention(q, k, v, heads, skip_reshape=False):
        out = F.scaled_dot_product_attention(q, k, v)
        batch, head_count, seq, dim = out.shape
        return out.transpose(1, 2).reshape(batch, seq, head_count * dim)

    attention.optimized_attention = optimized_attention
    sys.modules.update({
        "comfy": comfy, "comfy.ldm": ldm, "comfy.ldm.modules": modules,
        "comfy.ldm.modules.attention": attention,
    })
    return comfy, optimized_attention, ops_module


def read_safetensors_f32(path, prefix=None):
    """Read a safetensors file, tolerating trailing bytes.

    The downloaded H3 VAE files carry a 61-byte trailer appended by the
    download tool, which the strict safetensors reader rejects.
    """
    dtypes = {"F16": np.float16, "F32": np.float32, "F64": np.float64}
    with open(path, "rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
        base = 8 + header_length
        out = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if prefix and not name.startswith(prefix):
                continue
            start, stop = entry["data_offsets"]
            handle.seek(base + start)
            raw = handle.read(stop - start)
            array = np.frombuffer(raw, dtype=dtypes[entry["dtype"]])
            out[name] = torch.from_numpy(
                array.reshape(entry["shape"]).astype(np.float32).copy()
            )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vae")
    parser.add_argument("--ref", default=os.environ.get(
        "H3_REF", os.path.join(os.environ.get("TMPDIR", "/tmp"), "minimax-h3-ref")))
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures"))
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--encode", action="store_true",
                        help="run the encoder instead of the decoder")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    comfy, optimized_attention, ops_module = install_vae_shims()
    source = open(os.path.join(args.ref, "vae.py")).read()
    module = types.ModuleType("h3vae")
    module.__dict__.update({
        "torch": torch, "nn": nn, "F": F, "math": __import__("math"),
        "comfy": comfy, "ops": ops_module,
        "optimized_attention": optimized_attention,
    })
    exec(compile(source, "vae.py", "exec"), module.__dict__)

    state = read_safetensors_f32(args.vae)

    if args.encode:
        # _encode_moments: quant_conv(encoder(x)) on ImageNet-normalized pixels.
        encoder = module.EncoderFCN3D(
            ch=128, ch_mult=[1, 2, 2, 4, 4, 8], space_down=[2, 2, 2, 2, 1, 1],
            time_down=[1, 2, 2, 1, 1, 1], num_res_blocks=2, in_channels=3,
            z_channels=24, double_z=True,
        )
        missing, unexpected = encoder.load_state_dict(
            {k[len("encoder."):]: v for k, v in state.items()
             if k.startswith("encoder.")}, strict=False)
        assert not unexpected, unexpected
        encoder = encoder.float().eval()
        quant = nn.Conv3d(48, 48, 1)
        quant.load_state_dict({"weight": state["quant_conv.weight"],
                               "bias": state["quant_conv.bias"]})
        quant = quant.float().eval()

        generator = torch.Generator().manual_seed(args.seed)
        size = args.size * 16
        pixels = torch.rand(1, 3, 1, size, size, generator=generator,
                            dtype=torch.float32) * 2.0 - 1.0
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        latents_mean = torch.tensor(module.LATENTS_MEAN).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(module.LATENTS_STD).view(1, -1, 1, 1, 1)
        with torch.no_grad():
            normalized = (pixels + 1.0) * 0.5
            normalized = (normalized - pixel_mean) / pixel_std
            moments = quant(encoder(normalized))
            mean = torch.chunk(moments.float(), 2, dim=1)[0]
            latent = (mean - latents_mean) / latents_std
        print(f"pixels {tuple(pixels.shape)} -> latent {tuple(latent.shape)}")
        print("stats: min %.6f max %.6f mean %.6f std %.6f"
              % (latent.min(), latent.max(), latent.mean(), latent.std()))
        np.save(os.path.join(args.out, "encode_out.npy"), latent.numpy())
        write_safetensors(os.path.join(args.out, "encode_in.safetensors"),
                          {"pixels": pixels.numpy()})
        print(f"wrote fixtures to {args.out}")
        return

    decoder = module.ViT3DDecoder(patch_size=16, patch_size_t=4, in_channels=24,
                                  out_channels=3, operations=ops_module)
    decoder.load_state_dict(
        {k[len("decoder."):]: v for k, v in state.items()
         if k.startswith("decoder.")}, strict=False
    )
    decoder = decoder.float().eval()
    post_quant = nn.Conv3d(24, 24, 1)
    post_quant.load_state_dict({"weight": state["post_quant_conv.weight"],
                                "bias": state["post_quant_conv.bias"]})
    post_quant = post_quant.float().eval()

    generator = torch.Generator().manual_seed(args.seed)
    latent = torch.randn(1, 24, args.frames, args.size, args.size,
                         generator=generator, dtype=torch.float32)
    with torch.no_grad():
        # _decode_pixels: decoder(post_quant_conv(z))
        pixels = decoder(post_quant(latent))

    print(f"latent {tuple(latent.shape)} -> pixels {tuple(pixels.shape)}")
    print("stats: min %.6f max %.6f mean %.6f std %.6f"
          % (pixels.min(), pixels.max(), pixels.mean(), pixels.std()))
    np.save(os.path.join(args.out, "vae_out.npy"), pixels.numpy())
    write_safetensors(os.path.join(args.out, "vae_in.safetensors"),
                      {"latent": latent.numpy()})
    print(f"wrote fixtures to {args.out}")


if __name__ == "__main__":
    main()
