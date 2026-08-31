"""Reference run of H3's packed sequence through the first N DiT blocks.

Exercises the parts the token refiner does not: the time embedder, AdaLN
modulation across three modalities, 3-axis split-half RoPE, and the gated
residuals. Runs with fully decoded weights, so any difference is model logic
rather than quantization.

Usage:
    python3 dit_ref.py <transformer.gguf> [--layers 2] [--ref DIR] [--out DIR]
"""
import argparse
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn

from gguf import GGUFFile
from refiner_ref import (HEADS, HEAD_DIM, HIDDEN, FFN, install_comfy_shims,
                         load_reference, write_safetensors)

TIME_EMBED_DIM = 2688
TIMESTEP_INPUT_DIM = 256
PATCH = (1, 2, 2)
VIDEO_LATENTS = 24
AUDIO_LATENTS = 32
SIGMA_SHIFT_VIDEO = 12.0
SIGMA_SHIFT_AUDIO = 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ref", default=os.environ.get(
        "H3_REF", os.path.join(os.environ.get("TMPDIR", "/tmp"), "minimax-h3-ref")))
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures"))
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--keyframes", default="",
                        help="index:videoLatentFrames entries, comma separated")
    parser.add_argument("--final", action="store_true",
                        help="also run final_layer and unpatchify")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    comfy, optimized_attention = install_comfy_shims()
    reference = load_reference(args.ref, comfy, optimized_attention)
    gguf = GGUFFile(args.gguf)

    text_len, latent_t, latent_h, latent_w, audio_t = 7, 3, 8, 12, 5
    # Conditioning latents are generated first so both sides share them.
    cond_generator = torch.Generator().manual_seed(4242)
    keyframes = []
    for entry in filter(None, args.keyframes.split(",")):
        parts = entry.split(":")
        index, vt = int(parts[0]), int(parts[1])
        keyframes.append({
            "resolved_frame_index": index,
            "latent": torch.randn(1, VIDEO_LATENTS, vt, latent_h, latent_w,
                                  generator=cond_generator, dtype=torch.float32),
        })
    layout = reference.PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t,
                                    keyframes=keyframes or None)
    print(f"segments {layout.segments}  seq_len {layout.seq_len}")

    def tensor(name):
        return torch.from_numpy(gguf.decode(name))

    time_embedder = reference.TimeEmbedder(
        TIMESTEP_INPUT_DIM, HIDDEN, TIME_EMBED_DIM, operations=nn
    )
    time_embedder.load_state_dict({
        "proj_in.weight": tensor("time_embedder.proj_in.weight"),
        "proj_in.bias": tensor("time_embedder.proj_in.bias"),
        "proj_out.weight": tensor("time_embedder.proj_out.weight"),
        "proj_out.bias": tensor("time_embedder.proj_out.bias"),
    })
    video_patch = nn.Linear(VIDEO_LATENTS * PATCH[0] * PATCH[1] * PATCH[2], HIDDEN)
    video_patch.load_state_dict({"weight": tensor("video_patch_proj.weight"),
                                 "bias": tensor("video_patch_proj.bias")})
    audio_patch = nn.Linear(AUDIO_LATENTS, HIDDEN)
    audio_patch.load_state_dict({"weight": tensor("audio_patch_proj.weight"),
                                 "bias": tensor("audio_patch_proj.bias")})
    inv_freq = tensor("rope.inv_freq")

    blocks = []
    for index in range(args.layers):
        block = reference.DiTBlock(HIDDEN, HEADS, HEAD_DIM, FFN, TIME_EMBED_DIM,
                                   1e-5, 1e-5, operations=nn)
        state = {
            name[len(f"blocks.{index}."):]: tensor(name)
            for name in gguf.tensors if name.startswith(f"blocks.{index}.")
        }
        missing, unexpected = block.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        blocks.append(block.float().eval())
    print(f"loaded {len(blocks)} DiT block(s)")

    generator = torch.Generator().manual_seed(args.seed)
    video = torch.randn(1, VIDEO_LATENTS, latent_t, latent_h, latent_w,
                        generator=generator, dtype=torch.float32)
    audio = torch.randn(1, AUDIO_LATENTS, 2, audio_t,
                        generator=generator, dtype=torch.float32)
    text = torch.randn(text_len, HIDDEN, generator=generator, dtype=torch.float32)

    sigma = torch.tensor(args.sigma, dtype=torch.float32)
    t_video = float(1.0 - sigma)
    t_audio = float(1.0 - reference.time_shift_sigma(
        sigma, SIGMA_SHIFT_VIDEO, SIGMA_SHIFT_AUDIO))
    unique = sorted({t_video, t_audio})
    row_of = {value: index for index, value in enumerate(unique)}
    # aug == 1.0 disables the conditioning noise blend, so no RNG is involved
    # and both sides can be compared exactly.
    VIS_AUG = 1.0
    t_cond = max(t_video, VIS_AUG)
    unique = sorted(set(unique) | ({t_cond} if keyframes else set()))
    row_of = {value: index for index, value in enumerate(unique)}
    tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "cond_audio": 2}
    seg_time = {"text": t_video, "video": t_video, "audio": t_audio,
                "cond": t_cond, "cond_audio": max(t_audio, 1.0)}
    mod_segments = [(a, b, row_of[seg_time[kind]] * 3 + tag[kind])
                    for a, b, kind in layout.segments]

    with torch.no_grad():
        video_rows = reference.patchify_video(video, PATCH)
        audio_rows = reference.pack_audio(audio)
        cond_rows = [reference.patchify_video(kf["latent"], PATCH) for kf in keyframes]
        all_video_rows = torch.cat(cond_rows + [video_rows], 0) if cond_rows else video_rows
        video_embed = video_patch(all_video_rows)
        audio_embed = audio_patch(audio_rows)
        pieces, voff, aoff = [], 0, 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                pieces.append(text)
            elif kind in ("cond", "video"):
                pieces.append(video_embed[voff:voff + n]); voff += n
            else:
                pieces.append(audio_embed[aoff:aoff + n]); aoff += n
        hidden = torch.cat(pieces, 0)
        t_emb = time_embedder(torch.tensor(unique, dtype=torch.float32))

        position = layout.position_ids.to(torch.float32)
        per_axis = position.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        angles = torch.cat((half, half), dim=-1)
        rope = reference.rope_rotation_table(angles, torch.float32)

        for block in blocks:
            hidden = block(hidden, t_emb, mod_segments, rope)

    print(f"hidden {tuple(hidden.shape)}")
    print("stats: min %.6f max %.6f mean %.6f std %.6f"
          % (hidden.min(), hidden.max(), hidden.mean(), hidden.std()))

    if args.final:
        final = reference.FinalLayer(HIDDEN, TIME_EMBED_DIM,
                                     VIDEO_LATENTS * PATCH[0] * PATCH[1] * PATCH[2],
                                     AUDIO_LATENTS, 1e-5, operations=nn)
        state = {name[len("final_layer."):]: tensor(name)
                 for name in gguf.tensors if name.startswith("final_layer.")}
        missing, unexpected = final.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        final = final.float().eval()

        va, vb, _ = next(s for s in layout.segments if s[2] == "video")
        aa, ab, _ = next(s for s in layout.segments if s[2] == "audio")
        with torch.no_grad():
            v, a = final(hidden, t_emb,
                         (va, vb, row_of[t_video]), (aa, ab, row_of[t_audio]),
                         sigma, None, (SIGMA_SHIFT_VIDEO, SIGMA_SHIFT_AUDIO))
            video_out = reference.unpatchify_video(
                v, latent_t, latent_h // 2, latent_w // 2, VIDEO_LATENTS, PATCH)
            audio_out = reference.unpack_audio(a)
            video_out = -video_out
            audio_out = -audio_out
        print(f"video_out {tuple(video_out.shape)}  audio_out {tuple(audio_out.shape)}")
        print("video stats: min %.6f max %.6f mean %.6f"
              % (video_out.min(), video_out.max(), video_out.mean()))
        np.save(os.path.join(args.out, "forward_video.npy"), video_out.numpy())
        np.save(os.path.join(args.out, "forward_audio.npy"), audio_out.numpy())

    np.save(os.path.join(args.out, "dit_out.npy"), hidden.numpy())
    fixture = {"video": video.numpy(), "audio": audio.numpy(), "text": text.numpy()}
    for i, kf in enumerate(keyframes):
        fixture[f"cond_video_{i}"] = kf["latent"].numpy()
    write_safetensors(os.path.join(args.out, "dit_in.safetensors"), fixture)
    print(f"wrote fixtures to {args.out}")


if __name__ == "__main__":
    main()
