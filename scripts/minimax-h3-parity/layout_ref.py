"""Dump the reference PackedLayout, optionally with FL2VA keyframes.

Usage:
    python3 layout_ref.py [--keyframes 0:1,7:1] [--ref DIR] [--out DIR]
"""
import argparse
import os

import numpy as np
import torch

from refiner_ref import install_comfy_shims, load_reference


def parse_keyframes(spec):
    """'index:videoFrames[:audioFrames]' entries separated by commas."""
    keyframes = []
    for entry in filter(None, (spec or "").split(",")):
        parts = entry.split(":")
        keyframe = {"resolved_frame_index": int(parts[0])}
        if len(parts) > 1 and parts[1]:
            # only the latent's temporal extent matters to the layout
            keyframe["latent"] = torch.zeros(1, 24, int(parts[1]), 1, 1)
        if len(parts) > 2 and parts[2]:
            keyframe["audio_latent"] = torch.zeros(1, 32, 2, int(parts[2]))
        keyframes.append(keyframe)
    return keyframes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=os.environ.get(
        "H3_REF", os.path.join(os.environ.get("TMPDIR", "/tmp"), "minimax-h3-ref")))
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures"))
    parser.add_argument("--text", type=int, default=7)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--audio", type=int, default=5)
    parser.add_argument("--keyframes", default="")
    parser.add_argument("--name", default="layout_pos")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    comfy, optimized_attention = install_comfy_shims()
    reference = load_reference(args.ref, comfy, optimized_attention)

    layout = reference.PackedLayout(
        args.text, args.frames, args.height, args.width, args.audio,
        keyframes=parse_keyframes(args.keyframes) or None,
    )
    print(f"segments: {layout.segments}")
    print(f"seq_len: {layout.seq_len}")
    positions = layout.position_ids.numpy()
    print(f"position_ids: {positions.shape}")
    np.save(os.path.join(args.out, f"{args.name}.npy"), positions)
    print(f"wrote {args.name}.npy")


if __name__ == "__main__":
    main()
