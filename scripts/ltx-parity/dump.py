#!/usr/bin/env python3
"""Create deterministic LTX-2.3 video VAE parity fixtures.

Reference source: dgrauet/ltx-2-mlx v0.14.22
(commit 64b0a1d2ba358232bb73c6730095593e6e9f3ffb).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.video_vae.tiling import (
    TemporalTilingConfig,
    TilingConfig,
    prepare_tiles_for_decoding,
)
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
from ltx_core_mlx.utils.weights import load_split_safetensors


def load_decoder(model_directory: Path, input_dtype: str) -> tuple[VideoDecoder, int]:
    decoder = VideoDecoder()
    weights = load_split_safetensors(
        model_directory / "vae_decoder.safetensors",
        prefix="vae_decoder.",
    )
    if input_dtype == "float32":
        weights = {name: value.astype(mx.float32) for name, value in weights.items()}
    decoder.load_weights(list(weights.items()), strict=True)
    decoder.eval()
    mx.eval(decoder.parameters())
    return decoder, len(weights)


def tiling_configuration(arguments: argparse.Namespace) -> TilingConfig | None:
    if arguments.tiling_mode == "none":
        return None
    return TilingConfig(
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=arguments.tile_size_in_frames,
            tile_overlap_in_frames=arguments.tile_overlap_in_frames,
        )
    )


def dump_vae_decode(arguments: argparse.Namespace) -> None:
    model_directory: Path = arguments.model_dir
    weights_path = model_directory / "vae_decoder.safetensors"
    config_path = model_directory / "embedded_config.json"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    latent_frames = arguments.latent_frames
    if latent_frames is None:
        latent_frames = 4 if arguments.tiling_mode == "multi" else 2
    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    latent = mx.random.normal(
        (
            1,
            128,
            latent_frames,
            arguments.latent_height,
            arguments.latent_width,
        ),
        dtype=input_dtype,
        key=mx.random.key(arguments.seed),
    )
    tiling = tiling_configuration(arguments)
    tiles = prepare_tiles_for_decoding(latent.shape, tiling)
    if arguments.tiling_mode == "single" and len(tiles) != 1:
        raise ValueError(f"single case must create exactly 1 tile, got {len(tiles)}")
    if arguments.tiling_mode == "multi" and len(tiles) <= 1:
        raise ValueError(f"multi case must create more than 1 tile, got {len(tiles)}")

    decoder, tensor_count = load_decoder(model_directory, arguments.input_dtype)
    chunks = list(decoder.tiled_decode(latent, tiling))
    if not chunks:
        raise RuntimeError("VAE decoder returned no chunks")
    frames = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=2)
    mx.eval(latent, frames)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {"latent": latent, "frames": frames},
        metadata={
            "stage": "video_vae_decode",
            "reference": "dgrauet/ltx-2-mlx@v0.14.22",
            "reference_commit": "64b0a1d2ba358232bb73c6730095593e6e9f3ffb",
            "model": str(model_directory),
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
            "latent_layout": "BCFHW",
            "frame_layout": "BCFHW",
            "tiling_mode": arguments.tiling_mode,
            "tile_size_in_frames": str(arguments.tile_size_in_frames),
            "tile_overlap_in_frames": str(arguments.tile_overlap_in_frames),
            "tile_count": str(len(tiles)),
            "weight_tensors": str(tensor_count),
            "quantized_modules": "0",
        },
    )
    print(f"reference={arguments.output}")
    print(f"weights={tensor_count} tensors")
    print("quantized modules=0")
    print(f"input dtype={arguments.input_dtype}")
    print(f"tiling mode={arguments.tiling_mode} tile count={len(tiles)}")
    print(f"latent shape={latent.shape} dtype={latent.dtype}")
    print(f"frames shape={frames.shape} dtype={frames.dtype}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    decode = subparsers.add_parser("vae-decode")
    decode.add_argument("--model-dir", type=Path, required=True)
    decode.add_argument("--output", type=Path, required=True)
    decode.add_argument("--seed", type=int, default=7)
    decode.add_argument(
        "--input-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    decode.add_argument(
        "--tiling-mode",
        choices=("none", "single", "multi"),
        default="single",
    )
    decode.add_argument("--latent-frames", type=int)
    decode.add_argument("--latent-height", type=int, default=1)
    decode.add_argument("--latent-width", type=int, default=1)
    decode.add_argument("--tile-size-in-frames", type=int, default=16)
    decode.add_argument("--tile-overlap-in-frames", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "vae-decode":
        dump_vae_decode(arguments)
        return
    raise ValueError(f"unsupported command: {arguments.command}")


if __name__ == "__main__":
    main()
