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
    DimensionTilingConfig,
    TemporalTilingConfig,
    TileCountConfig,
    TilingConfig,
    prepare_tiles_for_decoding,
)
from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder, VideoEncoder
from ltx_core_mlx.model.video_vae.ops import remap_encoder_weight_keys
from ltx_core_mlx.model.transformer.model import LTXModel, LTXModelConfig
from ltx_core_mlx.model.transformer.modality import Modality
from ltx_core_mlx.model.transformer.rope import precompute_rope_freqs
from ltx_core_mlx.components.modality_tiling import VideoModalityTiler
from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel
from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2
from ltx_core_mlx.utils.weights import (
    apply_quantization,
    load_split_safetensors,
    remap_audio_vae_keys,
)


DEFAULT_VIDEO_FRAMES = 8
DEFAULT_VIDEO_HEIGHT = 16
DEFAULT_VIDEO_WIDTH = 16
DEFAULT_AUDIO_TOKENS = 256
DEFAULT_TEXT_TOKENS = 64
EFFECTIVE_TRANSFORMER_DTYPE = "bfloat16"


def resolve_model_relative_path(model_directory: Path, relative_name: str) -> Path:
    candidate = Path(relative_name)
    if candidate.is_absolute():
        raise ValueError("--weights 必須是相對於 --model-dir 的檔名，不可使用完整路徑")
    root = model_directory.resolve()
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("--weights 不可離開 --model-dir 目錄")
    return resolved


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


def load_encoder(model_directory: Path, input_dtype: str) -> tuple[VideoEncoder, int]:
    encoder = VideoEncoder()
    weights = load_split_safetensors(
        model_directory / "vae_encoder.safetensors",
        prefix="vae_encoder.",
    )
    weights = remap_encoder_weight_keys(weights)
    if input_dtype == "float32":
        weights = {name: value.astype(mx.float32) for name, value in weights.items()}
    encoder.load_weights(list(weights.items()), strict=True)
    encoder.eval()
    mx.eval(encoder.parameters())
    return encoder, len(weights)


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


def dump_vae_encode(arguments: argparse.Namespace) -> None:
    model_directory: Path = arguments.model_dir
    weights_path = model_directory / "vae_encoder.safetensors"
    config_path = model_directory / "embedded_config.json"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    pixels = mx.random.normal(
        (
            1,
            3,
            arguments.pixel_frames,
            arguments.pixel_height,
            arguments.pixel_width,
        ),
        dtype=input_dtype,
        key=mx.random.key(arguments.seed),
    )
    encoder, tensor_count = load_encoder(model_directory, arguments.input_dtype)
    latent = encoder.encode(pixels)
    mx.eval(pixels, latent)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {"pixels": pixels, "latent": latent},
        metadata={
            "stage": "video_vae_encode",
            "reference": "dgrauet/ltx-2-mlx@v0.14.22",
            "reference_commit": "64b0a1d2ba358232bb73c6730095593e6e9f3ffb",
            "model": str(model_directory),
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
            "pixel_layout": "BCFHW",
            "latent_layout": "BCFHW",
            "weight_tensors": str(tensor_count),
            "quantized_modules": "0",
        },
    )
    print(f"reference={arguments.output}")
    print(f"weights={tensor_count} tensors")
    print("quantized modules=0")
    print(f"input dtype={arguments.input_dtype}")
    print(f"pixels shape={pixels.shape} dtype={pixels.dtype}")
    print(f"latent shape={latent.shape} dtype={latent.dtype}")


def load_audio_decoder(model_directory: Path, input_dtype: str) -> tuple[AudioVAEDecoder, int]:
    decoder = AudioVAEDecoder()
    decoder_weights = load_split_safetensors(
        model_directory / "audio_vae.safetensors",
        prefix="audio_vae.decoder.",
    )
    all_audio = load_split_safetensors(
        model_directory / "audio_vae.safetensors",
        prefix="audio_vae.",
    )
    for name, value in all_audio.items():
        if name.startswith("per_channel_statistics."):
            decoder_weights[name] = value
    decoder_weights = remap_audio_vae_keys(decoder_weights)
    if input_dtype == "float32":
        decoder_weights = {
            name: value.astype(mx.float32) for name, value in decoder_weights.items()
        }
    decoder.load_weights(list(decoder_weights.items()), strict=True)
    decoder.eval()
    mx.eval(decoder.parameters())
    return decoder, len(decoder_weights)


def dump_audio_vae_decode(arguments: argparse.Namespace) -> None:
    model_directory: Path = arguments.model_dir
    weights_path = model_directory / "audio_vae.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    latent = mx.random.normal(
        (1, 8, arguments.latent_frames, 16),
        dtype=input_dtype,
        key=mx.random.key(arguments.seed),
    )
    decoder, tensor_count = load_audio_decoder(model_directory, arguments.input_dtype)
    mel = decoder.decode(latent)
    mx.eval(latent, mel)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {"latent": latent, "mel": mel},
        metadata={
            "stage": "audio_vae_decode",
            "reference": "dgrauet/ltx-2-mlx@v0.14.22",
            "reference_commit": "64b0a1d2ba358232bb73c6730095593e6e9f3ffb",
            "model": str(model_directory),
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
            "latent_layout": "BCTF",
            "mel_layout": "BCTM",
            "weight_tensors": str(tensor_count),
        },
    )
    print(f"reference={arguments.output}")
    print(f"weights={tensor_count} tensors")
    print(f"input dtype={arguments.input_dtype}")
    print(f"latent shape={latent.shape} dtype={latent.dtype}")
    print(f"mel shape={mel.shape} dtype={mel.dtype}")


def load_transformer(model_directory: Path, weights_name: str, input_dtype: str) -> tuple[LTXModel, int, int]:
    model = LTXModel(LTXModelConfig.from_checkpoint_dir(model_directory))
    weights = load_split_safetensors(
        resolve_model_relative_path(model_directory, weights_name),
        prefix="transformer.",
    )
    apply_quantization(model, weights)
    if input_dtype == "float32":
        weights = {
            name: value if value.dtype == mx.uint32 else value.astype(mx.float32)
            for name, value in weights.items()
        }
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    quantized_modules = sum(1 for name in weights if name.endswith(".scales"))
    return model, len(weights), quantized_modules


def dump_transformer_forward(arguments: argparse.Namespace) -> None:
    model_directory: Path = arguments.model_dir
    weights_path = resolve_model_relative_path(model_directory, arguments.weights)
    config_path = model_directory / "embedded_config.json"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    video_tokens = arguments.video_frames * arguments.video_height * arguments.video_width
    video_latent = mx.random.normal(
        (1, video_tokens, 128), dtype=input_dtype, key=mx.random.key(arguments.seed)
    )
    audio_latent = mx.random.normal(
        (1, arguments.audio_tokens, 128), dtype=input_dtype, key=mx.random.key(arguments.seed + 1)
    )
    timestep = mx.array([arguments.timestep], dtype=input_dtype)
    video_text_embeds = mx.random.normal(
        (1, arguments.text_tokens, 4096), dtype=input_dtype, key=mx.random.key(arguments.seed + 2)
    )
    audio_text_embeds = mx.random.normal(
        (1, arguments.text_tokens, 2048), dtype=input_dtype, key=mx.random.key(arguments.seed + 3)
    )
    video_positions = mx.array(
        [
            [frame, row, column]
            for frame in range(arguments.video_frames)
            for row in range(arguments.video_height)
            for column in range(arguments.video_width)
        ],
        dtype=mx.int32,
    )[None, :, :]
    audio_positions = mx.arange(arguments.audio_tokens, dtype=mx.int32)[None, :, None]
    video_rope_cos, video_rope_sin, _ = precompute_rope_freqs(
        video_positions,
        inner_dim=32 * 128,
        num_heads=32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        rope_type="split",
    )
    video_freq_grid = mx.array(
        10000.0
        ** mx.linspace(
            0.0,
            1.0,
            32 * 128 // (2 * 3),
        )
    ) * (mx.pi / 2.0)

    model, tensor_count, quantized_modules = load_transformer(
        model_directory, arguments.weights, arguments.input_dtype
    )
    video_output, audio_output = model(
        video_latent,
        audio_latent,
        timestep,
        video_text_embeds=video_text_embeds,
        audio_text_embeds=audio_text_embeds,
        video_positions=video_positions,
        audio_positions=audio_positions,
    )
    mx.eval(video_latent, audio_latent, timestep, video_text_embeds, audio_text_embeds,
            video_positions, audio_positions, video_output, audio_output)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "timestep": timestep,
            "video_text_embeds": video_text_embeds,
            "audio_text_embeds": audio_text_embeds,
            "video_positions": video_positions,
            "audio_positions": audio_positions,
            "video_rope_cos": video_rope_cos,
            "video_rope_sin": video_rope_sin,
            "video_freq_grid": video_freq_grid,
            "video_velocity": video_output,
            "audio_velocity": audio_output,
        },
        metadata={
            "stage": "transformer_forward",
            "reference": "dgrauet/ltx-2-mlx@v0.14.22",
            "reference_commit": "64b0a1d2ba358232bb73c6730095593e6e9f3ffb",
            "model": str(model_directory),
            "weights": arguments.weights,
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
            "effective_input_dtype": EFFECTIVE_TRANSFORMER_DTYPE,
            "video_frames": str(arguments.video_frames),
            "video_height": str(arguments.video_height),
            "video_width": str(arguments.video_width),
            "audio_tokens": str(arguments.audio_tokens),
            "text_tokens": str(arguments.text_tokens),
            "weight_tensors": str(tensor_count),
            "quantized_modules": str(quantized_modules),
        },
    )
    print(f"reference={arguments.output}")
    print(f"weights={tensor_count} tensors")
    print(f"quantized modules={quantized_modules}")
    print(f"input dtype={arguments.input_dtype}")
    print(f"video velocity shape={video_output.shape} dtype={video_output.dtype}")
    print(f"audio velocity shape={audio_output.shape} dtype={audio_output.dtype}")


def dump_transformer_tiled(arguments: argparse.Namespace) -> None:
    model_directory: Path = arguments.model_dir
    weights_path = resolve_model_relative_path(model_directory, arguments.weights)
    config_path = model_directory / "embedded_config.json"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    video_tokens = arguments.video_frames * arguments.video_height * arguments.video_width
    video_latent = mx.random.normal(
        (1, video_tokens, 128), dtype=input_dtype, key=mx.random.key(arguments.seed)
    )
    audio_latent = mx.random.normal(
        (1, arguments.audio_tokens, 128), dtype=input_dtype, key=mx.random.key(arguments.seed + 1)
    )
    timestep = mx.array([arguments.timestep], dtype=input_dtype)
    video_text_embeds = mx.random.normal(
        (1, arguments.text_tokens, 4096), dtype=input_dtype, key=mx.random.key(arguments.seed + 2)
    )
    audio_text_embeds = mx.random.normal(
        (1, arguments.text_tokens, 2048), dtype=input_dtype, key=mx.random.key(arguments.seed + 3)
    )
    video_positions = mx.array(
        [
            [frame, row, column]
            for frame in range(arguments.video_frames)
            for row in range(arguments.video_height)
            for column in range(arguments.video_width)
        ],
        dtype=mx.int32,
    )[None, :, :]
    audio_positions = mx.arange(arguments.audio_tokens, dtype=mx.int32)[None, :, None]

    model, tensor_count, quantized_modules = load_transformer(
        model_directory, arguments.weights, arguments.input_dtype
    )
    tile_configuration = TileCountConfig(
        frames=DimensionTilingConfig(arguments.tile_frames, arguments.tile_frame_overlap),
        height=DimensionTilingConfig(arguments.tile_height, arguments.tile_height_overlap),
        width=DimensionTilingConfig(arguments.tile_width, arguments.tile_width_overlap),
    )
    tiler = VideoModalityTiler(
        tile_configuration,
        (arguments.video_frames, arguments.video_height, arguments.video_width),
    )
    video_modality = Modality(
        latent=video_latent,
        sigma=timestep,
        timesteps=mx.broadcast_to(timestep[:, None], (1, video_tokens)),
        positions=video_positions,
        context=video_text_embeds,
    )

    tiled_video_output = None
    tiled_audio_outputs = []
    for tile in tiler.tiles:
        tile_modality, tile_context = tiler.tile_modality(
            video_modality, tile, normalize_positions=False
        )
        tile_video_output, tile_audio_output = model(
            tile_modality.latent,
            audio_latent,
            timestep,
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_positions=tile_modality.positions,
            audio_positions=audio_positions,
        )
        tiled_video_output = tiler.blend(
            tile_video_output, tile, tile_context, output=tiled_video_output
        )
        tiled_audio_outputs.append(tile_audio_output)
    video_output = tiled_video_output
    audio_output = (
        tiled_audio_outputs[0]
        if len(tiled_audio_outputs) == 1
        else mx.mean(mx.stack(tiled_audio_outputs, axis=0), axis=0)
    )
    mx.eval(video_latent, audio_latent, timestep, video_text_embeds, audio_text_embeds,
            video_positions, audio_positions, video_output, audio_output)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "timestep": timestep,
            "video_text_embeds": video_text_embeds,
            "audio_text_embeds": audio_text_embeds,
            "video_positions": video_positions,
            "audio_positions": audio_positions,
            "video_velocity": video_output,
            "audio_velocity": audio_output,
        },
        metadata={
            "stage": "transformer_tiled",
            "reference": "dgrauet/ltx-2-mlx@v0.14.22",
            "reference_commit": "64b0a1d2ba358232bb73c6730095593e6e9f3ffb",
            "model": str(model_directory),
            "weights": arguments.weights,
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
            "effective_input_dtype": EFFECTIVE_TRANSFORMER_DTYPE,
            "video_frames": str(arguments.video_frames),
            "video_height": str(arguments.video_height),
            "video_width": str(arguments.video_width),
            "audio_tokens": str(arguments.audio_tokens),
            "text_tokens": str(arguments.text_tokens),
            "tile_frames": str(arguments.tile_frames),
            "tile_frame_overlap": str(arguments.tile_frame_overlap),
            "tile_height": str(arguments.tile_height),
            "tile_height_overlap": str(arguments.tile_height_overlap),
            "tile_width": str(arguments.tile_width),
            "tile_width_overlap": str(arguments.tile_width_overlap),
            "tile_count": str(len(tiler.tiles)),
            "weight_tensors": str(tensor_count),
            "quantized_modules": str(quantized_modules),
        },
    )
    print(f"reference={arguments.output}")
    print(f"weights={tensor_count} tensors")
    print(f"quantized modules={quantized_modules}")
    print(f"input dtype={arguments.input_dtype} effective dtype=bfloat16")
    print(f"video token count={video_tokens} tile count={len(tiler.tiles)}")
    print(f"video velocity shape={video_output.shape} dtype={video_output.dtype}")
    print(f"audio velocity shape={audio_output.shape} dtype={audio_output.dtype}")


def _cast_parameter_tree(value, dtype):
    if isinstance(value, dict):
        return {key: _cast_parameter_tree(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_cast_parameter_tree(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_parameter_tree(item, dtype) for item in value)
    if isinstance(value, mx.array):
        return value if value.dtype == mx.uint32 else value.astype(dtype)
    return value


def dump_gemma_hidden(arguments: argparse.Namespace) -> None:
    model_directory: Path = arguments.gemma_dir
    config_path = model_directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    encoder = GemmaLanguageModel(model_directory)
    encoder.load()
    compute_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    encoder._model.update(
        _cast_parameter_tree(encoder._model.parameters(), compute_dtype),
        strict=False,
    )
    token_ids, attention_mask = encoder.tokenize(arguments.text, arguments.max_length)
    hidden_states = encoder.get_all_hidden_states(token_ids, attention_mask=attention_mask)
    mx.eval(token_ids, attention_mask, *hidden_states)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
    }
    arrays.update({f"hidden_{index}": value for index, value in enumerate(hidden_states)})
    mx.save_safetensors(
        str(arguments.output),
        arrays,
        metadata={
            "stage": "gemma3_all_hidden_states",
            "reference": "dgrauet/ltx-2-mlx@v0.14.22",
            "reference_commit": "64b0a1d2ba358232bb73c6730095593e6e9f3ffb",
            "model": str(model_directory),
            "input_dtype": arguments.input_dtype,
            "effective_input_dtype": arguments.input_dtype,
            "text": arguments.text,
            "max_length": str(arguments.max_length),
            "hidden_state_count": str(len(hidden_states)),
            "hidden_size": str(hidden_states[0].shape[-1]),
        },
    )
    print(f"reference={arguments.output}")
    print(f"hidden states={len(hidden_states)}")
    print(f"input dtype={arguments.input_dtype}")
    print(f"token ids shape={token_ids.shape} dtype={token_ids.dtype}")
    print(f"hidden shape={hidden_states[0].shape} dtype={hidden_states[0].dtype}")


def dump_gemma_features(arguments: argparse.Namespace) -> None:
    model_directory: Path = arguments.model_dir
    weights_path = model_directory / "connector.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    arrays = mx.load(str(arguments.input))
    hidden_states = [arrays[f"hidden_{index}"] for index in range(49)]
    attention_mask = arrays.get("attention_mask")
    extractor = GemmaFeaturesExtractorV2()
    weights = load_split_safetensors(weights_path, prefix="connector.")
    compute_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    weights = {
        name: value if value.dtype == mx.uint32 else value.astype(compute_dtype)
        for name, value in weights.items()
    }
    extractor.load_weights(list(weights.items()), strict=True)
    extractor.eval()
    video_embeds, audio_embeds = extractor(hidden_states, attention_mask=attention_mask)
    mx.eval(video_embeds, audio_embeds)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {
            "video_text_embeds": video_embeds,
            "audio_text_embeds": audio_embeds,
        },
        metadata={
            "stage": "gemma3_features",
            "reference": "dgrauet/ltx-2-mlx@v0.14.22",
            "reference_commit": "64b0a1d2ba358232bb73c6730095593e6e9f3ffb",
            "model": str(model_directory),
            "input": str(arguments.input),
            "input_dtype": arguments.input_dtype,
            "effective_input_dtype": arguments.input_dtype,
            "weight_tensors": str(len(weights)),
            "quantized_modules": str(sum(name.endswith(".scales") for name in weights)),
        },
    )
    print(f"reference={arguments.output}")
    print(f"weights={len(weights)} tensors")
    print(f"input dtype={arguments.input_dtype}")
    print(f"video embeds shape={video_embeds.shape} dtype={video_embeds.dtype}")
    print(f"audio embeds shape={audio_embeds.shape} dtype={audio_embeds.dtype}")


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

    encode = subparsers.add_parser("vae-encode")
    encode.add_argument("--model-dir", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--seed", type=int, default=7)
    encode.add_argument(
        "--input-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    encode.add_argument("--pixel-frames", type=int, default=9)
    encode.add_argument("--pixel-height", type=int, default=32)
    encode.add_argument("--pixel-width", type=int, default=32)

    audio_decode = subparsers.add_parser("audio-vae-decode")
    audio_decode.add_argument("--model-dir", type=Path, required=True)
    audio_decode.add_argument("--output", type=Path, required=True)
    audio_decode.add_argument("--seed", type=int, default=7)
    audio_decode.add_argument(
        "--input-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    audio_decode.add_argument("--latent-frames", type=int, default=2)

    transformer = subparsers.add_parser(
        "transformer-forward",
        help="代表性多模態 token 規模的單次 Transformer forward",
        description=(
            "產生固定的代表性 fixture：video 8x16x16（2048 tokens）、"
            "audio 256 tokens、text 64 tokens。--weights 是相對於 --model-dir 的檔名。"
        ),
    )
    transformer.add_argument("--model-dir", type=Path, required=True)
    transformer.add_argument("--output", type=Path, required=True)
    transformer.add_argument(
        "--weights",
        default="transformer-distilled-1.1.safetensors",
        help="相對於 --model-dir 的權重檔名（不是完整路徑）",
    )
    transformer.add_argument("--seed", type=int, default=7)
    transformer.add_argument(
        "--input-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    transformer.add_argument("--video-frames", type=int, default=DEFAULT_VIDEO_FRAMES)
    transformer.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    transformer.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    transformer.add_argument("--audio-tokens", type=int, default=DEFAULT_AUDIO_TOKENS)
    transformer.add_argument("--text-tokens", type=int, default=DEFAULT_TEXT_TOKENS)
    transformer.add_argument("--timestep", type=float, default=1.0)

    transformer_tiled = subparsers.add_parser(
        "transformer-tiled",
        help="代表性多模態 token 規模的分塊 Transformer forward",
        description=(
            "產生固定的代表性 fixture：video 8x16x16（2048 tokens）、"
            "audio 256 tokens、text 64 tokens，預設沿時間軸切成 2 tiles。"
            "--weights 是相對於 --model-dir 的檔名。"
        ),
    )
    transformer_tiled.add_argument("--model-dir", type=Path, required=True)
    transformer_tiled.add_argument("--output", type=Path, required=True)
    transformer_tiled.add_argument(
        "--weights",
        default="transformer-distilled-1.1.safetensors",
        help="相對於 --model-dir 的權重檔名（不是完整路徑）",
    )
    transformer_tiled.add_argument("--seed", type=int, default=7)
    transformer_tiled.add_argument(
        "--input-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    transformer_tiled.add_argument("--video-frames", type=int, default=DEFAULT_VIDEO_FRAMES)
    transformer_tiled.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    transformer_tiled.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    transformer_tiled.add_argument("--audio-tokens", type=int, default=DEFAULT_AUDIO_TOKENS)
    transformer_tiled.add_argument("--text-tokens", type=int, default=DEFAULT_TEXT_TOKENS)
    transformer_tiled.add_argument("--timestep", type=float, default=1.0)
    transformer_tiled.add_argument("--tile-frames", type=int, default=2)
    transformer_tiled.add_argument("--tile-frame-overlap", type=int, default=1)
    transformer_tiled.add_argument("--tile-height", type=int, default=1)
    transformer_tiled.add_argument("--tile-height-overlap", type=int, default=0)
    transformer_tiled.add_argument("--tile-width", type=int, default=1)
    transformer_tiled.add_argument("--tile-width-overlap", type=int, default=0)

    gemma_hidden = subparsers.add_parser(
        "gemma-hidden",
        help="輸出 Gemma3 的 49 組 hidden states",
        description=(
            "使用 LTX Python 參考環境產生 token_ids、attention_mask 與全部 hidden states。"
            "--gemma-dir 必須是 Gemma3 模型目錄。"
        ),
    )
    gemma_hidden.add_argument("--gemma-dir", type=Path, required=True)
    gemma_hidden.add_argument("--output", type=Path, required=True)
    gemma_hidden.add_argument("--text", required=True)
    gemma_hidden.add_argument("--max-length", type=int, default=1024)
    gemma_hidden.add_argument(
        "--input-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )

    gemma_features = subparsers.add_parser(
        "gemma-features",
        help="由 Gemma hidden states 產生 LTX 影片／音訊 conditioning",
        description=(
            "讀取 gemma-hidden 產生的 hidden states，執行 connector.safetensors。"
            "--model-dir 必須是含 connector.safetensors 的 LTX 模型目錄。"
        ),
    )
    gemma_features.add_argument("--model-dir", type=Path, required=True)
    gemma_features.add_argument("--input", type=Path, required=True)
    gemma_features.add_argument("--output", type=Path, required=True)
    gemma_features.add_argument(
        "--input-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "vae-decode":
        dump_vae_decode(arguments)
        return
    if arguments.command == "vae-encode":
        dump_vae_encode(arguments)
        return
    if arguments.command == "audio-vae-decode":
        dump_audio_vae_decode(arguments)
        return
    if arguments.command == "transformer-forward":
        dump_transformer_forward(arguments)
        return
    if arguments.command == "transformer-tiled":
        dump_transformer_tiled(arguments)
        return
    if arguments.command == "gemma-hidden":
        dump_gemma_hidden(arguments)
        return
    if arguments.command == "gemma-features":
        dump_gemma_features(arguments)
        return
    raise ValueError(f"unsupported command: {arguments.command}")


if __name__ == "__main__":
    main()
