#!/usr/bin/env python3
"""Create deterministic MiniMax Music 3 parity fixtures."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import resource
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

try:
    from mlx_minimax_music3.flow_transformer import FlowTransformer, FlowTransformerConfig
    from mlx_minimax_music3.language_model import LanguageModel, LanguageModelConfig
    from mlx_minimax_music3.condition_encoder import (
        ConditionEncoder,
        ConditionEncoderConfig,
    )
    from mlx_minimax_music3.config import GenerationConfig, ModelConfig
    from mlx_minimax_music3.prompt import build_cfg_token_ids, build_prompt_text
    from mlx_minimax_music3.rvq_decoder import RVQDecoderConfig, RVQDepthDecoder
    from mlx_minimax_music3.sampling import sample_top_k, semantic_guided_logits
    from mlx_minimax_music3.scheduler import (
        blend_overlap,
        carry_window,
        chunk_starts,
        euler_step,
        flow_timesteps,
        restore_overlap,
        waveform_crop,
    )
    from mlx_minimax_music3.vocoder import Vocoder, VocoderConfig
    from mlx_minimax_music3.audio import write_wav
except ModuleNotFoundError as error:
    _REFERENCE_RUNTIME_IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _REFERENCE_RUNTIME_IMPORT_ERROR = None


def load_vocoder(model_directory: Path) -> Vocoder:
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    vocoder = Vocoder(
        VocoderConfig(
            latent_channels=int(config["dit_in_channels"]),
            decoder_input_dim=int(config["vocoder_input_dim"]),
            decoder_hidden_dim=int(config["vocoder_hidden_dim"]),
            upsampling_ratios=tuple(int(value) for value in config["vocoder_upsampling_ratios"]),
            sampling_rate=int(config.get("output_sampling_rate", config["sample_rate"])),
        )
    )
    index = json.loads(
        (model_directory / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_names = sorted(
        {
            filename
            for key, filename in index["weight_map"].items()
            if key.startswith("vocoder.")
        }
    )
    weights: dict[str, mx.array] = {}
    for shard_name in shard_names:
        shard = mx.load(str(model_directory / shard_name))
        for key, value in shard.items():
            if key.startswith("vocoder."):
                local_key = key.removeprefix("vocoder.")
                if local_key.endswith(".alpha"):
                    value = value.transpose(0, 2, 1)
                if local_key in weights:
                    raise ValueError(f"duplicate vocoder tensor: {local_key}")
                weights[local_key] = value
    vocoder.load_weights(sorted(weights.items()), strict=True)
    vocoder.eval()
    return vocoder


def load_transformer(model_directory: Path) -> FlowTransformer:
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    transformer = FlowTransformer(
        FlowTransformerConfig(
            in_channels=int(config["dit_in_channels"]),
            condition_dim=int(config["condition_out_dim"]),
            num_layers=int(config["dit_num_layers"]),
            num_attention_heads=int(config["dit_num_heads"]),
            attention_head_dim=int(config["dit_head_dim"]),
            ff_inner_dim=int(config["dit_ff_inner_dim"]),
            rotary_dim=int(config["dit_rotary_dim"]),
            fourier_embedding_dim=int(config["dit_fourier_dim"]),
        )
    )

    def is_quantized(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if module.weight.shape[-1] % 64:
            return False
        if path in {"proj_in", "proj_out"}:
            return True
        return path.startswith("transformer_blocks.") and any(
            marker in path
            for marker in (
                ".attn.to_q",
                ".attn.to_k",
                ".attn.to_v",
                ".attn.to_out.0",
                ".ff_in",
                ".ff_out",
            )
        )

    nn.quantize(
        transformer,
        group_size=64,
        bits=4,
        mode="affine",
        class_predicate=is_quantized,
    )
    index = json.loads(
        (model_directory / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_names = sorted(
        {
            filename
            for key, filename in index["weight_map"].items()
            if key.startswith("transformer.")
        }
    )
    weights: dict[str, mx.array] = {}
    for shard_name in shard_names:
        shard = mx.load(str(model_directory / shard_name))
        for key, value in shard.items():
            if key.startswith("transformer."):
                local_key = key.removeprefix("transformer.")
                if local_key in weights:
                    raise ValueError(f"duplicate transformer tensor: {local_key}")
                weights[local_key] = value
    transformer.load_weights(sorted(weights.items()), strict=True)
    transformer.eval()
    print(f"transformer weights={len(weights)} tensors")
    return transformer


def load_language_model(model_directory: Path) -> LanguageModel:
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    language = LanguageModel(LanguageModelConfig.from_dict(config))

    def is_quantized(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear) or module.weight.shape[-1] % 64:
            return False
        return path.startswith("model.layers.") and any(
            marker in path
            for marker in (
                ".self_attn.q_proj",
                ".self_attn.k_proj",
                ".self_attn.v_proj",
                ".self_attn.o_proj",
                ".mlp.gate_proj",
                ".mlp.up_proj",
                ".mlp.down_proj",
            )
        )

    nn.quantize(
        language,
        group_size=64,
        bits=4,
        mode="affine",
        class_predicate=is_quantized,
    )
    index = json.loads(
        (model_directory / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_names = sorted(
        {
            filename
            for key, filename in index["weight_map"].items()
            if key.startswith("language_model.")
        }
    )
    weights: dict[str, mx.array] = {}
    for shard_name in shard_names:
        shard = mx.load(str(model_directory / shard_name))
        for key, value in shard.items():
            if key.startswith("language_model."):
                local_key = key.removeprefix("language_model.")
                if local_key in weights:
                    raise ValueError(f"duplicate language model tensor: {local_key}")
                weights[local_key] = value
    language.load_weights(sorted(weights.items()), strict=True)
    language.eval()
    print(f"language model weights={len(weights)} tensors")
    return language


def load_condition_encoder(model_directory: Path) -> ConditionEncoder:
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    condition = ConditionEncoder(
        ConditionEncoderConfig(
            condition_hidden_dim=int(config["hidden_size"]),
            num_condition_layers=int(config["num_condition_layers"]),
            out_dim=int(config["condition_out_dim"]),
            input_sampling_rate=int(config["input_sampling_rate"]),
            input_hop_length=int(config["input_hop_length"]),
            output_sampling_rate=int(config["output_sampling_rate"]),
            output_hop_length=int(config["output_hop_length"]),
        )
    )
    index = json.loads(
        (model_directory / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_names = sorted(
        {
            filename
            for key, filename in index["weight_map"].items()
            if key.startswith("condition_encoder.")
        }
    )
    weights: dict[str, mx.array] = {}
    for shard_name in shard_names:
        shard = mx.load(str(model_directory / shard_name))
        for key, value in shard.items():
            if key.startswith("condition_encoder."):
                local_key = key.removeprefix("condition_encoder.")
                if local_key in weights:
                    raise ValueError(f"duplicate condition encoder tensor: {local_key}")
                weights[local_key] = value
    condition.load_weights(sorted(weights.items()), strict=True)
    condition.eval()
    print(f"condition encoder weights={len(weights)} tensors")
    return condition


def load_rvq_decoder(model_directory: Path) -> RVQDepthDecoder:
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    decoder = RVQDepthDecoder(
        RVQDecoderConfig(
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["depth_num_layers"]),
            num_attention_heads=int(config["depth_num_heads"]),
            intermediate_size=int(config["depth_intermediate_size"]),
            audio_vocab_size=int(config["audio_vocab_size"]),
            num_codebooks=int(config["num_codebooks"]),
            max_position_embeddings=int(config["depth_max_position_embeddings"]),
            rms_norm_eps=float(config["rms_norm_eps"]),
        )
    )

    def is_quantized(path: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear) or module.weight.shape[-1] % 64:
            return False
        if not path.startswith("layers."):
            return False
        return any(
            marker in path
            for marker in (
                ".attn.to_q",
                ".attn.to_k",
                ".attn.to_v",
                ".attn.to_out",
                ".gate_proj",
                ".up_proj",
                ".down_proj",
            )
        )

    nn.quantize(
        decoder,
        group_size=64,
        bits=4,
        mode="affine",
        class_predicate=is_quantized,
    )
    index = json.loads(
        (model_directory / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_names = sorted(
        {
            filename
            for key, filename in index["weight_map"].items()
            if key.startswith("rvq_depth_decoder.")
        }
    )
    weights: dict[str, mx.array] = {}
    for shard_name in shard_names:
        shard = mx.load(str(model_directory / shard_name))
        for key, value in shard.items():
            if key.startswith("rvq_depth_decoder."):
                local_key = key.removeprefix("rvq_depth_decoder.")
                if local_key in weights:
                    raise ValueError(f"duplicate RVQ decoder tensor: {local_key}")
                weights[local_key] = value
    decoder.load_weights(sorted(weights.items()), strict=True)
    decoder.eval()
    print(f"RVQ decoder weights={len(weights)} tensors")
    return decoder


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "decode",
            "decode-chunks",
            "transformer-forward",
            "denoise",
            "generate",
            "level1",
            "language-forward",
            "attention-probe",
            "condition-forward",
            "rvq-forward",
            "rvq-embeddings",
            "frame-hiddens",
        ),
        default="decode",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--prompt", default="sunset over a quiet ocean")
    parser.add_argument("--lyrics", default="[Verse] gentle waves")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wav-output", type=Path)
    parser.add_argument("--latent-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--input-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="dtype for model inputs in parity probes",
    )
    parser.add_argument("--latent", type=Path)
    parser.add_argument("--latent-key", default="latent")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--cfg", type=float, default=1.7)
    parser.add_argument("--flow-cfg", type=float, default=1.7)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--previous-length", type=int)
    parser.add_argument(
        "--text-ids",
        default="151644,77091,151645,151643,151643",
        help="comma-separated token IDs for language-forward",
    )
    parser.add_argument(
        "--encode-prompt",
        action="store_true",
        help="encode --prompt and --lyrics into CFG token IDs for frame-hiddens",
    )
    parser.add_argument(
        "--token-trace",
        action="store_true",
        help="print each generated semantic and depth token",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--ar-cfg", type=float, default=1.5)
    parser.add_argument("--audio-duration", type=float, default=0.2)
    parser.add_argument("--expected-chunks", type=int)
    return parser.parse_args()


def require_model_directory(arguments: argparse.Namespace) -> Path:
    if arguments.model_dir is None:
        raise ValueError("--model-dir is required for this command")
    return arguments.model_dir


def dump_decode(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    if arguments.latent_frames < 1:
        raise ValueError("--latent-frames must be positive")
    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    vocoder = load_vocoder(model_directory)
    if arguments.latent:
        arrays = mx.load(str(arguments.latent))
        latent = arrays[arguments.latent_key]
    else:
        latent = mx.random.normal(
            (1, 128, arguments.latent_frames),
            dtype=input_dtype,
            key=mx.random.key(arguments.seed),
        )
    if latent.dtype != input_dtype:
        latent = latent.astype(input_dtype)
    if latent.ndim != 3 or latent.shape[0] < 1 or latent.shape[1] != 128:
        raise ValueError(f"latent must have shape [batch, 128, frames], got {latent.shape}")
    audio = mx.clip(vocoder(latent.transpose(0, 2, 1)).astype(mx.float32), -1.0, 1.0)
    mx.eval(latent, audio)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {"latent": latent, "audio": audio},
        metadata={
            "stage": "decode_chunks",
            "model": str(arguments.model_dir),
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
            "sampling_rate": "44100",
            "latent_layout": "BCL",
            "audio_layout": "BCS",
        },
    )
    print(f"reference={arguments.output}")
    print(f"latent shape={latent.shape} dtype={latent.dtype}")
    print(f"audio shape={audio.shape} dtype={audio.dtype}")


def dump_decode_chunks(arguments: argparse.Namespace) -> None:
    """Dump real denoiser chunks and the Python multi-chunk vocoder result.

    This follows the upstream MiniMax Music 3 generation path through
    autoregressive frame generation and flow denoising. The fixture therefore
    does not use synthetic, independently sampled latent chunks.
    """

    from mlx_audio.music import load as load_music_model
    from mlx_audio.music.models.minimax_music3.ar import generate_frame_hiddens
    from mlx_audio.music.models.minimax_music3.config import (
        CHUNK_FRAMES,
        CROP_LEFT_LATENT,
        CROP_RIGHT_LATENT,
        DIT_CFG_SCALE,
        LATENT_HOP_LENGTH,
        OVERLAP_LATENT_LENGTH,
    )
    from mlx_audio.music.models.minimax_music3.euler import denoise_chunk
    from mlx_audio.music.models.minimax_music3.minimax_music3 import _chunk_starts

    model_directory = require_model_directory(arguments)
    if arguments.audio_duration <= 0:
        raise ValueError("--audio-duration must be positive")
    if arguments.steps < 1:
        raise ValueError("--steps must be positive")
    if arguments.expected_chunks is not None and arguments.expected_chunks < 1:
        raise ValueError("--expected-chunks must be positive")

    model = load_music_model(model_directory, strict=True)
    maximum_frames = max(1, int(arguments.audio_duration * model.config.frame_rate))
    text_ids = model._text_ids(arguments.prompt, arguments.lyrics)
    frame_hiddens = generate_frame_hiddens(
        model.language_model,
        model.rvq_depth_decoder,
        model.config,
        text_ids,
        max_frames=maximum_frames,
        seed=arguments.seed,
    )
    mx.eval(frame_hiddens)

    starts = _chunk_starts(frame_hiddens.shape[1])
    latent_chunks = []
    previous_latent = None
    previous_condition = None
    mx.random.seed(arguments.seed + 7)
    for start in starts:
        end = min(start + CHUNK_FRAMES, frame_hiddens.shape[1])
        condition = model.condition_encoder(frame_hiddens[:, start:end])
        noise = mx.random.normal(
            (1, model.config.dit_in_channels, condition.shape[1])
        ).astype(condition.dtype)
        latents, condition = denoise_chunk(
            model.transformer,
            noise,
            condition,
            num_inference_steps=arguments.steps,
            guidance_scale=DIT_CFG_SCALE,
            previous_latent=previous_latent,
            previous_condition=previous_condition,
        )
        carry_start = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT_LENGTH)
        carry_end = max(carry_start, latents.shape[-1] - OVERLAP_LATENT_LENGTH)
        previous_latent = latents[..., carry_start:carry_end]
        previous_condition = condition[:, carry_start:carry_end]
        latent_chunks.append(latents)
        mx.eval(previous_latent, previous_condition, latents)

    if arguments.expected_chunks is not None and len(latent_chunks) != arguments.expected_chunks:
        raise RuntimeError(
            f"expected {arguments.expected_chunks} chunks, got {len(latent_chunks)} "
            f"from {frame_hiddens.shape[1]} real frames"
        )

    waveforms = []
    diagnostics = []
    chunk_count = len(latent_chunks)
    for index, latents in enumerate(latent_chunks):
        waveform = model.vocoder(latents)
        left = 0 if index == 0 else CROP_LEFT_LATENT * LATENT_HOP_LENGTH
        right = 0 if index == chunk_count - 1 else CROP_RIGHT_LATENT * LATENT_HOP_LENGTH
        raw_end = waveform.shape[-1] - right if right else waveform.shape[-1]
        cropped = waveform[..., left:raw_end]
        waveforms.append(cropped)
        diagnostics.append(
            {
                "latent_frames": int(latents.shape[-1]),
                "waveform_samples": int(waveform.shape[-1]),
                "crop_left": int(left),
                "crop_right": int(right),
                "raw_end": int(raw_end),
                "retained_samples": int(cropped.shape[-1]),
            }
        )
        mx.eval(cropped)

    audio = mx.clip(
        mx.concatenate(waveforms, axis=-1).astype(mx.float32),
        -1.0,
        1.0,
    )
    mx.eval(audio)
    values = {
        "audio": audio,
        "frame_hiddens": frame_hiddens,
    }
    values.update(
        {f"latent_chunk_{index}": chunk for index, chunk in enumerate(latent_chunks)}
    )
    metadata = {
        "stage": "decode_chunks",
        "reference": "Blaizzy/mlx-audio@784b29e2691a93ca7483147d86f61859dfaa6296",
        "model": str(model_directory),
        "seed": str(arguments.seed),
        "audio_duration": str(arguments.audio_duration),
        "steps": str(arguments.steps),
        "input_dtype": str(latent_chunks[0].dtype).removeprefix("mlx.core."),
        "sampling_rate": str(model.config.sample_rate),
        "num_frames": str(frame_hiddens.shape[1]),
        "chunk_count": str(chunk_count),
        "latent_layout": "BCL",
        "audio_layout": "BCS",
        "chunk_diagnostics": json.dumps(diagnostics, separators=(",", ":")),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(arguments.output), values, metadata=metadata)

    print(f"reference={arguments.output}")
    print(
        f"real frames={frame_hiddens.shape[1]} requested maximum={maximum_frames} "
        f"chunks={chunk_count}"
    )
    for index, diagnostic in enumerate(diagnostics):
        print(
            f"chunk[{index}] latent={diagnostic['latent_frames']} "
            f"waveform={diagnostic['waveform_samples']} "
            f"crop=({diagnostic['crop_left']},{diagnostic['crop_right']}) "
            f"raw_end={diagnostic['raw_end']} "
            f"retained={diagnostic['retained_samples']}"
        )
    print(f"audio shape={audio.shape} dtype={audio.dtype}")


def dump_transformer_forward(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    if arguments.frames < 1:
        raise ValueError("--frames must be positive")
    transformer = load_transformer(model_directory)
    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    keys = mx.random.split(mx.random.key(arguments.seed), 2)
    latents = mx.random.normal(
        (1, arguments.frames, 128), dtype=input_dtype, key=keys[0]
    )
    condition = mx.random.normal(
        (1, arguments.frames, 2048), dtype=input_dtype, key=keys[1]
    )
    timestep = mx.array([0.37], dtype=mx.float32)
    velocity = transformer(latents, timestep, condition)
    mx.eval(latents, timestep, condition, velocity)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {
            "latents": latents,
            "timestep": timestep,
            "condition": condition,
            "velocity": velocity,
        },
        metadata={
            "stage": "transformer_forward",
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
            "latent_layout": "BLC",
            "condition_layout": "BLC",
        },
    )
    print(f"reference={arguments.output}")
    print(f"latents shape={latents.shape} dtype={latents.dtype}")
    print(f"condition shape={condition.shape} dtype={condition.dtype}")
    print(f"velocity shape={velocity.shape} dtype={velocity.dtype}")


def dump_level1(arguments: argparse.Namespace) -> None:
    tokenizer_directory = arguments.tokenizer_dir
    if tokenizer_directory is None:
        tokenizer_directory = require_model_directory(arguments) / "tokenizer"
    tokenizer = Tokenizer.from_file(str(tokenizer_directory / "tokenizer.json"))
    prompt_text = build_prompt_text(arguments.prompt, arguments.lyrics)
    token_ids = build_cfg_token_ids(
        arguments.prompt,
        arguments.lyrics,
        lambda text: tokenizer.encode(text).ids,
    )
    logits = mx.array(
        [
            [0.25, 4.0, 2.0, float("nan"), -3.0, 1.0, 3.5, 2.5,
             1.5, 0.5, -0.5, -1.5, 5.0, 4.5, 3.0, 2.0],
            [0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.5, 0.5,
             0.25, -0.25, -1.0, -2.0, 0.5, 1.5, 2.5, 3.5],
        ],
        dtype=mx.float32,
    )
    allowed = mx.array(
        [True, False, True, True, False, False, True, True,
         False, False, False, False, True, True, True, False]
    )
    guided = semantic_guided_logits(
        logits,
        allowed,
        cfg_scale=arguments.ar_cfg,
        conditional_top_k=arguments.top_k,
    )
    initial_key = mx.random.key(arguments.seed)
    sampled, next_key = sample_top_k(guided, initial_key, top_k=arguments.top_k)
    token_ids_array = mx.array(token_ids, dtype=mx.int32)
    mx.eval(token_ids_array, logits, allowed, guided, initial_key, sampled, next_key)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {
            "conditional_token_ids": token_ids_array[0],
            "unconditional_token_ids": token_ids_array[1],
            "logits": logits,
            "allowed_vocabulary": allowed,
            "guided_logits": guided,
            "initial_key": initial_key,
            "sampled": sampled,
            "next_key": next_key,
        },
        metadata={
            "stage": "level1",
            "prompt": arguments.prompt,
            "lyrics": arguments.lyrics,
            "prompt_text": prompt_text,
            "seed": str(arguments.seed),
            "ar_cfg": str(arguments.ar_cfg),
            "top_k": str(arguments.top_k),
        },
    )
    print(f"reference={arguments.output}")
    print(f"prompt tokens shape={token_ids_array.shape}")
    print(f"guided logits shape={guided.shape} dtype={guided.dtype}")
    print(f"sampled={sampled.tolist()} next_key={next_key.tolist()}")


def dump_language_forward(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    token_values = [int(value.strip()) for value in arguments.text_ids.split(",") if value.strip()]
    if not token_values:
        raise ValueError("--text-ids must contain at least one token ID")
    language = load_language_model(model_directory)
    text_ids = mx.array([token_values], dtype=mx.int32)
    if arguments.input_dtype == "float32":
        input_embeddings = language.model.embed_tokens(text_ids).astype(mx.float32)
        hidden_states = language.hidden_states(input_embeddings=input_embeddings)
    else:
        hidden_states = language.hidden_states(input_ids=text_ids)
    logits = language.logits(hidden_states)
    mx.eval(text_ids, hidden_states, logits)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {"text_ids": text_ids, "hidden_states": hidden_states, "logits": logits},
        metadata={
            "stage": "language_forward",
            "input_dtype": arguments.input_dtype,
            "text_ids": arguments.text_ids,
        },
    )
    print(f"reference={arguments.output}")
    print(f"text_ids shape={text_ids.shape} dtype={text_ids.dtype}")
    print(f"hidden states shape={hidden_states.shape} dtype={hidden_states.dtype}")
    print(f"logits shape={logits.shape} dtype={logits.dtype}")


def dump_attention_probe(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    token_values = [int(value.strip()) for value in arguments.text_ids.split(",") if value.strip()]
    if not token_values:
        raise ValueError("--text-ids must contain at least one token ID")
    language = load_language_model(model_directory)
    text_ids = mx.array([token_values], dtype=mx.int32)
    layer = language.model.layers[0]
    embedded = language.model.embed_tokens(text_ids)
    normalized = layer.input_layernorm(embedded)
    batch, length, _ = normalized.shape
    attention = layer.self_attn
    queries = attention.q_norm(
        attention.q_proj(normalized).reshape(batch, length, attention.n_heads, -1)
    ).transpose(0, 2, 1, 3)
    keys = attention.k_norm(
        attention.k_proj(normalized).reshape(batch, length, attention.n_kv_heads, -1)
    ).transpose(0, 2, 1, 3)
    values = attention.v_proj(normalized).reshape(batch, length, attention.n_kv_heads, -1).transpose(
        0, 2, 1, 3
    )
    queries_before_rope = queries
    keys_before_rope = keys
    queries = attention.rope(queries)
    keys = attention.rope(keys)
    sdpa_output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=attention.scale, mask="causal"
    )
    attention_output = attention.o_proj(
        sdpa_output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    )
    attention_residual = embedded + attention_output
    mlp_input = layer.post_attention_layernorm(attention_residual)
    gate_output = layer.mlp.gate_proj(mlp_input)
    up_output = layer.mlp.up_proj(mlp_input)
    activated_output = nn.silu(gate_output) * up_output
    mlp_output = layer.mlp.down_proj(activated_output)
    layer_output = attention_residual + mlp_output
    mx.eval(
        embedded,
        normalized,
        queries_before_rope,
        keys_before_rope,
        queries,
        keys,
        values,
        sdpa_output,
        attention_output,
        attention_residual,
        mlp_input,
        gate_output,
        up_output,
        activated_output,
        mlp_output,
        layer_output,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {
            "text_ids": text_ids,
            "embedded": embedded,
            "normalized": normalized,
            "queries_before_rope": queries_before_rope,
            "keys_before_rope": keys_before_rope,
            "queries": queries,
            "keys": keys,
            "values": values,
            "sdpa_output": sdpa_output,
            "attention_output": attention_output,
            "attention_residual": attention_residual,
            "mlp_input": mlp_input,
            "gate_output": gate_output,
            "up_output": up_output,
            "activated_output": activated_output,
            "mlp_output": mlp_output,
            "layer_output": layer_output,
        },
        metadata={
            "stage": "attention_probe",
            "input_dtype": str(queries.dtype).removeprefix("mlx.core."),
            "text_ids": arguments.text_ids,
        },
    )
    print(f"reference={arguments.output}")
    print(f"queries shape={queries.shape} dtype={queries.dtype}")
    print(f"keys shape={keys.shape} dtype={keys.dtype}")
    print(f"values shape={values.shape} dtype={values.dtype}")
    print(f"output shape={layer_output.shape} dtype={layer_output.dtype}")


def dump_frame_hiddens(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    token_values = [int(value.strip()) for value in arguments.text_ids.split(",") if value.strip()]
    if not token_values:
        raise ValueError("--text-ids must contain at least one token ID")
    language = load_language_model(model_directory)
    decoder = load_rvq_decoder(model_directory)
    if arguments.encode_prompt:
        tokenizer_directory = arguments.tokenizer_dir or model_directory / "tokenizer"
        tokenizer = Tokenizer.from_file(str(tokenizer_directory / "tokenizer.json"))
        token_ids = build_cfg_token_ids(
            arguments.prompt,
            arguments.lyrics,
            lambda text: tokenizer.encode(text).ids,
            ModelConfig(),
        )
        text_ids = mx.array(token_ids, dtype=mx.int32)
    else:
        text_ids = mx.array([token_values, token_values], dtype=mx.int32)
    generation = GenerationConfig(
        audio_duration=arguments.audio_duration,
        seed=arguments.seed,
        ar_cfg_scale=arguments.ar_cfg,
        top_k=arguments.top_k,
    )
    model_config = ModelConfig()
    initial_key = mx.random.key(arguments.seed)
    key = initial_key
    from mlx_lm.models.cache import KVCache

    cache = [KVCache() for _ in range(language.config.num_hidden_layers)]
    if arguments.input_dtype == "float32":
        initial_embeddings = language.model.embed_tokens(text_ids).astype(mx.float32)
        initial_hidden = language.hidden_states(
            input_embeddings=initial_embeddings,
            cache=cache,
        )
    else:
        initial_hidden = language.hidden_states(input_ids=text_ids, cache=cache)
    hidden = initial_hidden
    last_hidden = hidden[:, -1]
    vocab = mx.arange(language.config.vocab_size, dtype=mx.int32)
    start = model_config.audio_code_offset
    stop = start + model_config.semantic_vocab_size
    allowed_vocab = ((vocab >= start) & (vocab < stop)) | (
        vocab == model_config.audio_end_token_id
    )
    frame_hiddens = []
    semantic_codes = []
    frame_codes_trace = []
    depth_guided_logits_trace = []
    stopped_by_end_token = False
    iterations = 0
    max_frames = generation.max_frames(model_config)
    for frame_index in range(max_frames + 1):
        iterations += 1
        logits = language.logits(last_hidden)
        guided = semantic_guided_logits(
            logits,
            allowed_vocab,
            cfg_scale=generation.ar_cfg_scale,
            conditional_top_k=generation.top_k,
        )
        sampled, key = sample_top_k(guided, key, top_k=generation.top_k)
        mx.eval(sampled)
        if int(sampled.item()) == model_config.audio_end_token_id:
            stopped_by_end_token = True
            break
        semantic_codes.append(sampled)
        semantic_code = (sampled - model_config.audio_code_offset).astype(mx.int32)
        semantic_pair = mx.repeat(semantic_code, 2, axis=0)
        semantic_embedding = language.model.embed_tokens(
            semantic_pair + model_config.audio_code_offset
        )
        if arguments.input_dtype == "float32":
            semantic_embedding = semantic_embedding.astype(mx.float32)
        sequence = [
            decoder.projection(last_hidden)[:, None, :],
            decoder.projection(semantic_embedding)[:, None, :],
        ]
        codes = [semantic_pair]
        hidden_parts = []
        depth_guided_logits = []
        for index in range(1, decoder.config.num_codebooks):
            depth_hidden = decoder(mx.concatenate(sequence, axis=1))
            hidden_parts.append(depth_hidden[:1, -1])
            depth_logits = decoder.audio_heads[index - 1](depth_hidden[:, -1]).astype(mx.float32)
            conditional, unconditional = depth_logits[0:1], depth_logits[1:2]
            depth_guided = unconditional + (
                conditional - unconditional
            ) * generation.ar_cfg_scale
            depth_guided_logits.append(depth_guided[0])
            depth_sampled, key = sample_top_k(
                depth_guided,
                key,
                top_k=generation.top_k,
            )
            code = mx.repeat(depth_sampled, 2, axis=0).astype(mx.int32)
            codes.append(code)
            if index < decoder.config.num_codebooks - 1:
                embedding_id = code + (index - 1) * decoder.config.audio_vocab_size
                sequence.append(
                    decoder.projection(decoder.audio_embeddings(embedding_id))[:, None, :]
                )
        if frame_index > 0:
            frame_hiddens.append(
                mx.concatenate((last_hidden[:1], mx.concatenate(hidden_parts, axis=-1)), axis=-1)
            )
            if len(frame_hiddens) >= max_frames:
                break
        frame_codes = mx.stack(codes, axis=1)
        frame_codes_trace.append(frame_codes)
        depth_guided_logits_trace.append(mx.stack(depth_guided_logits, axis=0))
        semantic_embedding = semantic_embedding[:, None, :]
        residual_codes = frame_codes[:, 1:]
        feedback = (
            semantic_embedding
            + mx.sum(
                decoder.audio_embeddings(
                    residual_codes
                    + mx.arange(decoder.config.num_codebooks - 1, dtype=mx.int32)[None, :]
                    * decoder.config.audio_vocab_size
                ),
                axis=1,
                keepdims=True,
            )
        ) * decoder.config.num_codebooks**-0.5
        if arguments.input_dtype == "float32":
            feedback = feedback.astype(mx.float32)
        hidden = language.hidden_states(input_embeddings=feedback, cache=cache)
        last_hidden = hidden[:, -1]
        mx.eval(last_hidden)
    if not frame_hiddens:
        raise ValueError("MiniMax Music 3 generated zero audio frames")
    frame_hiddens = mx.stack(frame_hiddens, axis=1)
    semantic_codes = mx.concatenate(semantic_codes, axis=0)
    frame_codes_trace = mx.stack(frame_codes_trace, axis=0)
    depth_guided_logits_trace = mx.stack(depth_guided_logits_trace, axis=0)
    mx.eval(
        text_ids,
        key,
        frame_hiddens,
        semantic_codes,
        frame_codes_trace,
        depth_guided_logits_trace,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {
            "text_ids": text_ids,
            "initial_key": initial_key,
            "initial_hidden": initial_hidden,
            "frame_hiddens": frame_hiddens,
            "semantic_codes": semantic_codes,
            "frame_codes": frame_codes_trace,
            "depth_guided_logits": depth_guided_logits_trace,
            "next_key": key,
        },
        metadata={
            "stage": "generate_frame_hiddens",
            "seed": str(arguments.seed),
            "audio_duration": str(arguments.audio_duration),
            "ar_cfg": str(arguments.ar_cfg),
            "top_k": str(arguments.top_k),
            "input_dtype": arguments.input_dtype,
            "iterations": str(iterations),
            "stopped_by_end_token": str(stopped_by_end_token),
        },
    )
    print(f"reference={arguments.output}")
    print(f"frame hiddens shape={frame_hiddens.shape} dtype={frame_hiddens.dtype}")
    print(f"iterations={iterations} stopped_by_end_token={stopped_by_end_token}")
    if arguments.token_trace:
        for index, semantic_code in enumerate(semantic_codes.tolist()):
            depth_codes = (
                frame_codes_trace[index, 0, :].tolist()
                if index < frame_codes_trace.shape[0]
                else []
            )
            print(f"token-step[{index + 1}] semantic={semantic_code} depth={depth_codes}")


def dump_condition_forward(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    if arguments.frames < 1:
        raise ValueError("--frames must be positive")
    condition_encoder = load_condition_encoder(model_directory)
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    hidden_states = mx.random.normal(
        (
            1,
            arguments.frames,
            int(config["num_condition_layers"]) * int(config["hidden_size"]),
        ),
        dtype=input_dtype,
        key=mx.random.key(arguments.seed),
    )
    condition = condition_encoder(hidden_states)
    mx.eval(hidden_states, condition)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {"hidden_states": hidden_states, "condition": condition},
        metadata={
            "stage": "condition_encoder",
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
            "input_layout": "BLF",
            "output_layout": "BLC",
        },
    )
    print(f"reference={arguments.output}")
    print(f"hidden states shape={hidden_states.shape} dtype={hidden_states.dtype}")
    print(f"condition shape={condition.shape} dtype={condition.dtype}")


def dump_rvq_forward(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    if arguments.frames < 1:
        raise ValueError("--frames must be positive")
    if arguments.frames > 16:
        raise ValueError("--frames must not exceed 16 for the RVQ depth decoder")
    decoder = load_rvq_decoder(model_directory)
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    inputs_embeds = mx.random.normal(
        (2, arguments.frames, int(config["hidden_size"])),
        dtype=input_dtype,
        key=mx.random.key(arguments.seed),
    )
    hidden_states = decoder(inputs_embeds)
    mx.eval(inputs_embeds, hidden_states)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {"inputs_embeds": inputs_embeds, "hidden_states": hidden_states},
        metadata={
            "stage": "rvq_decoder_forward",
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
            "input_layout": "BLC",
            "output_layout": "BLC",
        },
    )
    print(f"reference={arguments.output}")
    print(f"inputs embeds shape={inputs_embeds.shape} dtype={inputs_embeds.dtype}")
    print(f"hidden states shape={hidden_states.shape} dtype={hidden_states.dtype}")


def dump_rvq_embeddings(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    decoder = load_rvq_decoder(model_directory)
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    input_dtype = mx.bfloat16 if arguments.input_dtype == "bfloat16" else mx.float32
    key = mx.random.key(arguments.seed)
    keys = mx.random.split(key, 2)
    semantic_embedding = mx.random.normal(
        (2, 1, int(config["hidden_size"])), dtype=input_dtype, key=keys[0]
    )
    residual_codes = mx.random.randint(
        0,
        int(config["audio_vocab_size"]),
        (2, int(config["num_codebooks"]) - 1),
        dtype=mx.int32,
        key=keys[1],
    )
    offsets = mx.arange(int(config["num_codebooks"]) - 1, dtype=mx.int32)
    offsets = offsets * int(config["audio_vocab_size"])
    embedding_ids = residual_codes + offsets[None, :]
    residual_embeddings = decoder.audio_embeddings(embedding_ids)
    combined = (
        semantic_embedding + mx.sum(residual_embeddings, axis=1, keepdims=True)
    ) * int(arguments.num_codebooks) ** -0.5
    mx.eval(semantic_embedding, residual_codes, residual_embeddings, combined)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {
            "semantic_embedding": semantic_embedding,
            "residual_codes": residual_codes,
            "residual_embeddings": residual_embeddings,
            "combined": combined,
        },
        metadata={
            "stage": "rvq_audio_embedding",
            "seed": str(arguments.seed),
            "input_dtype": arguments.input_dtype,
        },
    )
    print(f"reference={arguments.output}")
    print(f"semantic embedding shape={semantic_embedding.shape} dtype={semantic_embedding.dtype}")
    print(f"residual codes shape={residual_codes.shape} dtype={residual_codes.dtype}")
    print(f"combined shape={combined.shape} dtype={combined.dtype}")


def dump_denoise(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    if arguments.frames < 1:
        raise ValueError("--frames must be positive")
    if arguments.steps < 1:
        raise ValueError("--steps must be positive")
    if arguments.overlap < 0 or arguments.overlap > arguments.frames:
        raise ValueError("--overlap must be within the latent sequence")
    previous_length = arguments.previous_length
    if previous_length is None:
        previous_length = arguments.overlap
    if previous_length < arguments.overlap:
        raise ValueError("--previous-length must cover --overlap")
    transformer = load_transformer(model_directory)
    key_values = mx.random.split(mx.random.key(arguments.seed), 4)
    latents = mx.random.normal(
        (1, arguments.frames, 128), dtype=mx.bfloat16, key=key_values[0]
    )
    initial_latents = latents
    condition = mx.random.normal(
        (1, arguments.frames, 2048), dtype=mx.bfloat16, key=key_values[1]
    )
    previous_latent = None
    if arguments.overlap:
        previous_latent = mx.random.normal(
            (1, previous_length, 128), dtype=mx.bfloat16, key=key_values[2]
        )
    noise_prompt = latents[:, :arguments.overlap]
    for timestep in mx.arange(arguments.steps, dtype=mx.float32) / arguments.steps:
        if arguments.overlap:
            from mlx_minimax_music3.scheduler import blend_overlap

            latents = blend_overlap(
                latents,
                noise_prompt,
                previous_latent,
                arguments.overlap,
                timestep,
            )
        time_batch = mx.broadcast_to(timestep, (latents.shape[0],))
        conditional = transformer(latents, time_batch, condition)
        unconditional = transformer(latents, time_batch, mx.zeros_like(condition))
        velocity = unconditional + arguments.cfg * (conditional - unconditional)
        from mlx_minimax_music3.scheduler import euler_step

        latents = euler_step(latents, velocity, arguments.steps)
        mx.eval(latents)
    if arguments.overlap:
        from mlx_minimax_music3.scheduler import restore_overlap

        latents = restore_overlap(latents, previous_latent, arguments.overlap)
    mx.eval(latents, noise_prompt)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "initial_latents": initial_latents,
        "condition": condition,
        "final_latents": latents,
    }
    if previous_latent is not None:
        values["previous_latent"] = previous_latent
        values["noise_prompt"] = noise_prompt
    mx.save_safetensors(
        str(arguments.output),
        values,
        metadata={
            "stage": "denoise_chunks",
            "seed": str(arguments.seed),
            "steps": str(arguments.steps),
            "cfg": str(arguments.cfg),
            "overlap": str(arguments.overlap),
        },
    )
    print(f"reference={arguments.output}")
    print(f"final_latents shape={latents.shape} dtype={latents.dtype}")
    print(f"steps={arguments.steps} overlap={arguments.overlap}")


def dump_generate(arguments: argparse.Namespace) -> None:
    model_directory = require_model_directory(arguments)
    started_at = time.perf_counter()
    frame_arguments = copy.copy(arguments)
    with tempfile.TemporaryDirectory(prefix="genimage-minimax-frame-") as temporary:
        frame_arguments.output = Path(temporary) / "frame-hiddens.safetensors"
        frame_arguments.encode_prompt = True
        frame_arguments.token_trace = False
        dump_frame_hiddens(frame_arguments)
        frame_arrays = mx.load(str(frame_arguments.output))
        frame_hiddens = frame_arrays["frame_hiddens"]
        next_key = frame_arrays["next_key"]
        frame_arrays = None

    gc.collect()
    model_config = ModelConfig()
    generation = GenerationConfig(
        audio_duration=arguments.audio_duration,
        seed=arguments.seed,
        num_inference_steps=arguments.steps,
        ar_cfg_scale=arguments.ar_cfg,
        flow_cfg_scale=arguments.flow_cfg,
        top_k=arguments.top_k,
    )
    generation.validate()
    condition_encoder = load_condition_encoder(model_directory)
    transformer = load_transformer(model_directory)
    vocoder = load_vocoder(model_directory)
    latent_chunks = []
    previous_latent = None
    previous_condition = None
    for start in chunk_starts(frame_hiddens.shape[1]):
        end = min(start + 200, frame_hiddens.shape[1])
        condition = condition_encoder(frame_hiddens[:, start:end])
        overlap = 0
        if previous_latent is not None and previous_condition is not None:
            overlap = min(previous_latent.shape[1], condition.shape[1])
            condition = mx.concatenate(
                (previous_condition[:, :overlap], condition[:, overlap:]), axis=1
            )
        next_key, noise_key = mx.random.split(next_key, 2)
        latents = mx.random.normal(
            (1, condition.shape[1], model_config.latent_channels),
            dtype=condition.dtype,
            key=noise_key,
        )
        noise_prompt = latents[:, :overlap]
        for timestep in flow_timesteps(generation.num_inference_steps):
            if overlap:
                latents = blend_overlap(
                    latents,
                    noise_prompt,
                    previous_latent,
                    overlap,
                    timestep,
                )
            time_batch = mx.broadcast_to(timestep, (latents.shape[0],))
            conditional = transformer(latents, time_batch, condition)
            unconditional = transformer(
                latents, time_batch, mx.zeros_like(condition)
            )
            velocity = unconditional + generation.flow_cfg_scale * (
                conditional - unconditional
            )
            latents = euler_step(
                latents, velocity, generation.num_inference_steps
            )
            mx.eval(latents)
        if overlap:
            latents = restore_overlap(latents, previous_latent, overlap)
        previous_latent = carry_window(latents)
        previous_condition = carry_window(condition)
        latent_chunks.append(latents)
        mx.eval(previous_latent, previous_condition, latents)

    waveforms = []
    hop_length = math.prod(vocoder.config.upsampling_ratios)
    for index, latents in enumerate(latent_chunks):
        waveform = vocoder(latents)
        crop = waveform_crop(index, len(latent_chunks), hop_length)
        end = waveform.shape[-1] - crop.right if crop.right else waveform.shape[-1]
        waveforms.append(waveform[..., crop.left:end])
    audio = mx.clip(mx.concatenate(waveforms, axis=-1).astype(mx.float32), -1.0, 1.0)
    mx.eval(audio)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(arguments.output),
        {"audio": audio},
        metadata={
            "stage": "generate",
            "seed": str(generation.seed),
            "audio_duration": str(generation.audio_duration),
            "steps": str(generation.num_inference_steps),
            "ar_cfg": str(generation.ar_cfg_scale),
            "flow_cfg": str(generation.flow_cfg_scale),
            "top_k": str(generation.top_k),
            "input_dtype": arguments.input_dtype,
            "sampling_rate": str(vocoder.config.sampling_rate),
            "num_frames": str(frame_hiddens.shape[1]),
            "num_chunks": str(len(latent_chunks)),
        },
    )
    if arguments.wav_output is not None:
        write_wav(arguments.wav_output, audio, vocoder.config.sampling_rate)
        print(f"wav={arguments.wav_output} duration={audio.shape[-1] / vocoder.config.sampling_rate}s")
    elapsed = time.perf_counter() - started_at
    peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1_048_576
    print(f"actual={arguments.output}")
    print(f"audio shape={audio.shape} dtype={audio.dtype}")
    print(f"sampling rate={vocoder.config.sampling_rate}")
    print(f"num frames={frame_hiddens.shape[1]} num chunks={len(latent_chunks)}")
    print(f"audio finite={bool(mx.all(mx.isfinite(audio)).item())}")
    print(f"audio peak={float(mx.max(mx.abs(audio)).item())}")
    print(f"elapsed seconds={elapsed:.3f}")
    print(f"peak memory MB={peak_memory_mb:.1f}")


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "decode-chunks":
        dump_decode_chunks(arguments)
    elif _REFERENCE_RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "mlx_minimax_music3 reference runtime is required for this command"
        ) from _REFERENCE_RUNTIME_IMPORT_ERROR
    elif arguments.command == "decode":
        dump_decode(arguments)
    elif arguments.command == "transformer-forward":
        dump_transformer_forward(arguments)
    elif arguments.command == "level1":
        dump_level1(arguments)
    elif arguments.command == "language-forward":
        dump_language_forward(arguments)
    elif arguments.command == "attention-probe":
        dump_attention_probe(arguments)
    elif arguments.command == "generate":
        dump_generate(arguments)
    elif arguments.command == "frame-hiddens":
        dump_frame_hiddens(arguments)
    elif arguments.command == "condition-forward":
        dump_condition_forward(arguments)
    elif arguments.command == "rvq-forward":
        dump_rvq_forward(arguments)
    elif arguments.command == "rvq-embeddings":
        dump_rvq_embeddings(arguments)
    else:
        dump_denoise(arguments)


if __name__ == "__main__":
    main()
