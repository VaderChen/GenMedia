"""Independent CPU-only NumPy oracle for Qwen3-VL image conditioning.

Equations follow Hugging Face Transformers v4.57.1's Qwen3-VL model:
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
Qwen3VLVisionModel, Qwen3VLVisionBlock, Qwen3VLVisionPatchMerger,
Qwen3VLTextRotaryEmbedding, Qwen3VLTextDecoderLayer, and _deepstack_process.
This is a small independently expressed numerical reference, not an inference
runtime. It needs NumPy only and never imports Swift, MLX, Torch, or model files.

The fixture stores MLX-native Conv3d weights (OTHWI); all other Linear weights
are OI. Expected values are computed with explicit dense attention and float32
arrays. Final text RMSNorm is intentionally excluded, as Qwen-Image 2.1 uses
the output of the final decoder layer before that normalization.
"""

import json
import math
from pathlib import Path

import numpy as np


F = np.float32
D = 8
MERGE = 2
PATCH = 2
TEMPORAL = 2
SIDE = 3
IMAGE_TOKEN = 28
RNG = np.random.default_rng(21031)
WEIGHTS = {}
CONFIG = {
    "model_type": "qwen3_vl",
    "image_token_id": IMAGE_TOKEN,
    "video_token_id": 29,
    "vision_start_token_id": 26,
    "vision_end_token_id": 27,
    "text_config": {
        "model_type": "qwen3_vl_text", "hidden_size": D,
        "intermediate_size": 16, "num_hidden_layers": 1,
        "num_attention_heads": 1, "num_key_value_heads": 1,
        "head_dim": D, "max_position_embeddings": 128,
        "vocab_size": 32, "rms_norm_eps": 1e-6,
        "rope_theta": 100.0, "tie_word_embeddings": True,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [1, 1, 2]},
    },
    "vision_config": {
        "model_type": "qwen3_vl", "depth": 1, "hidden_size": D,
        "intermediate_size": 16, "out_hidden_size": D, "num_heads": 1,
        "patch_size": PATCH, "spatial_merge_size": MERGE,
        "temporal_patch_size": TEMPORAL, "in_channels": 3,
        "num_position_embeddings": SIDE * SIDE,
        "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": [0],
    },
}


def random_tensor(shape, scale):
    return (RNG.standard_normal(shape) * scale).astype(np.float32)


def dense(name, output, inputs, bias=True, scale=0.27):
    WEIGHTS[name + ".weight"] = random_tensor((output, inputs), scale)
    if bias:
        WEIGHTS[name + ".bias"] = random_tensor((output,), 0.08)


def norm_weights(name, dimension, bias=True):
    WEIGHTS[name + ".weight"] = F(1) + random_tensor((dimension,), 0.13)
    if bias:
        WEIGHTS[name + ".bias"] = random_tensor((dimension,), 0.08)


def linear(x, name):
    return x @ WEIGHTS[name + ".weight"].T + WEIGHTS.get(name + ".bias", F(0))


def layer_norm(x, name):
    centered = x - x.mean(axis=-1, keepdims=True)
    normalized = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + F(1e-6))
    return normalized * WEIGHTS[name + ".weight"] + WEIGHTS[name + ".bias"]


def rms_norm(x, name):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + F(1e-6)) * WEIGHTS[name + ".weight"]


def gelu_tanh(x):
    return F(0.5) * x * (F(1) + np.tanh(F(math.sqrt(2 / math.pi)) * (x + F(0.044715) * x**3)))


def gelu_exact(x):
    # nn.GELU() in BOTH patch mergers is erf-based, unlike vision-block MLPs.
    erf = np.array([math.erf(float(v) / math.sqrt(2)) for v in x.flat], dtype=np.float32).reshape(x.shape)
    return F(0.5) * x * (F(1) + erf)


def gelu_sigmoid(x):
    return x / (F(1) + np.exp(F(-1.702) * x))


def rotate_half(x):
    first, second = np.split(x, 2, axis=-1)
    return np.concatenate([-second, first], axis=-1)


def rotary(x, half_angles):
    angles = np.concatenate([half_angles, half_angles], axis=-1)
    return x * np.cos(angles) + rotate_half(x) * np.sin(angles)


def attention(q, k, v, causal=False):
    scores = q @ k.T * F(D ** -0.5)
    if causal:
        scores = np.where(np.tril(np.ones(scores.shape, dtype=bool)), scores, -np.inf)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities @ v


def patch_coordinates(height, width):
    # Each contiguous group of four patches forms a 2x2 spatial merge block.
    return [(block_y + dy, block_x + dx)
            for block_y in range(0, height, MERGE)
            for block_x in range(0, width, MERGE)
            for dy in range(MERGE) for dx in range(MERGE)]


def interpolate_positions(height, width):
    table = WEIGHTS["vision_tower.pos_embed.weight"].reshape(SIDE, SIDE, D)
    points = []
    for y, x in patch_coordinates(height, width):
        # Official fast_pos_embed_interpolate aligns grid endpoints.
        fy = F(y * (SIDE - 1) / (height - 1))
        fx = F(x * (SIDE - 1) / (width - 1))
        lo_y, lo_x = int(fy), int(fx)
        hi_y, hi_x = min(lo_y + 1, SIDE - 1), min(lo_x + 1, SIDE - 1)
        dy, dx = fy - F(lo_y), fx - F(lo_x)
        points.append(table[lo_y, lo_x] * ((1 - dy) * (1 - dx))
                      + table[lo_y, hi_x] * ((1 - dy) * dx)
                      + table[hi_y, lo_x] * (dy * (1 - dx))
                      + table[hi_y, hi_x] * (dy * dx))
    return np.array(points, dtype=np.float32)


def merge_patches(x, name, postshuffle):
    if postshuffle:
        x = x.reshape(-1, D * MERGE * MERGE)
    x = layer_norm(x, name + ".norm").reshape(-1, D * MERGE * MERGE)
    return linear(gelu_exact(linear(x, name + ".linear_fc1")), name + ".linear_fc2")


def encode_vision(pixels, height, width, activation=gelu_tanh):
    # Input patches are CTHW, whereas fixture Conv3d weights are OTHWC.
    patch_kernel = WEIGHTS["vision_tower.patch_embed.proj.weight"].transpose(0, 4, 1, 2, 3)
    x = pixels @ patch_kernel.reshape(D, -1).T + WEIGHTS["vision_tower.patch_embed.proj.bias"]
    x += interpolate_positions(height, width)
    coordinates = np.array(patch_coordinates(height, width), dtype=np.float32)
    # Vision uses head_dim/2 rotary dimensions and concatenates H then W.
    frequencies = F(1) / F(10000) ** (np.arange(0, D // 2, 2, dtype=np.float32) / F(D // 2))
    angles = (coordinates[..., None] * frequencies).reshape(height * width, D // 2)
    prefix = "vision_tower.blocks.0"
    q, k, v = np.split(linear(layer_norm(x, prefix + ".norm1"), prefix + ".attn.qkv"), 3, axis=-1)
    x = x + linear(attention(rotary(q, angles), rotary(k, angles), v), prefix + ".attn.proj")
    x = x + linear(activation(linear(layer_norm(x, prefix + ".norm2"), prefix + ".mlp.linear_fc1")),
                   prefix + ".mlp.linear_fc2")
    return (merge_patches(x, "vision_tower.merger", False),
            merge_patches(x, "vision_tower.deepstack_merger_list.0", True))


def text_positions(tokens, height, width):
    start = tokens.index(IMAGE_TOKEN)
    visual_positions = [[0, y, x]
                        for y in range(height // MERGE) for x in range(width // MERGE)]
    visual_positions = np.array(visual_positions, dtype=np.int32) + start
    suffix_start = int(visual_positions.max()) + 1
    suffix_count = len(tokens) - start - len(visual_positions)
    return np.concatenate([np.repeat(np.arange(start)[:, None], 3, axis=1),
                           visual_positions,
                           np.repeat(np.arange(suffix_start, suffix_start + suffix_count)[:, None], 3, axis=1)])


def encode_text(tokens, features, deepstack, positions):
    x = WEIGHTS["language_model.model.embed_tokens.weight"][tokens].copy()
    image_mask = np.array(tokens) == IMAGE_TOKEN
    x[image_mask] = features
    prefix = "language_model.model.layers.0"
    normalized = rms_norm(x, prefix + ".input_layernorm")
    q = rms_norm(linear(normalized, prefix + ".self_attn.q_proj"), prefix + ".self_attn.q_norm")
    k = rms_norm(linear(normalized, prefix + ".self_attn.k_proj"), prefix + ".self_attn.k_norm")
    v = linear(normalized, prefix + ".self_attn.v_proj")
    # Interleaved MRoPE assigns H at indexes 1,4,... and W at 2,5,...
    # within each section's extent. Remaining dimensions keep T.
    inv_freq = F(1) / F(CONFIG["text_config"]["rope_theta"]) ** (np.arange(0, D, 2, dtype=np.float32) / F(D))
    all_angles = positions.astype(np.float32)[:, :, None] * inv_freq
    angles = all_angles[:, 0].copy()
    section = CONFIG["text_config"]["rope_scaling"]["mrope_section"]
    for axis in (1, 2):
        for index in range(axis, min(section[axis] * 3, D // 2), 3):
            angles[:, index] = all_angles[:, axis, index]
    x += linear(attention(rotary(q, angles), rotary(k, angles), v, causal=True), prefix + ".self_attn.o_proj")
    normalized = rms_norm(x, prefix + ".post_attention_layernorm")
    gate = linear(normalized, prefix + ".mlp.gate_proj")
    x += linear(gate / (F(1) + np.exp(-gate)) * linear(normalized, prefix + ".mlp.up_proj"), prefix + ".mlp.down_proj")
    # Deepstack adds ONLY at image-token positions, after this decoder layer.
    x[image_mask] += deepstack
    return x[None]


def tensor(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "values": value.flatten().tolist()}


def build_weights():
    WEIGHTS["vision_tower.patch_embed.proj.weight"] = random_tensor((D, TEMPORAL, PATCH, PATCH, 3), 0.19)
    WEIGHTS["vision_tower.patch_embed.proj.bias"] = random_tensor((D,), 0.08)
    WEIGHTS["vision_tower.pos_embed.weight"] = random_tensor((SIDE * SIDE, D), 0.31)
    prefix = "vision_tower.blocks.0"
    norm_weights(prefix + ".norm1", D)
    norm_weights(prefix + ".norm2", D)
    dense(prefix + ".attn.qkv", 3 * D, D)
    dense(prefix + ".attn.proj", D, D)
    dense(prefix + ".mlp.linear_fc1", 16, D, scale=0.4)
    dense(prefix + ".mlp.linear_fc2", D, 16)
    for name, postshuffle in (("merger", False), ("deepstack_merger_list.0", True)):
        prefix = "vision_tower." + name
        norm_weights(prefix + ".norm", D * MERGE**2 if postshuffle else D)
        dense(prefix + ".linear_fc1", D * MERGE**2, D * MERGE**2, scale=0.18)
        dense(prefix + ".linear_fc2", D, D * MERGE**2, scale=0.18)
    WEIGHTS["language_model.model.embed_tokens.weight"] = random_tensor((32, D), 0.5)
    norm_weights("language_model.model.norm", D, bias=False)
    # A nontrivial final norm ensures accidentally returning normalized states fails.
    WEIGHTS["language_model.model.norm.weight"] *= F(2.7)
    prefix = "language_model.model.layers.0"
    for name in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
        norm_weights(prefix + "." + name, D, bias=False)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        dense(prefix + ".self_attn." + name, D, D, bias=False)
    dense(prefix + ".mlp.gate_proj", 16, D, bias=False)
    dense(prefix + ".mlp.up_proj", 16, D, bias=False)
    dense(prefix + ".mlp.down_proj", D, 16, bias=False)


def build_case(height, width):
    tokens = [1, 5, 26, IMAGE_TOKEN, IMAGE_TOKEN, 27, 6, 9]
    pixels = random_tensor((height * width, 3 * TEMPORAL * PATCH**2), 0.7)
    positions = text_positions(tokens, height, width)
    features, deepstack = encode_vision(pixels, height, width)
    expected = encode_text(tokens, features, deepstack, positions)
    wrong_features, wrong_deepstack = encode_vision(pixels, height, width, gelu_sigmoid)
    wrong_activation = encode_text(tokens, wrong_features, wrong_deepstack, positions)
    no_deepstack = encode_text(tokens, features, np.zeros_like(deepstack), positions)
    wrong_positions = np.repeat(np.arange(len(tokens))[:, None], 3, axis=1)
    one_axis = encode_text(tokens, features, deepstack, wrong_positions)
    sensitivity = {
        "sigmoidGELUMaxError": float(np.abs(expected - wrong_activation).max()),
        "missingDeepstackMaxError": float(np.abs(expected - no_deepstack).max()),
        "oneDimensionalRoPEMaxError": float(np.abs(expected - one_axis).max()),
    }
    assert min(sensitivity.values()) > 0.001, sensitivity
    return {"name": f"{height}x{width}-patch-image", "grid": [1, height, width],
            "tokens": tokens, "pixels": tensor(pixels), "positionIDs": positions.T.tolist(),
            "visionFeatures": tensor(features), "deepstackFeatures": tensor(deepstack),
            "expected": tensor(expected), "sensitivity": sensitivity}


def main():
    build_weights()
    cases = [build_case(2, 4), build_case(4, 2)]
    fixture = {
        "source": "huggingface/transformers v4.57.1 modeling_qwen3_vl.py",
        "configuration": json.dumps(CONFIG, separators=(",", ":")),
        "weights": {key: tensor(value) for key, value in WEIGHTS.items()},
        "cases": cases,
    }
    path = Path(__file__).resolve().parents[2] / "Tests/QwenImage21RuntimeTests/Fixtures/vision-oracle.json"
    path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    for case in cases:
        print(case["name"], case["sensitivity"])
    print(f"Wrote {len(WEIGHTS)} tensors to {path}")


if __name__ == "__main__":
    main()
