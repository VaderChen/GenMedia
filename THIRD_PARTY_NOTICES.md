# Qwen-Image 2.1 Runtime notices

The Swift implementation in `Sources/QwenImage21Runtime` follows the Qwen-Image 2.1 architecture published by Qwen and the Hugging Face Diffusers implementation. Single-frame VAE shortcuts also adapt the mflux implementation. Production inference uses Swift and MLX only; the NumPy scripts under `scripts/qwen21` generate test fixtures.

- Qwen upstream: https://github.com/QwenLM/Qwen-Image-2.1
- Diffusers: https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/qwenimage21 (Apache-2.0; see `docs/licenses/diffusers-Apache-2.0.txt`).
- Transformers Qwen3-VL: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py (Copyright 2025 The Qwen Team and The HuggingFace Inc. team, Apache-2.0). The independent vision/text numerical oracle follows this version; the Apache-2.0 license is included in `docs/licenses/diffusers-Apache-2.0.txt`.
- mflux: https://github.com/mflux-community/mflux (MIT; notice below).
- MLX Swift LM: https://github.com/ml-explore/mlx-swift-lm, pinned to 3.31.4. The patch in `Patches/Qwen3VL-Image-Conditioning.patch` extends Apple's Qwen3-VL implementation (Copyright © 2025 Apple Inc., MIT) with pre-normalization hidden-state access. The separate `Qwen3VL-Vision-GELU.patch` aligns the vision MLP with the checkpoint's `gelu_pytorch_tanh` activation.

Model weights are separately licensed under the **Qwen Research License**, as stated by the official Qwen repository. The third-party MLX conversion's metadata does not replace the upstream model license. Weights are downloaded separately and are excluded from Git.

## Diffusers attribution

Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved.

Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.

The QwenImage21 pipeline, transformer and VAE architecture have been adapted into Swift and MLX tensor operations, with a staged loader, app integration and sRGB preprocessing. These adaptations retain the upstream Apache-2.0 notice and license provided in `docs/licenses/diffusers-Apache-2.0.txt`.

## mflux

MIT License

Copyright (c) 2026 Filip Strand

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
