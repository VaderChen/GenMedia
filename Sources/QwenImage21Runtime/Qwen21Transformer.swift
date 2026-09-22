// Qwen-Image 2.1 single-stream DiT. See THIRD_PARTY_NOTICES.md for upstream references.
import Foundation
import MLX
import MLXFast
import MLXNN

final class Qwen21Transformer {
    let config: Qwen21TransformerConfiguration
    let weights: Qwen21Weights

    init(directory: URL) throws {
        config = try readJSON(Qwen21TransformerConfiguration.self, at: directory.appendingPathComponent("config.json"))
        guard config.in_channels == 64, config.out_channels == 64,
              config.axes_dims_rope.count == 3,
              config.axes_dims_rope.reduce(0, +) == config.attention_head_dim,
              config.quantization.mode == "affine", config.quantization.bits == 4 else {
            throw Qwen21Error.invalid("不相容的 Transformer 設定，請使用指定的 MLX 4-bit 權重。")
        }
        weights = try Qwen21Weights(directory: directory, groupSize: config.quantization.group_size,
                                   bits: config.quantization.bits)
    }

    func callAsFunction(latents: MLXArray, conditioning: MLXArray, sigma: Float,
                        layout: Qwen21Layout) throws -> MLXArray {
        let d = config.hidden, heads = config.num_attention_heads, hd = config.attention_head_dim
        let eps = config.eps
        let normalizedText = Self.zeroCenteredRMSNorm(conditioning,
            weight: try weights.tensor("txt_in.text_norm.weight"), eps: eps)
        let text = try weights.linear(geluApproximate(weights.linear(normalizedText, "txt_in.in_layer")), "txt_in.out_layer")
        let image = try weights.linear(latents, "img_in")
        var hidden = MLXArray.zeros([1, layout.count, d], dtype: .bfloat16)
        var sourceText = 0
        for segment in layout.segments where !segment.image {
            let indices = MLXArray(Array(layout.textIndices[sourceText..<(sourceText + segment.range.count)]))
            hidden[0..., segment.range, 0...] = text[0..., indices, 0...]
            sourceText += segment.range.count
        }
        hidden[0..., MLXArray(layout.imageIndices), 0...] = image
        let freq = exp(-log(Float(10_000)) * MLXArray(0..<128).asType(.float32) / 128)
        let time = MLXArray([sigma * 1000, 0]).expandedDimensions(axis: -1) * freq
        let embedding = concatenated([cos(time), sin(time)], axis: -1).asType(.bfloat16)
        let temb = try weights.linear(silu(weights.linear(embedding,
            "time_text_embed.linear_1")), "time_text_embed.linear_2")
        // The MLX conversion numbers its sole Sequential linear as 0 (Diffusers uses 1).
        let modulation = try weights.linear(silu(temb), "modulation.0")
        let chunks = split(modulation, parts: 4, axis: -1)
        func rows(_ x: MLXArray) -> MLXArray {
            concatenated([
                broadcast(x[1..<2, .newAxis, 0...], to: [1, layout.targetStart, d]),
                broadcast(x[0..<1, .newAxis, 0...], to: [1, layout.count - layout.targetStart, d])
            ], axis: 1)
        }
        let scale1 = rows(chunks[0]), gate1 = tanh(rows(chunks[1]))
        let scale2 = rows(chunks[2]), gate2 = tanh(rows(chunks[3]))
        let (cosine, sine) = layout.rotary(axes: config.axes_dims_rope)
        func rope(_ x: MLXArray) -> MLXArray {
            let pairs = x.asType(.float32).reshaped([1, heads, layout.count, hd / 2, 2])
            let a = pairs[0..., 0..., 0..., 0..., 0], b = pairs[0..., 0..., 0..., 0..., 1]
            return stacked([a * cosine - b * sine, a * sine + b * cosine], axis: -1)
                .reshaped(x.shape).asType(x.dtype)
        }
        for i in 0..<config.num_layers {
            try Task.checkCancellation()
            let p = "transformer_blocks.\(i)"
            let n = MLXFast.layerNorm(hidden, weight: nil, bias: nil, eps: eps) * (1 + scale1)
            func projection(_ name: String) throws -> MLXArray {
                try weights.linear(n, p + ".attn." + name)
                    .reshaped([1, layout.count, heads, hd]).transposed(0, 2, 1, 3)
            }
            let q = try rope(MLXFast.rmsNorm(projection("to_q"), weight: weights.tensor(p + ".attn.norm_q.weight"), eps: eps))
            let k = try rope(MLXFast.rmsNorm(projection("to_k"), weight: weights.tensor(p + ".attn.norm_k.weight"), eps: eps))
            let v = try projection("to_v")
            var attended: [MLXArray] = []
            for segment in layout.segments {
                let end = segment.range.upperBound
                let query = q[0..., 0..., segment.range, 0...]
                let key = k[0..., 0..., ..<end, 0...], value = v[0..., 0..., ..<end, 0...]
                // MLX's causal mask is bottom-right aligned, including preceding image blocks.
                attended.append(MLXFast.scaledDotProductAttention(queries: query, keys: key, values: value,
                    scale: 1 / sqrt(Float(hd)), mask: segment.image ? .none : .causal))
            }
            let attention = concatenated(attended, axis: 2).transposed(0, 2, 1, 3).reshaped([1, layout.count, d])
            hidden = hidden + gate1 * (try weights.linear(attention, p + ".attn.to_out.0"))
            let ff = MLXFast.layerNorm(hidden, weight: nil, bias: nil, eps: eps) * (1 + scale2)
            let feed = try silu(weights.linear(ff, p + ".img_mlp.gate_layer")) * weights.linear(ff, p + ".img_mlp.proj")
            hidden = hidden + gate2 * (try weights.linear(feed, p + ".img_mlp.out"))
            eval(hidden)
        }
        // Only the target image is returned; the reference latents remain fixed for every step.
        let target = hidden[0..., layout.targetStart..., 0...]
        let finalScale = try weights.linear(silu(temb[0..<1]), "norm_out.linear")[0..., .newAxis, 0...]
        return try weights.linear(MLXFast.layerNorm(target, weight: nil, bias: nil, eps: eps) * (1 + finalScale), "proj_out")
    }

    static func zeroCenteredRMSNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        // The checkpoint stores scale - 1; adding 1 in BF16 rounds the effective scale too early.
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32) + 1, eps: eps)
            .asType(x.dtype)
    }
}
