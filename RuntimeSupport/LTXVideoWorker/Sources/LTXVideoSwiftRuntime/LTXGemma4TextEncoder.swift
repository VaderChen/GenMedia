// LTX-2.5's tuned Gemma4 text tower. Architecture follows MLX-LM's Gemma4
// (MIT) and Lightricks' 49-state encoder convention; see THIRD_PARTY_NOTICES.md.
import Foundation
import MLX
import MLXNN

public final class LTXGemma4TextEncoder {
    struct Configuration: Decodable {
        struct Text: Decodable {
            struct Rope: Decodable { let rope_theta: Float; let partial_rotary_factor: Float? }
            let hidden_size: Int, num_hidden_layers: Int, num_attention_heads: Int
            let intermediate_size: Int, head_dim: Int, global_head_dim: Int
            let num_key_value_heads: Int, num_global_key_value_heads: Int
            let vocab_size: Int, hidden_size_per_layer_input: Int, num_kv_shared_layers: Int
            let rms_norm_eps: Float
            let attention_k_eq_v: Bool, enable_moe_block: Bool, use_double_wide_mlp: Bool
            let layer_types: [String]
            let rope_parameters: [String: Rope]
        }
        struct Quantization: Decodable { let bits: Int; let group_size: Int; let mode: String? }
        let text_config: Text
        let quantization: Quantization?
    }
    private let config: Configuration.Text
    private let weights: [String: MLXArray]
    private let bits: Int
    private let groupSize: Int

    public convenience init(directory: URL) throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(Configuration.self, from: data)
        let tensors = try MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        try self.init(configuration: configuration, tensors: tensors)
    }

    init(configuration: Configuration, tensors: [String: MLXArray]) throws {
        config = configuration.text_config
        bits = configuration.quantization?.bits ?? 4
        groupSize = configuration.quantization?.group_size ?? 64
        guard config.hidden_size > 0, config.num_hidden_layers > 0,
              config.hidden_size_per_layer_input == 0, config.num_kv_shared_layers == 0,
              !config.enable_moe_block, !config.use_double_wide_mlp,
              config.layer_types.count == config.num_hidden_layers,
              config.rms_norm_eps > 0, config.rms_norm_eps.isFinite,
              [4, 8].contains(bits), groupSize > 0,
              configuration.quantization?.mode == nil || configuration.quantization?.mode == "affine" else {
            throw Self.invalid("需要 LTX 專用的 Gemma4 文字編碼器。")
        }
        var normalized: [String: MLXArray] = [:]
        for (key, tensor) in tensors {
            let prefix = "model.language_model."
            guard key.hasPrefix(prefix) else { continue } // The unused vision/audio towers are excluded.
            let name = String(key.dropFirst(prefix.count))
            guard normalized[name] == nil else { throw Self.invalid("重複權重：\(name)") }
            normalized[name] = tensor
        }
        weights = normalized
        try validateMatrix("embed_tokens", output: config.vocab_size, input: config.hidden_size)
        try validateVector("norm.weight", count: config.hidden_size)
        for i in 0..<config.num_hidden_layers {
            let p = "layers.\(i)", sliding = config.layer_types[i] == "sliding_attention"
            guard sliding || config.layer_types[i] == "full_attention",
                  let rope = config.rope_parameters[config.layer_types[i]], rope.rope_theta > 1 else {
                throw Self.invalid("不支援的 Gemma4 attention／RoPE。")
            }
            let dim = sliding ? config.head_dim : config.global_head_dim
            let kv = !sliding && config.attention_k_eq_v ? config.num_global_key_value_heads : config.num_key_value_heads
            let fraction = rope.partial_rotary_factor ?? 1
            guard dim > 0, dim % 2 == 0, kv > 0, config.num_attention_heads > 0,
                  config.num_attention_heads % kv == 0, fraction > 0, fraction <= 1 else {
                throw Self.invalid("Gemma4 attention 維度無效。")
            }
            for name in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                try validateVector(p + "." + name + ".weight", count: config.hidden_size)
            }
            try validateVector(p + ".layer_scalar", count: 1)
            for name in ["q_norm", "k_norm"] { try validateVector(p + ".self_attn." + name + ".weight", count: dim) }
            try validateMatrix(p + ".self_attn.q_proj", output: config.num_attention_heads * dim, input: config.hidden_size)
            try validateMatrix(p + ".self_attn.k_proj", output: kv * dim, input: config.hidden_size)
            if sliding || !config.attention_k_eq_v {
                try validateMatrix(p + ".self_attn.v_proj", output: kv * dim, input: config.hidden_size)
            }
            try validateMatrix(p + ".self_attn.o_proj", output: config.hidden_size, input: config.num_attention_heads * dim)
            for name in ["gate_proj", "up_proj"] {
                try validateMatrix(p + ".mlp." + name, output: config.intermediate_size, input: config.hidden_size)
            }
            try validateMatrix(p + ".mlp.down_proj", output: config.hidden_size, input: config.intermediate_size)
        }
    }

    public static func promptLayout(tokenIDs: [Int], maxLength: Int = 1024, bos: Int = 2, pad: Int = 0) throws -> LTXGemmaPromptLayout {
        guard maxLength > 0 else { throw Self.invalid("文字長度必須為正數。") }
        var tokens = Array(tokenIDs.prefix(maxLength))
        if tokens.first != bos { tokens = Array(([bos] + tokens).prefix(maxLength)) }
        return try LTXGemmaFeaturePreparation.leftPad(tokenIDs: tokens, maxLength: maxLength, padTokenID: pad)
    }

    public func allHiddenStates(tokenIDs: MLXArray, attentionMask: MLXArray) throws -> [MLXArray] {
        guard tokenIDs.ndim == 2, tokenIDs.shape.allSatisfy({ $0 > 0 }), attentionMask.shape == tokenIDs.shape,
              min(tokenIDs).item(Int.self) >= 0, max(tokenIDs).item(Int.self) < config.vocab_size else {
            throw Self.invalid("Token 或 padding mask 不符。")
        }
        let batch = tokenIDs.dim(0), count = tokenIDs.dim(1)
        let ids = tokenIDs.flattened()
        let embedding = try tensor("embed_tokens.weight")[ids]
        var hidden: MLXArray
        if let scales = weights["embed_tokens.scales"], let biases = weights["embed_tokens.biases"] {
            hidden = dequantized(embedding, scales: scales[ids], biases: biases[ids], groupSize: groupSize, bits: bits)
        } else { hidden = embedding }
        hidden = hidden.reshaped([batch, count, config.hidden_size]) * sqrt(Float(config.hidden_size))
        // LTX supplies a uniform causal + padding mask to every layer, including sliding layers.
        let causal = triu(MLXArray.full([count, count], values: MLXArray(Float(-1e9)), dtype: hidden.dtype), k: 1)
        let mask = causal[.newAxis, .newAxis, 0..., 0...]
            + (1 - attentionMask[0..., .newAxis, .newAxis, 0...].asType(hidden.dtype)) * -1e9
        // Only two position tables are needed by all Q/K projections. Keep them scoped
        // to this prompt so a later request cannot reuse tables for a different length.
        var rotations: [String: PreparedRotary] = [:]
        for type in Set(config.layer_types) {
            let rope = config.rope_parameters[type]!
            rotations[type] = PreparedRotary(count: count,
                dimension: type == "sliding_attention" ? config.head_dim : config.global_head_dim,
                base: rope.rope_theta, fraction: rope.partial_rotary_factor ?? 1)
        }
        var states = [hidden]
        states.reserveCapacity(config.num_hidden_layers + 1)
        for i in 0..<config.num_hidden_layers {
            try Task.checkCancellation()
            let p = "layers.\(i)", sliding = config.layer_types[i] == "sliding_attention"
            let dim = sliding ? config.head_dim : config.global_head_dim
            let kv = !sliding && config.attention_k_eq_v ? config.num_global_key_value_heads : config.num_key_value_heads
            let rotation = rotations[config.layer_types[i]]!
            let input = try norm(hidden, p + ".input_layernorm")
            let q = try norm(linear(input, p + ".self_attn.q_proj").reshaped([batch, count, config.num_attention_heads, dim]), p + ".self_attn.q_norm")
            let kRaw = try linear(input, p + ".self_attn.k_proj").reshaped([batch, count, kv, dim])
            let k = try norm(kRaw, p + ".self_attn.k_norm")
            let vRaw = !sliding && config.attention_k_eq_v ? kRaw : try linear(input, p + ".self_attn.v_proj").reshaped([batch, count, kv, dim])
            let v = MLXFast.rmsNorm(vRaw, weight: MLXArray.mlxNone, eps: config.rms_norm_eps).transposed(0, 2, 1, 3)
            let attention = MLXFast.scaledDotProductAttention(
                queries: rotation(q.transposed(0, 2, 1, 3)),
                keys: rotation(k.transposed(0, 2, 1, 3)),
                values: v, scale: 1, mask: .array(mask)).transposed(0, 2, 1, 3).reshaped([batch, count, -1])
            hidden = hidden + (try norm(linear(attention, p + ".self_attn.o_proj"), p + ".post_attention_layernorm"))
            let ff = try norm(hidden, p + ".pre_feedforward_layernorm")
            let mlp = try linear(geluApproximate(linear(ff, p + ".mlp.gate_proj")) * linear(ff, p + ".mlp.up_proj"), p + ".mlp.down_proj")
            hidden = (hidden + (try norm(mlp, p + ".post_feedforward_layernorm"))) * (try tensor(p + ".layer_scalar"))
            eval(hidden)
            states.append(hidden)
        }
        // HF Gemma4 exposes final norm on the last tapped state; Gemma3 does not.
        states[states.count - 1] = try norm(states[states.count - 1], "norm")
        eval(states[states.count - 1])
        return states
    }

    struct PreparedRotary {
        private let dim: Int, half: Int, rotated: Int
        private let c: MLXArray, s: MLXArray

        init(count: Int, dimension: Int, base: Float, fraction: Float) {
            dim = dimension; half = dimension / 2; rotated = Int(Float(half) * fraction)
            let frequencies = pow(base, -MLXArray(0..<rotated).asType(.float32) * 2 / Float(dim))
            let angles = MLXArray(0..<count).asType(.float32).expandedDimensions(axis: -1) * frequencies
            c = cos(angles); s = sin(angles)
            eval(c, s)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let a = x[.ellipsis, 0..<rotated].asType(.float32), b = x[.ellipsis, half..<(half + rotated)].asType(.float32)
            return concatenated([a * c - b * s, x[.ellipsis, rotated..<half].asType(.float32),
                                 b * c + a * s, x[.ellipsis, (half + rotated)..<dim].asType(.float32)], axis: -1).asType(x.dtype)
        }
    }

    private func norm(_ x: MLXArray, _ name: String) throws -> MLXArray {
        MLXFast.rmsNorm(x, weight: try tensor(name + ".weight"), eps: config.rms_norm_eps)
    }
    private func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let weight = try tensor(name + ".weight")
        if let scales = weights[name + ".scales"], let biases = weights[name + ".biases"] {
            return quantizedMatmul(x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
        }
        return matmul(x, weight.T.asType(x.dtype))
    }
    private func tensor(_ name: String) throws -> MLXArray {
        guard let value = weights[name] else { throw Self.invalid("缺少權重：\(name)") }
        return value
    }
    private func validateVector(_ name: String, count: Int) throws {
        let value = try tensor(name)
        guard value.shape == [count], value.dtype.isFloatingPoint else { throw Self.invalid("權重尺寸不符：\(name)") }
    }
    private func validateMatrix(_ name: String, output: Int, input: Int) throws {
        let weight = try tensor(name + ".weight")
        if let scales = weights[name + ".scales"] {
            guard input % groupSize == 0, weight.dtype == .uint32, weight.shape == [output, input / (32 / bits)],
                  scales.shape == [output, input / groupSize],
                  weights[name + ".biases"]?.shape == scales.shape else { throw Self.invalid("量化層不符：\(name)") }
        } else if weight.shape != [output, input] || !weight.dtype.isFloatingPoint {
            throw Self.invalid("線性層不符：\(name)")
        }
    }
    private static func invalid(_ message: String) -> LTXVideoRuntimeError {
        .invalidConfiguration("LTX Gemma4：\(message)")
    }
}
