import Foundation
import MLX

/// Evaluates low-rank residuals separately: merging into BF16 or packed INT4 loses small updates.
struct Qwen21LoRAAdapter {
    struct Pair {
        let a: MLXArray
        let b: MLXArray
        let scale: Float
    }
    let pairs: [String: Pair]

    init(url: URL, base: Qwen21Weights, scale: Double = 1) throws {
        try self.init(tensors: MLX.loadArrays(url: url), base: base, scale: scale)
    }

    init(tensors: [String: MLXArray], base: Qwen21Weights, scale: Double = 1) throws {
        guard scale.isFinite, Float(scale).isFinite, !tensors.isEmpty else {
            throw Qwen21Error.invalid("LoRA 權重為空或倍率無效。")
        }
        var grouped: [String: [String: MLXArray]] = [:]
        for (key, tensor) in tensors {
            let suffixes = [(".lora_A.weight", "a"), (".lora_B.weight", "b"), (".alpha", "alpha")]
            guard let (suffix, component) = suffixes.first(where: { key.hasSuffix($0.0) }) else {
                throw Qwen21Error.invalid("不支援的 LoRA 權重：\(key)")
            }
            let layer = Self.normalized(String(key.dropLast(suffix.count)))
            guard grouped[layer]?[component] == nil else {
                throw Qwen21Error.invalid("LoRA 層重複：\(key)")
            }
            grouped[layer, default: [:]][component] = tensor
        }
        var result: [String: Pair] = [:]
        for (layer, components) in grouped {
            guard let a = components["a"], let b = components["b"],
                  a.ndim == 2, b.ndim == 2, a.dim(0) > 0, a.dim(0) == b.dim(1),
                  a.dtype.isFloatingPoint, b.dtype.isFloatingPoint,
                  let weight = base.tensors[layer + ".weight"], weight.ndim == 2 else {
                throw Qwen21Error.invalid("LoRA 配對或基底層不符：\(layer)")
            }
            let packed = base.tensors[layer + ".scales"] != nil
            guard !packed || (weight.dtype == .uint32 && base.bits > 0 && 32 % base.bits == 0),
                  a.dim(1) == weight.dim(1) * (packed ? 32 / base.bits : 1),
                  b.dim(0) == weight.dim(0) else {
                throw Qwen21Error.invalid("LoRA 與基底層尺寸不符：\(layer)")
            }
            var strength = Float(scale)
            if let alpha = components["alpha"] {
                guard alpha.size == 1 else { throw Qwen21Error.invalid("LoRA alpha 必須為純量。") }
                strength *= alpha.item(Float.self) / Float(a.dim(0))
            }
            guard strength.isFinite else { throw Qwen21Error.invalid("LoRA alpha 無效。") }
            result[layer] = Pair(a: a, b: b, scale: strength)
        }
        pairs = result
    }

    static func normalized(_ raw: String) -> String {
        var name = raw
        if name.hasPrefix("transformer.") { name.removeFirst("transformer.".count) }
        name = name.replacingOccurrences(of: "time_text_embed.timestep_embedder.", with: "time_text_embed.")
        if name == "modulation.1" { name = "modulation.0" }
        return name
    }

    func delta(_ x: MLXArray, layer: String) -> MLXArray? {
        guard let pair = pairs[layer] else { return nil }
        return matmul(matmul(x.asType(.float32), pair.a.T.asType(.float32)),
                      pair.b.T.asType(.float32)) * pair.scale
    }
}
