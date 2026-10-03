import Foundation
import GenImageCore
import MLX

public struct MiniMaxH3LoRAConfiguration: Codable, Sendable {
    public let path: String
    public let scale: Float

    public init(path: String, scale: Float) {
        self.path = path
        self.scale = scale
    }
}

public enum MiniMaxH3LoRAError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let reason): "H3 LoRA：\(reason)" }
    }
}

/// Keeps the GGUF-derived quantized base intact and evaluates low-rank residuals.
public struct MiniMaxH3LoRAAdapter {
    private struct Delta {
        let a: MLXArray
        let b: MLXArray
        let scale: Float
    }
    private struct Pair {
        var a: MLXArray?
        var b: MLXArray?
        var alpha: MLXArray?
    }
    private var deltas: [String: [Delta]] = [:]
    public var layerCount: Int { deltas.count }

    public static func validateRequest(
        loras: [MiniMaxH3LoRAConfiguration], acceleration: MiniMaxH3Acceleration?,
        steps: Int, isRef2VA: Bool = false
    ) throws {
        guard steps > 0 else { throw MiniMaxH3LoRAError.invalid("步數必須大於 0。") }
        guard let acceleration else {
            guard loras.isEmpty else { throw MiniMaxH3LoRAError.invalid("缺少加速 LoRA 的採樣設定。") }
            return
        }
        guard !isRef2VA, loras.count == 1 else {
            throw MiniMaxH3LoRAError.invalid("加速 Profile 需使用 FL2VA 基底及單一蒸餾 LoRA。")
        }
        guard acceleration.allowedSteps.contains(steps) else {
            throw MiniMaxH3LoRAError.invalid("此加速 LoRA 的步數需為 \(acceleration.allowedSteps)。")
        }
        guard loras[0].scale.isFinite, loras[0].scale > 0, loras[0].scale <= 1 else {
            throw MiniMaxH3LoRAError.invalid("加速 LoRA 權重需大於 0 且不超過 1，建議使用 1。")
        }
        guard FileManager.default.fileExists(atPath: loras[0].path) else {
            throw MiniMaxH3LoRAError.invalid("找不到權重檔案：\(loras[0].path)")
        }
    }

    public static func load(
        _ configurations: [MiniMaxH3LoRAConfiguration], shapes: [String: [Int]]
    ) throws -> Self {
        var result = Self()
        for configuration in configurations {
            try Task.checkCancellation()
            guard configuration.scale.isFinite, (0...1).contains(configuration.scale) else {
                throw MiniMaxH3LoRAError.invalid("權重需介於 0 到 1。")
            }
            guard configuration.scale != 0 else { continue }
            let tensors = try MLX.loadArrays(url: URL(fileURLWithPath: configuration.path))
            var pairs: [String: Pair] = [:]
            for (key, tensor) in tensors {
                let suffixes = [(".lora_A.weight", "a"), (".lora_B.weight", "b"),
                                (".lora_down.weight", "a"), (".lora_up.weight", "b"), (".alpha", "alpha")]
                guard let (suffix, kind) = suffixes.first(where: { key.hasSuffix($0.0) }) else {
                    throw MiniMaxH3LoRAError.invalid("不支援的權重：\(key)")
                }
                let path = normalize(String(key.dropLast(suffix.count)))
                var pair = pairs[path] ?? Pair()
                switch kind {
                case "a":
                    guard pair.a == nil else { throw MiniMaxH3LoRAError.invalid("重複權重：\(path)") }
                    pair.a = tensor
                case "b":
                    guard pair.b == nil else { throw MiniMaxH3LoRAError.invalid("重複權重：\(path)") }
                    pair.b = tensor
                default:
                    guard pair.alpha == nil else { throw MiniMaxH3LoRAError.invalid("重複 alpha：\(path)") }
                    pair.alpha = tensor
                }
                pairs[path] = pair
            }
            guard !pairs.isEmpty else { throw MiniMaxH3LoRAError.invalid("找不到 A/B 權重配對。") }
            for (path, pair) in pairs {
                try Task.checkCancellation()
                guard let a = pair.a, let b = pair.b, a.ndim == 2, b.ndim == 2,
                      a.dim(0) > 0, a.dim(0) == b.dim(1), a.dtype.isFloatingPoint, b.dtype.isFloatingPoint,
                      shapes[path] == [b.dim(0), a.dim(1)] else {
                    throw MiniMaxH3LoRAError.invalid("A/B 配對或基底形狀不相容：\(path)")
                }
                var strength = configuration.scale
                if let alpha = pair.alpha {
                    guard alpha.size == 1 else { throw MiniMaxH3LoRAError.invalid("alpha 必須為純量：\(path)") }
                    strength *= alpha.item(Float.self) / Float(a.dim(0))
                }
                guard strength.isFinite else { throw MiniMaxH3LoRAError.invalid("倍率無效：\(path)") }
                MLX.eval(a, b)
                result.deltas[path, default: []].append(Delta(a: a, b: b, scale: strength))
            }
        }
        return result
    }

    static func normalize(_ path: String) -> String {
        var path = path
        let prefixes = ["model.diffusion_model.", "diffusion_model.", "base_model.model."]
        while let prefix = prefixes.first(where: { path.hasPrefix($0) }) { path.removeFirst(prefix.count) }
        return path
    }

    static func shapes(weights: [String: MLXArray], quantizedPrefixes: Set<String>) -> [String: [Int]] {
        var result: [String: [Int]] = [:]
        for (name, tensor) in weights where name.hasSuffix(".weight") && tensor.ndim == 2 {
            let path = String(name.dropLast(".weight".count))
            let multiplier = quantizedPrefixes.contains(path) ? 32 / MiniMaxH3GGUFQuantizedLoader.targetBits : 1
            result[path] = [tensor.dim(0), tensor.dim(1) * multiplier]
        }
        return result
    }

    func apply(to output: MLXArray, input: MLXArray, path: String) -> MLXArray {
        guard let contributions = deltas[path] else { return output }
        var result = output.asType(.float32)
        for delta in contributions {
            let hidden = matmul(input.asType(.float32), delta.a.asType(.float32).T)
            result = result + matmul(hidden, delta.b.asType(.float32).T) * delta.scale
        }
        return result.asType(output.dtype)
    }
}
