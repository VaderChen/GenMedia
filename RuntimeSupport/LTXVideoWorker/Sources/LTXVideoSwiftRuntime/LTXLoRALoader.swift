import Foundation
import MLX
import MLXNN

public struct LTXLoRAConfiguration: Sendable {
    public let url: URL
    public let scale: Float

    public init(url: URL, scale: Float) {
        self.url = url
        self.scale = scale
    }
}

public enum LTXLoRAError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let reason): "LTX LoRA：\(reason)" }
    }
}

/// Adds low-rank residuals without dequantizing or modifying the base model weights.
public enum LTXLoRALoader {
    @discardableResult
    public static func apply(_ configurations: [LTXLoRAConfiguration], to model: Module) throws -> Int {
        guard !configurations.isEmpty else { return 0 }
        let layers = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        var contributions: [String: [LTXLoRADelta]] = [:]
        for configuration in configurations {
            try Task.checkCancellation()
            guard configuration.scale.isFinite, (0...1).contains(configuration.scale) else {
                throw LTXLoRAError.invalid("權重必須介於 0 到 1。")
            }
            guard configuration.scale != 0 else { continue }
            let tensors = try MLX.loadArrays(url: configuration.url)
            let pairs = try parse(tensors)
            guard !pairs.isEmpty else { throw LTXLoRAError.invalid("找不到 A/B 權重配對。") }
            for path in pairs.keys.sorted() {
                try Task.checkCancellation()
                let pair = pairs[path]!
                guard let a = pair.a, let b = pair.b, a.ndim == 2, b.ndim == 2,
                      a.dim(0) > 0, b.dim(1) == a.dim(0),
                      a.dtype.isFloatingPoint, b.dtype.isFloatingPoint else {
                    throw LTXLoRAError.invalid("權重配對不完整或形狀無效：\(path)")
                }
                guard let linear = layers[path] as? Linear,
                      linear.shape.0 == b.dim(0), linear.shape.1 == a.dim(1) else {
                    throw LTXLoRAError.invalid("權重與目前 Transformer 不相容：\(path)")
                }
                var strength = configuration.scale
                if let alpha = pair.alpha {
                    guard alpha.size == 1 else { throw LTXLoRAError.invalid("alpha 不是純量：\(path)") }
                    let value = alpha.item(Float.self)
                    guard value.isFinite else { throw LTXLoRAError.invalid("alpha 無效：\(path)") }
                    strength *= value / Float(a.dim(0))
                }
                guard strength.isFinite else { throw LTXLoRAError.invalid("權重倍率溢位：\(path)") }
                eval(a, b)
                contributions[path, default: []].append(LTXLoRADelta(a: a, b: b, scale: strength))
            }
        }
        // Validate every adapter before replacing anything, so failures leave the model intact.
        try Task.checkCancellation()
        let replacements: [(String, Module)] = contributions.keys.sorted().map { path in
            (path, LTXLoRALinear(base: layers[path] as! Linear, deltas: contributions[path]!))
        }
        guard !replacements.isEmpty else { return 0 }
        try model.update(modules: ModuleChildren.unflattened(replacements), verify: .noUnusedKeys)
        return replacements.count
    }

    private struct Pair {
        var a: MLXArray?
        var b: MLXArray?
        var alpha: MLXArray?
    }

    private static func parse(_ tensors: [String: MLXArray]) throws -> [String: Pair] {
        var pairs: [String: Pair] = [:]
        let suffixes = [(".lora_A.weight", "a"), (".lora_B.weight", "b"),
                        (".lora_down.weight", "a"), (".lora_up.weight", "b"),
                        (".lora_A", "a"), (".lora_B", "b"), (".alpha", "alpha")]
        for key in tensors.keys.sorted() {
            guard let (suffix, kind) = suffixes.first(where: { key.hasSuffix($0.0) }) else {
                throw LTXLoRAError.invalid("不支援的權重：\(key)；目前接受一般 Linear LoRA。")
            }
            let path = normalize(String(key.dropLast(suffix.count)))
            var pair = pairs[path] ?? Pair()
            switch kind {
            case "a":
                guard pair.a == nil else { throw LTXLoRAError.invalid("重複權重：\(path)") }
                pair.a = tensors[key]
            case "b":
                guard pair.b == nil else { throw LTXLoRAError.invalid("重複權重：\(path)") }
                pair.b = tensors[key]
            default:
                guard pair.alpha == nil else { throw LTXLoRAError.invalid("重複 alpha：\(path)") }
                pair.alpha = tensors[key]
            }
            pairs[path] = pair
        }
        return pairs
    }

    static func normalize(_ key: String) -> String {
        var result = key
        let prefixes = ["model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer."]
        while let prefix = prefixes.first(where: { result.hasPrefix($0) }) {
            result.removeFirst(prefix.count)
        }
        return result.replacingOccurrences(of: ".to_out.0", with: ".to_out")
            .replacingOccurrences(of: ".ff.net.0.proj", with: ".ff.proj_in")
            .replacingOccurrences(of: ".ff.net.2", with: ".ff.proj_out")
            .replacingOccurrences(of: ".audio_ff.net.0.proj", with: ".audio_ff.proj_in")
            .replacingOccurrences(of: ".audio_ff.net.2", with: ".audio_ff.proj_out")
    }
}

private struct LTXLoRADelta {
    let a: MLXArray
    let b: MLXArray
    let scale: Float
}

private final class LTXLoRALinear: Linear {
    @ModuleInfo var base: Linear
    private let deltas: [LTXLoRADelta]
    override var shape: (Int, Int) { base.shape }

    init(base: Linear, deltas: [LTXLoRADelta]) {
        self._base = ModuleInfo(wrappedValue: base)
        self.deltas = deltas
        // These are references to existing arrays, including packed INT4/INT8 weights.
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        var result = base(input)
        for delta in deltas {
            let lowRank = matmul(input.asType(.float32), delta.a.asType(.float32).T)
            let update = matmul(lowRank, delta.b.asType(.float32).T) * delta.scale
            result = (result.asType(.float32) + update).asType(result.dtype)
        }
        return result
    }
}
