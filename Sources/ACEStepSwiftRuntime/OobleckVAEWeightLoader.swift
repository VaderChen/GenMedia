import MLX
import MLXNN
import Foundation

enum OobleckVAEWeightError: LocalizedError {
    case missingCompanion(String)
    case parameterMismatch(missing: [String], unexpected: [String])

    var errorDescription: String? {
        switch self {
        case let .missingCompanion(name):
            "VAE weight normalization 缺少配對 Tensor：\(name)"
        case let .parameterMismatch(missing, unexpected):
            "VAE 權重與 Swift 模型不一致。缺少：\(missing)，多餘：\(unexpected)"
        }
    }
}

enum OobleckVAEWeightLoader {
    static func load(
        model: OobleckVAEDecoder,
        from url: URL,
        dtype: DType = .bfloat16
    ) throws {
        let source = try MLX.loadArrays(url: url)
        let converted = try convertDecoderWeights(source, dtype: dtype)
        let expectedKeys = Set(model.parameters().flattened().map(\.0))
        let convertedKeys = Set(converted.keys)
        let missing = expectedKeys.subtracting(convertedKeys).sorted()
        let unexpected = convertedKeys.subtracting(expectedKeys).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw OobleckVAEWeightError.parameterMismatch(
                missing: missing,
                unexpected: unexpected
            )
        }
        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
    }

    private static func convertDecoderWeights(
        _ source: [String: MLXArray],
        dtype: DType
    ) throws -> [String: MLXArray] {
        var converted: [String: MLXArray] = [:]
        for key in source.keys.sorted() where key.hasPrefix("decoder.") {
            if key.hasSuffix(".weight_v") {
                continue
            }
            if key.hasSuffix(".weight_g") {
                let base = String(key.dropLast(".weight_g".count))
                let vectorKey = base + ".weight_v"
                guard let scale = source[key], let vector = source[vectorKey] else {
                    throw OobleckVAEWeightError.missingCompanion(vectorKey)
                }
                converted[base + ".weight"] = fusedWeight(
                    scale: scale,
                    vector: vector,
                    isTransposedConvolution: base.contains(".conv_t1"),
                    dtype: dtype
                )
                continue
            }
            guard let value = source[key] else { continue }
            if key.hasSuffix(".alpha") || key.hasSuffix(".beta") {
                converted[key] = value.squeezed().asType(dtype)
            } else {
                converted[key] = value.asType(dtype)
            }
        }
        return converted
    }

    private static func fusedWeight(
        scale: MLXArray,
        vector: MLXArray,
        isTransposedConvolution: Bool,
        dtype: DType
    ) -> MLXArray {
        let floatScale = scale.asType(.float32)
        let floatVector = vector.asType(.float32)
        let axes = Array(1..<floatVector.ndim)
        let norm = floatVector.square().sum(axes: axes, keepDims: true).sqrt()
        let fused = floatScale * floatVector / (norm + 1e-9)
        if isTransposedConvolution {
            return fused.transposed(1, 2, 0).asType(dtype)
        }
        return fused.transposed(0, 2, 1).asType(dtype)
    }
}
