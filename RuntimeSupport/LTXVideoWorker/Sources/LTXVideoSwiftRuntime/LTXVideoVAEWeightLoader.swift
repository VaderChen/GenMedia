import Foundation
import MLX
import MLXNN

public struct LTXVideoVAEWeightLoadReport: Sendable {
    public let tensorCount: Int
    public let quantizedModuleCount: Int
    public let sourceDTypes: [String]
    public let computeDType: LTXVideoComputeDType

    public init(
        tensorCount: Int,
        quantizedModuleCount: Int,
        sourceDTypes: [String],
        computeDType: LTXVideoComputeDType
    ) {
        self.tensorCount = tensorCount
        self.quantizedModuleCount = quantizedModuleCount
        self.sourceDTypes = sourceDTypes
        self.computeDType = computeDType
    }
}

public enum LTXVideoVAEWeightLoader {
    public static func loadDecoder(
        model: LTXVideoVAEDecoder,
        from modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> LTXVideoVAEWeightLoadReport {
        let weightsURL = modelDirectory.appendingPathComponent("vae_decoder.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXVideoRuntimeError.missingFile(weightsURL)
        }
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        let source = try MLX.loadArrays(url: weightsURL)
        var converted: [String: MLXArray] = [:]
        var sourceDTypes: Set<String> = []
        for (sourceKey, sourceValue) in source where sourceKey.hasPrefix("vae_decoder.") {
            let key = String(sourceKey.dropFirst("vae_decoder.".count))
            guard let expectedValue = expected[key] else { continue }
            guard converted[key] == nil else {
                throw LTXVideoRuntimeError.duplicateWeight(key)
            }
            guard sourceValue.shape == expectedValue.shape else {
                throw LTXVideoRuntimeError.weightShapeMismatch(
                    name: key,
                    expected: expectedValue.shape,
                    actual: sourceValue.shape
                )
            }
            sourceDTypes.insert(String(describing: sourceValue.dtype))
            converted[key] = sourceValue.asType(computeDType.mlxDType)
        }

        let expectedKeys = Set(expected.keys)
        let convertedKeys = Set(converted.keys)
        let missing = expectedKeys.subtracting(convertedKeys).sorted()
        guard missing.isEmpty else {
            throw LTXVideoRuntimeError.missingWeights(missing)
        }
        let unexpected = convertedKeys.subtracting(expectedKeys).sorted()
        guard unexpected.isEmpty else {
            throw LTXVideoRuntimeError.unexpectedWeights(unexpected)
        }

        try model.update(parameters: ModuleParameters.unflattened(converted), verify: .all)
        MLX.eval(model)
        return LTXVideoVAEWeightLoadReport(
            tensorCount: converted.count,
            quantizedModuleCount: 0,
            sourceDTypes: sourceDTypes.sorted(),
            computeDType: computeDType
        )
    }
}
