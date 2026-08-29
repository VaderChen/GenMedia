import Foundation
import MLX
import MLXNN

public struct LTXVocoderWeightLoadReport: Sendable {
    public let tensorCount: Int
    public let sourceDTypes: [String]
    public let computeDType: LTXVideoComputeDType

    public init(tensorCount: Int, sourceDTypes: [String], computeDType: LTXVideoComputeDType) {
        self.tensorCount = tensorCount
        self.sourceDTypes = sourceDTypes
        self.computeDType = computeDType
    }
}

public enum LTXVocoderWeightLoader {
    public static func load(
        model: LTXVocoderWithBWE,
        from modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .float32
    ) throws -> LTXVocoderWeightLoadReport {
        let weightsURL = firstExisting(
            in: modelDirectory,
            paths: [
                "vocoder.safetensors",
                "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"
            ]
        ) ?? modelDirectory.appendingPathComponent("vocoder.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXVideoRuntimeError.missingFile(weightsURL)
        }
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) })
        let source = try MLX.loadArrays(url: weightsURL)
        var converted: [String: MLXArray] = [:]
        var sourceDTypes: Set<String> = []
        for (sourceKey, sourceValue) in source where sourceKey.hasPrefix("vocoder.") {
            let rawKey = String(sourceKey.dropFirst("vocoder.".count))
            let key: String
            if rawKey.hasPrefix("bwe_generator.") || rawKey.hasPrefix("mel_stft.") {
                key = rawKey
            } else if rawKey.hasPrefix("vocoder.") {
                key = "base_vocoder." + String(rawKey.dropFirst("vocoder.".count))
            } else {
                key = "base_vocoder." + rawKey
            }
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
        let missing = Set(expected.keys).subtracting(converted.keys).sorted()
        guard missing.isEmpty else {
            throw LTXVideoRuntimeError.missingWeights(missing)
        }
        let parameterTree = ModuleParameters.unflattened(
            converted.sorted { $0.key < $1.key }
        )
        try model.update(parameters: parameterTree, verify: .all)
        MLX.eval(model)
        return LTXVocoderWeightLoadReport(
            tensorCount: converted.count,
            sourceDTypes: sourceDTypes.sorted(),
            computeDType: computeDType
        )
    }

    private static func firstExisting(in directory: URL, paths: [String]) -> URL? {
        paths
            .map { directory.appendingPathComponent($0) }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }
}
