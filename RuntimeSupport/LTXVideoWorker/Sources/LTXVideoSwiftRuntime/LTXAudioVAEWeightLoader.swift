import Foundation
import MLX
import MLXNN

public struct LTXAudioVAEWeightLoadReport: Sendable {
    public let tensorCount: Int
    public let sourceDTypes: [String]
    public let computeDType: LTXVideoComputeDType

    public init(tensorCount: Int, sourceDTypes: [String], computeDType: LTXVideoComputeDType) {
        self.tensorCount = tensorCount
        self.sourceDTypes = sourceDTypes
        self.computeDType = computeDType
    }
}

public enum LTXAudioVAEWeightLoader {
    public static func loadDecoder(
        model: LTXAudioVAEDecoder,
        from modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> LTXAudioVAEWeightLoadReport {
        let weightsURL = firstExisting(
            in: modelDirectory,
            paths: [
                "audio_vae.safetensors",
                "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"
            ]
        ) ?? modelDirectory.appendingPathComponent("audio_vae.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXVideoRuntimeError.missingFile(weightsURL)
        }

        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        let source = try MLX.loadArrays(url: weightsURL)
        var converted: [String: MLXArray] = [:]
        var sourceDTypes = Set<String>()

        for (sourceKey, sourceValue) in source {
            let key: String?
            if sourceKey.hasPrefix("audio_vae.decoder.") {
                key = String(sourceKey.dropFirst("audio_vae.decoder.".count))
            } else if sourceKey.hasPrefix("audio_vae.per_channel_statistics.") {
                let suffix = String(sourceKey.dropFirst("audio_vae.".count))
                key = suffix
                    .replacingOccurrences(of: "per_channel_statistics.mean-of-means", with: "per_channel_statistics.meanOfMeans")
                    .replacingOccurrences(of: "per_channel_statistics.std-of-means", with: "per_channel_statistics.stdOfMeans")
                    .replacingOccurrences(of: "_mean_of_means", with: "meanOfMeans")
                    .replacingOccurrences(of: "_std_of_means", with: "stdOfMeans")
            } else {
                key = nil
            }
            guard let key, let expectedValue = expected[key] else { continue }
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
        return LTXAudioVAEWeightLoadReport(
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
