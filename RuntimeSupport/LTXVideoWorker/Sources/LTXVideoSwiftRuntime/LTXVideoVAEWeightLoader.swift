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
    public static func loadEncoderStatistics(
        from modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> LTXVideoLatentStatistics {
        let weightsURL = firstExisting(
            in: modelDirectory,
            paths: [
                "vae_encoder.safetensors",
                "vae/ltx-2.3-22b-distilled_video_vae.safetensors"
            ]
        ) ?? modelDirectory.appendingPathComponent("vae_encoder.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXVideoRuntimeError.missingFile(weightsURL)
        }
        let source = try MLX.loadArrays(url: weightsURL)
        let mean = try statistic(
            in: source,
            candidates: [
                "vae_encoder.per_channel_statistics._mean_of_means",
                "vae_encoder.per_channel_statistics.mean_of_means",
                "vae_encoder.per_channel_statistics.meanOfMeans",
                "per_channel_statistics.mean-of-means"
            ],
            label: "per_channel_statistics.meanOfMeans"
        )
        let standardDeviation = try statistic(
            in: source,
            candidates: [
                "vae_encoder.per_channel_statistics._std_of_means",
                "vae_encoder.per_channel_statistics.std_of_means",
                "vae_encoder.per_channel_statistics.stdOfMeans",
                "per_channel_statistics.std-of-means"
            ],
            label: "per_channel_statistics.stdOfMeans"
        )
        return try LTXVideoLatentStatistics(
            meanOfMeans: mean.asType(computeDType.mlxDType),
            stdOfMeans: standardDeviation.asType(computeDType.mlxDType)
        )
    }

    public static func loadDecoder(
        model: LTXVideoVAEDecoder,
        from modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> LTXVideoVAEWeightLoadReport {
        let weightsURL = firstExisting(
            in: modelDirectory,
            paths: [
                "vae_decoder.safetensors",
                "vae/ltx-2.3-22b-distilled_video_vae.safetensors"
            ]
        ) ?? modelDirectory.appendingPathComponent("vae_decoder.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXVideoRuntimeError.missingFile(weightsURL)
        }
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        let source = try MLX.loadArrays(url: weightsURL)
        var converted: [String: MLXArray] = [:]
        var sourceDTypes: Set<String> = []
        for (sourceKey, sourceValue) in source {
            let key: String
            if sourceKey.hasPrefix("vae_decoder.") {
                key = String(sourceKey.dropFirst("vae_decoder.".count))
            } else if sourceKey.hasPrefix("decoder.") {
                key = String(sourceKey.dropFirst("decoder.".count))
            } else if sourceKey == "per_channel_statistics.mean-of-means" {
                key = "per_channel_statistics.mean"
            } else if sourceKey == "per_channel_statistics.std-of-means" {
                key = "per_channel_statistics.std"
            } else {
                continue
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

        let parameterTree = ModuleParameters.unflattened(
            converted.sorted { $0.key < $1.key }
        )
        try model.update(parameters: parameterTree, verify: .all)
        MLX.eval(model)
        return LTXVideoVAEWeightLoadReport(
            tensorCount: converted.count,
            quantizedModuleCount: 0,
            sourceDTypes: sourceDTypes.sorted(),
            computeDType: computeDType
        )
    }

    public static func loadEncoder(
        model: LTXVideoVAEEncoder,
        from modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> LTXVideoVAEWeightLoadReport {
        let weightsURL = firstExisting(
            in: modelDirectory,
            paths: [
                "vae_encoder.safetensors",
                "vae/ltx-2.3-22b-distilled_video_vae.safetensors"
            ]
        ) ?? modelDirectory.appendingPathComponent("vae_encoder.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXVideoRuntimeError.missingFile(weightsURL)
        }
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        let source = try MLX.loadArrays(url: weightsURL)
        var converted: [String: MLXArray] = [:]
        var sourceDTypes: Set<String> = []
        for (sourceKey, sourceValue) in source where sourceKey.hasPrefix("vae_encoder.") {
            let rawKey = String(sourceKey.dropFirst("vae_encoder.".count))
            let key = rawKey
                .replacingOccurrences(of: "per_channel_statistics._mean_of_means", with: "per_channel_statistics.meanOfMeans")
                .replacingOccurrences(of: "per_channel_statistics._std_of_means", with: "per_channel_statistics.stdOfMeans")
                .replacingOccurrences(of: "per_channel_statistics.mean-of-means", with: "per_channel_statistics.meanOfMeans")
                .replacingOccurrences(of: "per_channel_statistics.std-of-means", with: "per_channel_statistics.stdOfMeans")
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
        return LTXVideoVAEWeightLoadReport(
            tensorCount: converted.count,
            quantizedModuleCount: 0,
            sourceDTypes: sourceDTypes.sorted(),
            computeDType: computeDType
        )
    }

    private static func statistic(
        in source: [String: MLXArray],
        candidates: [String],
        label: String
    ) throws -> MLXArray {
        guard let value = candidates.lazy.compactMap({ source[$0] }).first else {
            throw LTXVideoRuntimeError.missingWeights([label])
        }
        guard value.shape == [128] else {
            throw LTXVideoRuntimeError.weightShapeMismatch(
                name: label,
                expected: [128],
                actual: value.shape
            )
        }
        return value
    }

    private static func firstExisting(in directory: URL, paths: [String]) -> URL? {
        paths
            .map { directory.appendingPathComponent($0) }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }
}
