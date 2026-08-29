import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3VocoderWeightLoadReport: Sendable {
    public let tensorCount: Int
    public let shardNames: [String]
    public let transposedAlphaTensorCount: Int

    public init(tensorCount: Int, shardNames: [String], transposedAlphaTensorCount: Int) {
        self.tensorCount = tensorCount
        self.shardNames = shardNames
        self.transposedAlphaTensorCount = transposedAlphaTensorCount
    }
}

public enum MiniMaxMusic3VocoderWeightLoader {
    public static func load(
        model: MiniMaxMusic3Vocoder,
        from modelDirectory: URL
    ) throws -> MiniMaxMusic3VocoderWeightLoadReport {
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        let shardNames = try vocoderShardNames(from: modelDirectory)
        var converted: [String: MLXArray] = [:]
        var transposedAlphaTensorCount = 0

        for shardName in shardNames {
            let shardURL = modelDirectory.appendingPathComponent(shardName)
            let arrays = try MLX.loadArrays(url: shardURL)
            for (sourceKey, sourceValue) in arrays where sourceKey.hasPrefix("vocoder.") {
                let key = String(sourceKey.dropFirst("vocoder.".count))
                guard expected[key] != nil else { continue }
                guard converted[key] == nil else {
                    throw MiniMaxMusic3VocoderError.unexpectedWeights([key])
                }
                var value = sourceValue
                if key.hasSuffix(".alpha") {
                    value = value.transposed(0, 2, 1)
                    transposedAlphaTensorCount += 1
                }
                guard value.shape == expected[key]!.shape else {
                    throw MiniMaxMusic3VocoderError.weightShapeMismatch(
                        name: key,
                        expected: expected[key]!.shape,
                        actual: value.shape
                    )
                }
                converted[key] = value
            }
        }

        let expectedKeys = Set(expected.keys)
        let convertedKeys = Set(converted.keys)
        let missing = expectedKeys.subtracting(convertedKeys).sorted()
        guard missing.isEmpty else {
            throw MiniMaxMusic3VocoderError.missingWeights(missing)
        }
        let unexpected = convertedKeys.subtracting(expectedKeys).sorted()
        guard unexpected.isEmpty else {
            throw MiniMaxMusic3VocoderError.unexpectedWeights(unexpected)
        }

        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
        return MiniMaxMusic3VocoderWeightLoadReport(
            tensorCount: converted.count,
            shardNames: shardNames,
            transposedAlphaTensorCount: transposedAlphaTensorCount
        )
    }

    public static func loadGGUF(
        model: MiniMaxMusic3Vocoder,
        from fileURL: URL
    ) throws -> MiniMaxMusic3VocoderWeightLoadReport {
        let raw = try MiniMaxMusic3GGUFWeightLoader.loadRawArrays(from: fileURL)
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        var converted: [String: MLXArray] = [:]
        for (sourceKey, sourceValue) in raw.arrays {
            if sourceKey.hasSuffix(".weight_v") {
                let prefix = String(sourceKey.dropLast(".weight_v".count))
                guard let weightG = raw.arrays[prefix + ".weight_g"] else {
                    throw MiniMaxMusic3GGUFError.missingWeights([prefix + ".weight_g"])
                }
                let canonicalWeightV: MLXArray
                let canonicalWeightG: MLXArray
                if sourceValue.ndim == 2, weightG.ndim == 1, weightG.size == 1 {
                    canonicalWeightV = sourceValue.reshaped(
                        1,
                        sourceValue.shape[0],
                        sourceValue.shape[1]
                    )
                    canonicalWeightG = weightG.reshaped(1, 1, 1)
                } else {
                    canonicalWeightV = sourceValue
                    canonicalWeightG = weightG
                }
                let normalized = canonicalWeightV * canonicalWeightG
                    / sqrt(canonicalWeightV.square().sum(axes: [1, 2], keepDims: true))
                let key = prefix + ".weight"
                let value = prefix.hasSuffix(".conv_t1")
                    ? normalized.transposed(1, 2, 0)
                    : normalized.transposed(0, 2, 1)
                guard expected[key] != nil else { continue }
                guard value.shape == expected[key]!.shape else {
                    throw MiniMaxMusic3GGUFError.weightShapeMismatch(
                        name: key,
                        expected: expected[key]!.shape,
                        actual: value.shape
                    )
                }
                converted[key] = value
            } else if sourceKey.hasSuffix(".weight_g") {
                continue
            } else if sourceKey.hasSuffix(".alpha") {
                let key = sourceKey
                let value: MLXArray
                if sourceValue.ndim == 2, sourceValue.shape[1] == 1 {
                    value = sourceValue.reshaped(1, 1, sourceValue.shape[0])
                } else {
                    value = sourceValue
                }
                guard expected[key] != nil else { continue }
                guard value.shape == expected[key]!.shape else {
                    throw MiniMaxMusic3GGUFError.weightShapeMismatch(
                        name: key,
                        expected: expected[key]!.shape,
                        actual: value.shape
                    )
                }
                converted[key] = value
            } else if expected[sourceKey] != nil {
                let value = sourceValue.ndim == 3
                    ? sourceValue.transposed(0, 2, 1)
                    : sourceValue
                guard value.shape == expected[sourceKey]!.shape else {
                    throw MiniMaxMusic3GGUFError.weightShapeMismatch(
                        name: sourceKey,
                        expected: expected[sourceKey]!.shape,
                        actual: value.shape
                    )
                }
                converted[sourceKey] = value
            }
        }
        let missing = Set(expected.keys).subtracting(converted.keys).sorted()
        guard missing.isEmpty else {
            throw MiniMaxMusic3GGUFError.missingWeights(missing)
        }
        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
        return MiniMaxMusic3VocoderWeightLoadReport(
            tensorCount: converted.count,
            shardNames: [fileURL.lastPathComponent],
            transposedAlphaTensorCount: 0
        )
    }

    private static func vocoderShardNames(from modelDirectory: URL) throws -> [String] {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            let files = try FileManager.default.contentsOfDirectory(
                at: modelDirectory,
                includingPropertiesForKeys: nil
            )
            let safetensors = files
                .filter { $0.pathExtension == "safetensors" }
                .map { $0.lastPathComponent }
                .sorted()
            guard !safetensors.isEmpty else {
                throw MiniMaxMusic3VocoderError.missingFile(indexURL)
            }
            return safetensors
        }

        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        let shardNames = Set(
            index.weightMap.compactMap { key, file in
                key.hasPrefix("vocoder.") ? file : nil
            }
        ).sorted()
        guard !shardNames.isEmpty else {
            throw MiniMaxMusic3VocoderError.missingWeights(["vocoder.*"])
        }
        for shardName in shardNames {
            let shardURL = modelDirectory.appendingPathComponent(shardName)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw MiniMaxMusic3VocoderError.missingFile(shardURL)
            }
        }
        return shardNames
    }

    private struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }
}
