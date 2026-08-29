import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3ConditionEncoderWeightLoadReport: Sendable {
    public let tensorCount: Int
    public let shardNames: [String]

    public init(tensorCount: Int, shardNames: [String]) {
        self.tensorCount = tensorCount
        self.shardNames = shardNames
    }
}

public enum MiniMaxMusic3ConditionEncoderWeightLoader {
    public static func load(
        model: MiniMaxMusic3ConditionEncoder,
        from modelDirectory: URL
    ) throws -> MiniMaxMusic3ConditionEncoderWeightLoadReport {
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        let shardNames = try shardNames(from: modelDirectory)
        var converted: [String: MLXArray] = [:]

        for shardName in shardNames {
            let shardURL = modelDirectory.appendingPathComponent(shardName)
            let arrays = try MLX.loadArrays(url: shardURL)
            for (sourceKey, sourceValue) in arrays where sourceKey.hasPrefix("condition_encoder.") {
                let key = String(sourceKey.dropFirst("condition_encoder.".count))
                guard expected[key] != nil else { continue }
                guard converted[key] == nil else {
                    throw MiniMaxMusic3ConditionEncoderError.unexpectedWeights([key])
                }
                let value = sourceValue
                guard value.shape == expected[key]!.shape else {
                    throw MiniMaxMusic3ConditionEncoderError.weightShapeMismatch(
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
            throw MiniMaxMusic3ConditionEncoderError.missingWeights(missing)
        }
        let unexpected = convertedKeys.subtracting(expectedKeys).sorted()
        guard unexpected.isEmpty else {
            throw MiniMaxMusic3ConditionEncoderError.unexpectedWeights(unexpected)
        }

        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
        return MiniMaxMusic3ConditionEncoderWeightLoadReport(
            tensorCount: converted.count,
            shardNames: shardNames
        )
    }

    public static func loadGGUF(
        model: MiniMaxMusic3ConditionEncoder,
        from fileURL: URL
    ) throws -> MiniMaxMusic3ConditionEncoderWeightLoadReport {
        let raw = try MiniMaxMusic3GGUFWeightLoader.loadRawArrays(from: fileURL)
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        var converted: [String: MLXArray] = [:]
        for (key, sourceValue) in raw.arrays {
            let value = key == "proj.weight"
                ? sourceValue.transposed(0, 2, 1)
                : sourceValue
            guard expected[key] != nil else { continue }
            guard value.shape == expected[key]!.shape else {
                throw MiniMaxMusic3GGUFError.weightShapeMismatch(
                    name: key,
                    expected: expected[key]!.shape,
                    actual: value.shape
                )
            }
            converted[key] = value
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
        return MiniMaxMusic3ConditionEncoderWeightLoadReport(
            tensorCount: converted.count,
            shardNames: [fileURL.lastPathComponent]
        )
    }

    private static func shardNames(from modelDirectory: URL) throws -> [String] {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            let files = try FileManager.default.contentsOfDirectory(
                at: modelDirectory,
                includingPropertiesForKeys: nil
            )
            let names = files
                .filter { $0.pathExtension == "safetensors" }
                .map(\.lastPathComponent)
                .sorted()
            guard !names.isEmpty else {
                throw MiniMaxMusic3ConditionEncoderError.missingFile(indexURL)
            }
            return names
        }

        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        let names = Set(
            index.weightMap.compactMap { key, file in
                key.hasPrefix("condition_encoder.") ? file : nil
            }
        ).sorted()
        guard !names.isEmpty else {
            throw MiniMaxMusic3ConditionEncoderError.missingWeights(["condition_encoder.*"])
        }
        for name in names {
            let url = modelDirectory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MiniMaxMusic3ConditionEncoderError.missingFile(url)
            }
        }
        return names
    }

    private struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }
}
