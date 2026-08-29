import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3FlowTransformerWeightLoadReport: Sendable {
    public let tensorCount: Int
    public let quantizedModuleCount: Int
    public let shardNames: [String]

    public init(tensorCount: Int, quantizedModuleCount: Int, shardNames: [String]) {
        self.tensorCount = tensorCount
        self.quantizedModuleCount = quantizedModuleCount
        self.shardNames = shardNames
    }
}

public enum MiniMaxMusic3FlowTransformerWeightLoader {
    public static func load(
        model: MiniMaxMusic3FlowTransformer,
        from modelDirectory: URL
    ) throws -> MiniMaxMusic3FlowTransformerWeightLoadReport {
        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        let shardNames = try transformerShardNames(from: modelDirectory)
        var converted: [String: MLXArray] = [:]

        for shardName in shardNames {
            let shardURL = modelDirectory.appendingPathComponent(shardName)
            let arrays = try MLX.loadArrays(url: shardURL)
            for (sourceKey, sourceValue) in arrays where sourceKey.hasPrefix("transformer.") {
                let key = String(sourceKey.dropFirst("transformer.".count))
                guard expected[key] != nil else { continue }
                guard converted[key] == nil else {
                    throw MiniMaxMusic3FlowTransformerError.unexpectedWeights([key])
                }
                guard sourceValue.shape == expected[key]!.shape else {
                    throw MiniMaxMusic3FlowTransformerError.weightShapeMismatch(
                        name: key,
                        expected: expected[key]!.shape,
                        actual: sourceValue.shape
                    )
                }
                converted[key] = sourceValue
            }
        }

        let expectedKeys = Set(expected.keys)
        let convertedKeys = Set(converted.keys)
        let missing = expectedKeys.subtracting(convertedKeys).sorted()
        guard missing.isEmpty else {
            throw MiniMaxMusic3FlowTransformerError.missingWeights(missing)
        }
        let unexpected = convertedKeys.subtracting(expectedKeys).sorted()
        guard unexpected.isEmpty else {
            throw MiniMaxMusic3FlowTransformerError.unexpectedWeights(unexpected)
        }

        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
        return MiniMaxMusic3FlowTransformerWeightLoadReport(
            tensorCount: converted.count,
            quantizedModuleCount: (model.configuration.numLayers * 6) + 2,
            shardNames: shardNames
        )
    }

    public static func loadGGUF(
        model: MiniMaxMusic3FlowTransformer,
        from fileURL: URL
    ) throws -> MiniMaxMusic3FlowTransformerWeightLoadReport {
        let report = try MiniMaxMusic3GGUFWeightLoader.load(
            model: model,
            from: fileURL,
            valueTransform: { name, value in
                switch name {
                case "preprocess_conv.weight", "postprocess_conv.weight":
                    return value.transposed(0, 2, 1)
                default:
                    return value
                }
            }
        )
        return MiniMaxMusic3FlowTransformerWeightLoadReport(
            tensorCount: report.parameterCount,
            quantizedModuleCount: report.quantizedTensorCount,
            shardNames: [fileURL.lastPathComponent]
        )
    }

    private static func transformerShardNames(from modelDirectory: URL) throws -> [String] {
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
                throw MiniMaxMusic3FlowTransformerError.missingFile(indexURL)
            }
            return safetensors
        }

        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        let shardNames = Set(
            index.weightMap.compactMap { key, file in
                key.hasPrefix("transformer.") ? file : nil
            }
        ).sorted()
        guard !shardNames.isEmpty else {
            throw MiniMaxMusic3FlowTransformerError.missingWeights(["transformer.*"])
        }
        for shardName in shardNames {
            let shardURL = modelDirectory.appendingPathComponent(shardName)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw MiniMaxMusic3FlowTransformerError.missingFile(shardURL)
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
