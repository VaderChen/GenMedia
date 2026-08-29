import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3RVQDecoderWeightLoadReport: Sendable {
    public let tensorCount: Int
    public let quantizedModuleCount: Int
    public let shardNames: [String]

    public init(tensorCount: Int, quantizedModuleCount: Int, shardNames: [String]) {
        self.tensorCount = tensorCount
        self.quantizedModuleCount = quantizedModuleCount
        self.shardNames = shardNames
    }
}

public enum MiniMaxMusic3RVQDecoderWeightLoader {
    public static func load(
        model: MiniMaxMusic3RVQDepthDecoder,
        from modelDirectory: URL
    ) throws -> MiniMaxMusic3RVQDecoderWeightLoadReport {
        var quantizedPaths = Set<String>()
        quantize(
            model: model,
            groupSize: MiniMaxMusic3GGUFQuantizationStrategy.groupSize,
            bits: 4,
            mode: MiniMaxMusic3GGUFQuantizationStrategy.mode,
            filter: { path, module in
                guard module is Linear, path.hasPrefix("layers.") else { return false }
                guard path.contains(".attn.to_q")
                    || path.contains(".attn.to_k")
                    || path.contains(".attn.to_v")
                    || path.contains(".attn.to_out")
                    || path.contains(".gate_proj")
                    || path.contains(".up_proj")
                    || path.contains(".down_proj") else {
                    return false
                }
                quantizedPaths.insert(path)
                return true
            }
        )

        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        let shardNames = try shardNames(from: modelDirectory)
        var converted: [String: MLXArray] = [:]

        for shardName in shardNames {
            let shardURL = modelDirectory.appendingPathComponent(shardName)
            let arrays = try MLX.loadArrays(url: shardURL)
            for (sourceKey, sourceValue) in arrays where sourceKey.hasPrefix("rvq_depth_decoder.") {
                let key = String(sourceKey.dropFirst("rvq_depth_decoder.".count))
                guard expected[key] != nil else { continue }
                guard converted[key] == nil else {
                    throw MiniMaxMusic3RVQDecoderError.unexpectedWeights([key])
                }
                guard sourceValue.shape == expected[key]!.shape else {
                    throw MiniMaxMusic3RVQDecoderError.weightShapeMismatch(
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
            throw MiniMaxMusic3RVQDecoderError.missingWeights(missing)
        }
        let unexpected = convertedKeys.subtracting(expectedKeys).sorted()
        guard unexpected.isEmpty else {
            throw MiniMaxMusic3RVQDecoderError.unexpectedWeights(unexpected)
        }

        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
        return MiniMaxMusic3RVQDecoderWeightLoadReport(
            tensorCount: converted.count,
            quantizedModuleCount: quantizedPaths.count,
            shardNames: shardNames
        )
    }

    public static func loadGGUF(
        model: MiniMaxMusic3RVQDepthDecoder,
        from fileURL: URL
    ) throws -> MiniMaxMusic3RVQDecoderWeightLoadReport {
        let report = try MiniMaxMusic3GGUFWeightLoader.load(model: model, from: fileURL)
        return MiniMaxMusic3RVQDecoderWeightLoadReport(
            tensorCount: report.parameterCount,
            quantizedModuleCount: report.quantizedTensorCount,
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
                throw MiniMaxMusic3RVQDecoderError.missingFile(indexURL)
            }
            return names
        }

        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        let names = Set(
            index.weightMap.compactMap { key, file in
                key.hasPrefix("rvq_depth_decoder.") ? file : nil
            }
        ).sorted()
        guard !names.isEmpty else {
            throw MiniMaxMusic3RVQDecoderError.missingWeights(["rvq_depth_decoder.*"])
        }
        for name in names {
            let url = modelDirectory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MiniMaxMusic3RVQDecoderError.missingFile(url)
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
