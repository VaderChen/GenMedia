import Foundation
import MLX
import MLXNN

public struct LTXTransformerWeightLoadReport: Sendable {
    public let tensorCount: Int
    public let quantizedModuleCount: Int
    public let sourceDTypes: [String]
    public let bits: Int?
    public let groupSize: Int?

    public init(
        tensorCount: Int,
        quantizedModuleCount: Int,
        sourceDTypes: [String],
        bits: Int?,
        groupSize: Int?
    ) {
        self.tensorCount = tensorCount
        self.quantizedModuleCount = quantizedModuleCount
        self.sourceDTypes = sourceDTypes
        self.bits = bits
        self.groupSize = groupSize
    }
}

public enum LTXTransformerWeightLoader {
    public static func load(
        from weightsURL: URL,
        modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> (model: LTXTransformer, report: LTXTransformerWeightLoadReport) {
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXVideoRuntimeError.missingFile(weightsURL)
        }
        let configuration = try LTXTransformerConfiguration.load(from: modelDirectory)
        let model = LTXTransformer(configuration: configuration)
        let loaded = try MLX.loadArrays(url: weightsURL)
        var weights: [String: MLXArray] = [:]
        var sourceDTypes = Set<String>()
        for (key, value) in loaded {
            let normalized = key.hasPrefix("transformer.")
                ? String(key.dropFirst("transformer.".count))
                : key
            weights[normalized] = value
            sourceDTypes.insert(String(describing: value.dtype))
        }

        let quantizedLayers = Set(weights.keys.compactMap { key -> String? in
            guard key.hasSuffix(".scales") else { return nil }
            return String(key.dropLast(".scales".count))
        })
        var bits: Int?
        var groupSize: Int?
        if !quantizedLayers.isEmpty {
            let quantization = try readQuantization(from: modelDirectory)
            bits = quantization.bits
            groupSize = quantization.groupSize
            quantize(
                model: model,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                filter: { path, module in
                    quantizedLayers.contains(path) && module is Linear
                }
            )
        }

        var converted: [String: MLXArray] = [:]
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) })
        for (key, value) in weights {
            guard let expectedValue = expected[key] else { continue }
            guard value.shape == expectedValue.shape else {
                throw LTXVideoRuntimeError.weightShapeMismatch(
                    name: key,
                    expected: expectedValue.shape,
                    actual: value.shape
                )
            }
            converted[key] = value.dtype == .uint32
                ? value
                : value.asType(computeDType.mlxDType)
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
        return (
            model,
            LTXTransformerWeightLoadReport(
                tensorCount: converted.count,
                quantizedModuleCount: quantizedLayers.count,
                sourceDTypes: sourceDTypes.sorted(),
                bits: bits,
                groupSize: groupSize
            )
        )
    }

    private static func readQuantization(from modelDirectory: URL) throws -> (bits: Int, groupSize: Int) {
        let url = modelDirectory.appendingPathComponent("quantize_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            return (4, 64)
        }
        struct Root: Decodable {
            let quantization: Values
        }
        struct Values: Decodable {
            let bits: Int
            let groupSize: Int

            enum CodingKeys: String, CodingKey {
                case bits
                case groupSize = "group_size"
            }
        }
        let values = try JSONDecoder().decode(Root.self, from: Data(contentsOf: url)).quantization
        return (values.bits, values.groupSize)
    }
}
