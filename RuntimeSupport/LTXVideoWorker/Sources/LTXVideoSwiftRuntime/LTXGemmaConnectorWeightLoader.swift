import Foundation
import MLX
import MLXNN

public struct LTXGemmaConnectorWeightLoadReport: Sendable {
    public let tensorCount: Int
    public let sourceDTypes: [String]
    public let computeDType: LTXVideoComputeDType

    public init(tensorCount: Int, sourceDTypes: [String], computeDType: LTXVideoComputeDType) {
        self.tensorCount = tensorCount
        self.sourceDTypes = sourceDTypes
        self.computeDType = computeDType
    }
}

public enum LTXGemmaConnectorWeightLoader {
    public static func load(
        connector: LTXGemmaTextEncoderConnector,
        from modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> LTXGemmaConnectorWeightLoadReport {
        let weightsURL = modelDirectory.appendingPathComponent("connector.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXGemmaConnectorRuntimeError.missingWeightsFile(weightsURL)
        }

        let expected = Dictionary(
            uniqueKeysWithValues: connector.parameters().flattened().map { ($0.0, $0.1) }
        )
        let loaded = try MLX.loadArrays(url: weightsURL)
        var converted: [String: MLXArray] = [:]
        var sourceDTypes = Set<String>()

        for (sourceKey, sourceValue) in loaded {
            let key = sourceKey.hasPrefix("connector.")
                ? String(sourceKey.dropFirst("connector.".count))
                : sourceKey
            guard let expectedValue = expected[key] else { continue }
            guard converted[key] == nil else {
                throw LTXGemmaConnectorRuntimeError.duplicateWeight(key)
            }
            guard sourceValue.shape == expectedValue.shape else {
                throw LTXGemmaConnectorRuntimeError.weightShapeMismatch(
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
            throw LTXGemmaConnectorRuntimeError.missingWeights(missing)
        }
        try connector.update(
            parameters: ModuleParameters.unflattened(
                converted.sorted { $0.key < $1.key }
            ),
            verify: .all
        )
        MLX.eval(connector)
        return LTXGemmaConnectorWeightLoadReport(
            tensorCount: converted.count,
            sourceDTypes: sourceDTypes.sorted(),
            computeDType: computeDType
        )
    }

    public static func loadGGUF(
        connector: LTXGemmaTextEncoderConnector,
        mainWeightsURL: URL,
        connectorWeightsURL: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> LTXGemmaConnectorWeightLoadReport {
        guard FileManager.default.fileExists(atPath: connectorWeightsURL.path) else {
            throw LTXGemmaConnectorRuntimeError.missingWeightsFile(connectorWeightsURL)
        }
        let file = try LTXVideo096GGUFFile(url: mainWeightsURL)
        let expected = Dictionary(
            uniqueKeysWithValues: connector.parameters().flattened().map { ($0.0, $0.1) }
        )
        var q4Names = Set<String>()
        var q8Names = Set<String>()
        for tensor in file.tensors where tensor.name.hasSuffix(".weight") {
            guard expected[tensor.name] != nil else { continue }
            switch LTXVideo096GGUFQuantization.targetBits(for: tensor.typeName) {
            case 4: q4Names.insert(parameterPath(tensor.name))
            case 8: q8Names.insert(parameterPath(tensor.name))
            default: break
            }
        }
        if !q4Names.isEmpty || !q8Names.isEmpty {
            ltxQuantizeSorted(model: connector, groupSize: 64) { path, module in
                guard module is Linear else { return nil }
                if q4Names.contains(path) { return 4 }
                if q8Names.contains(path) { return 8 }
                return nil
            }
        }

        var converted: [String: MLXArray] = [:]
        var sourceDTypes = Set<String>()
        var quantizedCount = 0
        let projections = try MLX.loadArrays(url: connectorWeightsURL)
        for (sourceKey, sourceValue) in projections {
            let key = sourceKey.hasPrefix("connector.")
                ? String(sourceKey.dropFirst("connector.".count))
                : sourceKey
            guard expected[key] != nil else { continue }
            sourceDTypes.insert(String(describing: sourceValue.dtype))
            let value = sourceValue.asType(computeDType.mlxDType)
            try insert(value, name: key, expected: expected, into: &converted)
        }

        for tensor in file.tensors {
            guard expected[tensor.name] != nil else { continue }
            let source = try file.array(for: tensor)
            sourceDTypes.insert(tensor.typeName)
            if let bits = LTXVideo096GGUFQuantization.targetBits(for: tensor.typeName),
               tensor.name.hasSuffix(".weight") {
                guard tensor.shape.count == 2,
                      tensor.shape[1].isMultiple(of: 64) else {
                    throw LTXVideo096GGUFError.invalidTensor(tensor.name)
                }
                let quantized = MLX.quantized(
                    source.asType(.float32), groupSize: 64, bits: bits, mode: .affine
                )
                try insert(quantized.wq, name: tensor.name, expected: expected, into: &converted)
                let prefix = String(tensor.name.dropLast(".weight".count))
                try insert(quantized.scales, name: prefix + ".scales", expected: expected, into: &converted)
                if let biases = quantized.biases {
                    try insert(biases, name: prefix + ".biases", expected: expected, into: &converted)
                }
                quantizedCount += 1
            } else {
                try insert(
                    source.asType(computeDType.mlxDType),
                    name: tensor.name,
                    expected: expected,
                    into: &converted
                )
            }
        }

        let missing = Set(expected.keys).subtracting(converted.keys).sorted()
        guard missing.isEmpty else {
            throw LTXGemmaConnectorRuntimeError.missingWeights(missing)
        }
        try connector.update(
            parameters: ModuleParameters.unflattened(
                converted.sorted { $0.key < $1.key }
            ),
            verify: .all
        )
        MLX.eval(connector)
        return LTXGemmaConnectorWeightLoadReport(
            tensorCount: converted.count,
            sourceDTypes: sourceDTypes.sorted(),
            computeDType: computeDType
        )
    }

    private static func parameterPath(_ name: String) -> String {
        name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
    }

    private static func insert(
        _ value: MLXArray,
        name: String,
        expected: [String: MLXArray],
        into converted: inout [String: MLXArray]
    ) throws {
        guard expected[name] != nil else { return }
        guard converted[name] == nil else {
            throw LTXGemmaConnectorRuntimeError.duplicateWeight(name)
        }
        guard value.shape == expected[name]!.shape else {
            throw LTXGemmaConnectorRuntimeError.weightShapeMismatch(
                name: name,
                expected: expected[name]!.shape,
                actual: value.shape
            )
        }
        converted[name] = value
    }
}
