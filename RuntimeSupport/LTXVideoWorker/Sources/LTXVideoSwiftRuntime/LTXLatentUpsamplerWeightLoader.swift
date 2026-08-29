import Foundation
import MLX
import MLXNN

public struct LTXLatentUpsamplerWeightLoadReport: Sendable {
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

public enum LTXLatentUpsamplerWeightLoader {
    public static func load(
        from weightsURL: URL,
        modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> (model: LTXLatentUpsampler, report: LTXLatentUpsamplerWeightLoadReport) {
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw LTXVideoRuntimeError.missingFile(weightsURL)
        }
        let baseName = weightsURL.deletingPathExtension().lastPathComponent
        let configurationURL = modelDirectory.appendingPathComponent("\(baseName)_config.json")
        let configuration: LTXLatentUpsamplerConfiguration
        if FileManager.default.fileExists(atPath: configurationURL.path) {
            configuration = try LTXLatentUpsamplerConfiguration.load(from: configurationURL)
        } else {
            configuration = try LTXLatentUpsamplerConfiguration()
        }
        let model = try LTXLatentUpsampler(configuration: configuration)
        let source = try MLX.loadArrays(url: weightsURL)
        let expected: [String: MLXArray] = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        var converted: [String: MLXArray] = [:]
        var sourceDTypes: Set<String> = []
        for (sourceKey, sourceValue) in source {
            let key = normalize(sourceKey, baseName: baseName)
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
        return (
            model,
            LTXLatentUpsamplerWeightLoadReport(
                tensorCount: converted.count,
                quantizedModuleCount: 0,
                sourceDTypes: sourceDTypes.sorted(),
                computeDType: computeDType
            )
        )
    }

    private static func normalize(_ key: String, baseName: String) -> String {
        if key.hasPrefix("\(baseName).") {
            return String(key.dropFirst(baseName.count + 1))
        }
        if key.hasPrefix("model.\(baseName).") {
            return String(key.dropFirst("model.\(baseName).".count))
        }
        return key
    }
}
