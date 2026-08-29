import Foundation
import MLX
import MLXNN

public enum LTXTransformerGGUFWeightLoader {
    public static func load(
        from weightsURL: URL,
        configuration: LTXTransformerConfiguration = try! .init(),
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> (model: LTXTransformer, report: LTXTransformerWeightLoadReport) {
        let file = try LTXVideo096GGUFFile(url: weightsURL)
        let layerCount = file.tensors.compactMap { tensor -> Int? in
            let components = tensor.name.split(separator: ".")
            guard components.count > 1,
                  components[0] == "transformer_blocks",
                  let index = Int(components[1]) else { return nil }
            return index
        }.max().map { $0 + 1 } ?? configuration.numLayers
        let modelConfiguration = try LTXTransformerConfiguration(
            numLayers: layerCount,
            videoDim: configuration.videoDim,
            audioDim: configuration.audioDim,
            videoNumHeads: configuration.videoNumHeads,
            audioNumHeads: configuration.audioNumHeads,
            videoHeadDim: configuration.videoHeadDim,
            audioHeadDim: configuration.audioHeadDim,
            avCrossNumHeads: configuration.avCrossNumHeads,
            avCrossHeadDim: configuration.avCrossHeadDim,
            videoPatchChannels: configuration.videoPatchChannels,
            audioPatchChannels: configuration.audioPatchChannels,
            ffMult: configuration.ffMult,
            timestepEmbeddingDim: configuration.timestepEmbeddingDim,
            timestepScaleMultiplier: configuration.timestepScaleMultiplier,
            avCATimestepScaleMultiplier: configuration.avCATimestepScaleMultiplier,
            ropeTheta: configuration.ropeTheta,
            ropeType: configuration.ropeType,
            positionalEmbeddingMaxPos: configuration.positionalEmbeddingMaxPos,
            audioPositionalEmbeddingMaxPos: configuration.audioPositionalEmbeddingMaxPos,
            normEps: configuration.normEps
        )
        let model = LTXTransformer(configuration: modelConfiguration)
        let sourceWeights = file.tensors.reduce(into: [String: LTXVideo096GGUFTensor]()) { result, tensor in
            result[normalize(tensor.name)] = tensor
        }
        let q4Names = Set(sourceWeights.values.filter {
            ["Q3_K", "Q4_K"].contains($0.typeName)
        }.map { parameterPath(normalize($0.name)) })
        let q8Names = Set(sourceWeights.values.filter {
            ["Q5_K", "Q6_K", "IQ4_XS"].contains($0.typeName)
        }.map { parameterPath(normalize($0.name)) })

        if !q4Names.isEmpty || !q8Names.isEmpty {
            ltxQuantizeSorted(model: model, groupSize: 64) { path, module in
                guard module is Linear else { return nil }
                if q4Names.contains(path) { return 4 }
                if q8Names.contains(path) { return 8 }
                return nil
            }
        }

        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        var converted: [String: MLXArray] = [:]
        var sourceDTypes = Set<String>()
        var loadedSourceCount = 0
        var quantizedModuleCount = 0

        for tensor in file.tensors {
            let key = normalize(tensor.name)
            guard expected[key] != nil else { continue }
            let source = try file.array(for: tensor)
            sourceDTypes.insert(tensor.typeName)
            if let bits = LTXVideo096GGUFQuantization.targetBits(for: tensor.typeName) {
                guard tensor.shape.count == 2,
                      tensor.shape[0] % 32 == 0,
                      tensor.shape[1] % 64 == 0 else {
                    throw LTXVideo096GGUFError.invalidTensor(tensor.name)
                }
                let quantized = MLX.quantized(
                    source.asType(.float32),
                    groupSize: 64,
                    bits: bits,
                    mode: .affine
                )
                try insert(quantized.wq, name: key, expected: expected, into: &converted)
                let prefix = String(key.dropLast(".weight".count))
                try insert(quantized.scales, name: prefix + ".scales", expected: expected, into: &converted)
                if let biases = quantized.biases {
                    try insert(biases, name: prefix + ".biases", expected: expected, into: &converted)
                }
                quantizedModuleCount += 1
            } else {
                let value = source.dtype == .uint32
                    ? source
                    : source.asType(computeDType.mlxDType)
                try insert(value, name: key, expected: expected, into: &converted)
            }
            loadedSourceCount += 1
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
                tensorCount: loadedSourceCount,
                quantizedModuleCount: quantizedModuleCount,
                sourceDTypes: sourceDTypes.sorted(),
                bits: q8Names.isEmpty && q4Names.isEmpty ? nil : 4,
                groupSize: q8Names.isEmpty && q4Names.isEmpty ? nil : 64
            )
        )
    }

    private static func normalize(_ sourceKey: String) -> String {
        var key = sourceKey
        while key.hasPrefix("transformer.") {
            key = String(key.dropFirst("transformer.".count))
        }
        key = key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
        key = key.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
        key = key.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
        return key
    }

    private static func parameterPath(_ name: String) -> String {
        name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count))
            : name
    }

    private static func insert(
        _ value: MLXArray,
        name: String,
        expected: [String: MLXArray],
        into converted: inout [String: MLXArray]
    ) throws {
        guard expected[name] != nil else { return }
        guard converted[name] == nil else {
            throw LTXVideo096GGUFError.duplicateWeight(name)
        }
        guard value.shape == expected[name]!.shape else {
            throw LTXVideo096GGUFError.weightShapeMismatch(
                name: name,
                expected: expected[name]!.shape,
                actual: value.shape
            )
        }
        converted[name] = value
    }
}
