import Foundation
import MLX
import MLXNN

public struct LTXGemma3TextWeightLoadReport: Sendable {
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

public enum LTXGemma3TextWeightLoader {
    public static func load(
        from modelDirectory: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> (model: LTXGemma3AllLayerModel, report: LTXGemma3TextWeightLoadReport) {
        let configuration = try LTXGemma3TextConfiguration.load(from: modelDirectory)
        let model = LTXGemma3AllLayerModel(configuration: configuration)
        let loaded = try loadWeightArrays(from: modelDirectory)
        var normalized: [String: MLXArray] = [:]
        var sourceDTypes = Set<String>()
        for (sourceKey, value) in loaded {
            let key = normalize(sourceKey)
            guard normalized[key] == nil else {
                throw LTXGemma3TextEncoderError.duplicateWeight(key)
            }
            normalized[key] = value
            sourceDTypes.insert(String(describing: value.dtype))
        }

        let quantizedLayers = Set(normalized.keys.compactMap { key -> String? in
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
                    quantizedLayers.contains(path) && (module is Linear || module is Embedding)
                }
            )
        }

        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        var converted: [String: MLXArray] = [:]
        for (key, value) in normalized {
            guard let expectedValue = expected[key] else { continue }
            guard value.shape == expectedValue.shape else {
                throw LTXGemma3TextEncoderError.weightShapeMismatch(
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
            throw LTXGemma3TextEncoderError.missingWeights(missing)
        }
        let parameterTree = ModuleParameters.unflattened(
            converted.sorted { $0.key < $1.key }
        )
        try model.update(parameters: parameterTree, verify: .all)
        MLX.eval(model)
        return (
            model,
            LTXGemma3TextWeightLoadReport(
                tensorCount: converted.count,
                quantizedModuleCount: quantizedLayers.count,
                sourceDTypes: sourceDTypes.sorted(),
                bits: bits,
                groupSize: groupSize
            )
        )
    }

    public static func loadGGUF(
        from weightsURL: URL,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws -> (model: LTXGemma3AllLayerModel, report: LTXGemma3TextWeightLoadReport) {
        let file = try LTXVideo096GGUFFile(url: weightsURL)
        let configuration = try gemmaConfiguration(from: file)
        let model = LTXGemma3AllLayerModel(configuration: configuration)
        var q4Names = Set<String>()
        var q8Names = Set<String>()
        for tensor in file.tensors where tensor.name.hasSuffix(".weight") {
            let key = normalizeGGUF(tensor.name)
            switch LTXVideo096GGUFQuantization.targetBits(for: tensor.typeName) {
            case 4: q4Names.insert(parameterPath(key))
            case 8: q8Names.insert(parameterPath(key))
            default: break
            }
        }
        if !q4Names.isEmpty || !q8Names.isEmpty {
            ltxQuantizeSorted(model: model, groupSize: 64) { path, module in
                guard module is Linear || module is Embedding else { return nil }
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
        var quantizedCount = 0
        for tensor in file.tensors {
            guard let key = normalizeGGUFIfKnown(tensor.name), expected[key] != nil else {
                continue
            }
            sourceDTypes.insert(tensor.typeName)
            let source = try file.array(for: tensor)
            if let bits = LTXVideo096GGUFQuantization.targetBits(for: tensor.typeName),
               tensor.name.hasSuffix(".weight") {
                guard tensor.shape.count == 2,
                      tensor.shape[1].isMultiple(of: 64) else {
                    throw LTXVideo096GGUFError.invalidTensor(tensor.name)
                }
                let quantized = MLX.quantized(
                    source.asType(.float32), groupSize: 64, bits: bits, mode: .affine
                )
                try insert(quantized.wq, name: key, expected: expected, into: &converted)
                let prefix = String(key.dropLast(".weight".count))
                try insert(quantized.scales, name: prefix + ".scales", expected: expected, into: &converted)
                if let biases = quantized.biases {
                    try insert(biases, name: prefix + ".biases", expected: expected, into: &converted)
                }
                quantizedCount += 1
            } else {
                guard source.shape == expected[key]!.shape else {
                    throw LTXGemma3TextEncoderError.weightShapeMismatch(
                        name: key,
                        expected: expected[key]!.shape,
                        actual: source.shape
                    )
                }
                let value = source.dtype == .uint32
                    ? source
                    : source.asType(computeDType.mlxDType)
                try insert(value, name: key, expected: expected, into: &converted)
            }
        }
        let missing = Set(expected.keys).subtracting(converted.keys).sorted()
        guard missing.isEmpty else {
            throw LTXGemma3TextEncoderError.missingWeights(missing)
        }
        let parameterTree = ModuleParameters.unflattened(
            converted.sorted { $0.key < $1.key }
        )
        try model.update(parameters: parameterTree, verify: .all)
        MLX.eval(model)
        return (
            model,
            LTXGemma3TextWeightLoadReport(
                tensorCount: converted.count,
                quantizedModuleCount: quantizedCount,
                sourceDTypes: sourceDTypes.sorted(),
                bits: quantizedCount == 0 ? nil : 4,
                groupSize: quantizedCount == 0 ? nil : 64
            )
        )
    }

    private static func normalize(_ key: String) -> String {
        var normalized = key
        let wrapperPrefixes = ["language_model.", "model."]
        var didStripPrefix = true
        while didStripPrefix {
            didStripPrefix = false
            for prefix in wrapperPrefixes where normalized.hasPrefix(prefix) {
                normalized = String(normalized.dropFirst(prefix.count))
                didStripPrefix = true
                break
            }
        }
        return normalized
    }

    private static func normalizeGGUF(_ key: String) -> String {
        guard let normalized = normalizeGGUFIfKnown(key) else { return key }
        return normalized
    }

    private static func normalizeGGUFIfKnown(_ key: String) -> String? {
        if key == "token_embd.weight" { return "embed_tokens.weight" }
        if key == "output_norm.weight" { return "norm.weight" }
        guard key.hasPrefix("blk."), key.hasSuffix(".weight") else { return nil }
        let components = key.split(separator: ".")
        guard components.count >= 3, let index = Int(components[1]) else { return nil }
        let suffix = components.dropFirst(2).joined(separator: ".")
        let mapping: [String: String] = [
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "attn_q_norm": "self_attn.q_norm",
            "attn_k_norm": "self_attn.k_norm",
            "attn_norm": "input_layernorm",
            "post_attention_norm": "post_attention_layernorm",
            "ffn_norm": "pre_feedforward_layernorm",
            "post_ffw_norm": "post_feedforward_layernorm",
            "ffn_gate": "mlp.gate_proj",
            "ffn_up": "mlp.up_proj",
            "ffn_down": "mlp.down_proj"
        ]
        guard let mapped = mapping[suffix] else { return nil }
        return "layers.\(index).\(mapped).weight"
    }

    private static func parameterPath(_ name: String) -> String {
        name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
    }

    private static func gemmaConfiguration(
        from file: LTXVideo096GGUFFile
    ) throws -> LTXGemma3TextConfiguration {
        let vocabularySize = file.tensors.first(where: { $0.name == "token_embd.weight" })?.shape.first
            ?? file.stringArrayMetadata["tokenizer.ggml.tokens"]?.count
            ?? 0
        return try LTXGemma3TextConfiguration(
            hiddenSize: Int(file.numberMetadata["gemma3.embedding_length"] ?? 0),
            hiddenLayers: Int(file.numberMetadata["gemma3.block_count"] ?? 0),
            intermediateSize: Int(file.numberMetadata["gemma3.feed_forward_length"] ?? 0),
            attentionHeads: Int(file.numberMetadata["gemma3.attention.head_count"] ?? 0),
            headDimension: Int(file.numberMetadata["gemma3.attention.key_length"] ?? 0),
            rmsNormEpsilon: Float(file.numberMetadata["gemma3.attention.layer_norm_rms_epsilon"] ?? 1e-6),
            vocabularySize: vocabularySize,
            keyValueHeads: Int(file.numberMetadata["gemma3.attention.head_count_kv"] ?? 0),
            ropeTheta: Float(file.numberMetadata["gemma3.rope.freq_base"] ?? 1_000_000),
            localRopeBaseFrequency: 10_000,
            queryPreAttentionScalar: 256,
            slidingWindow: Int(file.numberMetadata["gemma3.attention.sliding_window"] ?? 1024),
            slidingWindowPattern: 6,
            maxPositionEmbeddings: Int(file.numberMetadata["gemma3.context_length"] ?? 131_072)
        )
    }

    private static func insert(
        _ value: MLXArray,
        name: String,
        expected: [String: MLXArray],
        into converted: inout [String: MLXArray]
    ) throws {
        guard expected[name] != nil else { return }
        guard converted[name] == nil else {
            throw LTXGemma3TextEncoderError.duplicateWeight(name)
        }
        guard value.shape == expected[name]!.shape else {
            throw LTXGemma3TextEncoderError.weightShapeMismatch(
                name: name,
                expected: expected[name]!.shape,
                actual: value.shape
            )
        }
        converted[name] = value
    }

    private static func loadWeightArrays(from modelDirectory: URL) throws -> [String: MLXArray] {
        let fileNames: [String]
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            struct Index: Decodable {
                let weightMap: [String: String]

                enum CodingKeys: String, CodingKey {
                    case weightMap = "weight_map"
                }
            }
            guard let index = try? JSONDecoder().decode(
                Index.self,
                from: Data(contentsOf: indexURL)
            ) else {
                throw LTXGemma3TextEncoderError.unsupportedWeightIndex(indexURL)
            }
            fileNames = Array(Set(index.weightMap.values)).sorted()
        } else {
            let entries = try FileManager.default.contentsOfDirectory(
                at: modelDirectory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
            fileNames = entries
                .filter { $0.pathExtension == "safetensors" }
                .map(\.lastPathComponent)
                .sorted()
        }
        guard !fileNames.isEmpty else {
            throw LTXGemma3TextEncoderError.missingFile(
                modelDirectory.appendingPathComponent("model.safetensors")
            )
        }

        var arrays: [String: MLXArray] = [:]
        for fileName in fileNames {
            let url = modelDirectory.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw LTXGemma3TextEncoderError.missingFile(url)
            }
            for (key, value) in try MLX.loadArrays(url: url) {
                guard arrays[key] == nil else {
                    throw LTXGemma3TextEncoderError.duplicateWeight(key)
                }
                arrays[key] = value
            }
        }
        return arrays
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
        let values = try JSONDecoder().decode(
            Root.self,
            from: Data(contentsOf: url)
        ).quantization
        return (values.bits, values.groupSize)
    }
}
