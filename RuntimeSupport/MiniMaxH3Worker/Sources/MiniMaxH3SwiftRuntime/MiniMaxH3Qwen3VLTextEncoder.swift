import Foundation
import GenImageGGUF
import Hub
import MLX
import MLXNN
import Tokenizers

/// The text-only part of the Qwen3-VL encoder shipped with MiniMax H3.
///
/// H3 uses the raw hidden states after the last retained language-model layer;
/// it does not use a chat template or the Qwen final RMS norm.  The released
/// H3 GGUF contains 50 language layers (the upstream config describes 64), so
/// this implementation derives and validates the retained depth from the
/// checkpoint inventory instead of silently constructing missing layers.
public final class MiniMaxH3Qwen3VLTextEncoder {
    public struct Configuration: Sendable, Equatable {
        public let hiddenSize: Int
        public let vocabularySize: Int
        public let layerCount: Int
        public let intermediateSize: Int
        public let attentionHeadCount: Int
        public let keyValueHeadCount: Int
        public let headDimension: Int
        public let ropeTheta: Float
        public let rmsNormEps: Float
        public let maxPositionEmbeddings: Int

        public static let minimaxH3 = Configuration(
            hiddenSize: 5120,
            vocabularySize: 151936,
            layerCount: 50,
            intermediateSize: 25600,
            attentionHeadCount: 64,
            keyValueHeadCount: 8,
            headDimension: 128,
            ropeTheta: 5_000_000,
            rmsNormEps: 1e-6,
            maxPositionEmbeddings: 262_144
        )

        public init(
            hiddenSize: Int,
            vocabularySize: Int,
            layerCount: Int,
            intermediateSize: Int,
            attentionHeadCount: Int,
            keyValueHeadCount: Int,
            headDimension: Int,
            ropeTheta: Float,
            rmsNormEps: Float,
            maxPositionEmbeddings: Int
        ) {
            self.hiddenSize = hiddenSize
            self.vocabularySize = vocabularySize
            self.layerCount = layerCount
            self.intermediateSize = intermediateSize
            self.attentionHeadCount = attentionHeadCount
            self.keyValueHeadCount = keyValueHeadCount
            self.headDimension = headDimension
            self.ropeTheta = ropeTheta
            self.rmsNormEps = rmsNormEps
            self.maxPositionEmbeddings = maxPositionEmbeddings
        }
    }

    public struct LoadReport: Sendable {
        public let sourceTensorCount: Int
        public let loadedTensorCount: Int
        public let quantizedModuleCount: Int
        public let denseTensorCount: Int
        public let sourceQuantizationCounts: [String: Int]

        public init(
            sourceTensorCount: Int,
            loadedTensorCount: Int,
            quantizedModuleCount: Int,
            denseTensorCount: Int,
            sourceQuantizationCounts: [String: Int]
        ) {
            self.sourceTensorCount = sourceTensorCount
            self.loadedTensorCount = loadedTensorCount
            self.quantizedModuleCount = quantizedModuleCount
            self.denseTensorCount = denseTensorCount
            self.sourceQuantizationCounts = sourceQuantizationCounts
        }
    }

    public enum Error: LocalizedError, Sendable {
        case invalidCheckpoint(String)
        case missingTensor(String)
        case invalidTensor(name: String, expected: [Int], actual: [Int])
        case missingFile(URL)
        case invalidTokenizer(String)
        case invalidPrompt(String)

        public var errorDescription: String? {
            switch self {
            case let .invalidCheckpoint(message):
                "MiniMax H3 Qwen3-VL checkpoint 無效：\(message)"
            case let .missingTensor(name):
                "MiniMax H3 Qwen3-VL 缺少權重：\(name)"
            case let .invalidTensor(name, expected, actual):
                "MiniMax H3 Qwen3-VL 權重 \(name) shape 不符，預期 \(expected)，實際 \(actual)。"
            case let .missingFile(url):
                "MiniMax H3 Qwen3-VL 缺少檔案：\(url.path)"
            case let .invalidTokenizer(message):
                "MiniMax H3 tokenizer 無效：\(message)"
            case let .invalidPrompt(message):
                "MiniMax H3 prompt 無效：\(message)"
            }
        }
    }

    private struct QuantizedWeight {
        let weights: MLXArray
        let scales: MLXArray
        let biases: MLXArray?
    }

    public let configuration: Configuration
    public let report: LoadReport
    public let vocabularySize: Int
    public let layerCount: Int

    private let tokenizer: Tokenizer
    private let tensors: [String: MLXArray]
    private let quantizedWeights: [String: QuantizedWeight]
    private let tokenEmbedding: MLXArray

    private init(
        configuration: Configuration,
        report: LoadReport,
        tokenizer: Tokenizer,
        tensors: [String: MLXArray],
        quantizedWeights: [String: QuantizedWeight]
    ) throws {
        guard let tokenEmbedding = tensors["model.embed_tokens.weight"] else {
            throw Error.missingTensor("model.embed_tokens.weight")
        }
        self.configuration = configuration
        self.report = report
        self.vocabularySize = configuration.vocabularySize
        self.layerCount = configuration.layerCount
        self.tokenizer = tokenizer
        self.tensors = tensors
        self.quantizedWeights = quantizedWeights
        self.tokenEmbedding = tokenEmbedding
    }

    /// Loads only the language-model tensors from the text-encoder GGUF.
    /// Visual tensors are deliberately not materialized by the text-only
    /// FL2VA path; image/keyframe conditioning remains a separate block.
    public static func load(
        fileURL: URL,
        tokenizerDirectory: URL,
        configuration: Configuration = .minimaxH3,
        layerCount: Int? = nil,
        quantizeLinear: Bool = true,
        useMetalQuantizer: Bool = MiniMaxH3GGUFQuantizedLoader.defaultUseMetalQuantizer
    ) throws -> MiniMaxH3Qwen3VLTextEncoder {
        let effectiveLayerCount = layerCount ?? configuration.layerCount
        guard effectiveLayerCount > 0, effectiveLayerCount <= configuration.layerCount else {
            throw Error.invalidCheckpoint("語言層數必須介於 1 到 \(configuration.layerCount) 之間")
        }
        let effectiveConfiguration = Configuration(
            hiddenSize: configuration.hiddenSize,
            vocabularySize: configuration.vocabularySize,
            layerCount: effectiveLayerCount,
            intermediateSize: configuration.intermediateSize,
            attentionHeadCount: configuration.attentionHeadCount,
            keyValueHeadCount: configuration.keyValueHeadCount,
            headDimension: configuration.headDimension,
            ropeTheta: configuration.ropeTheta,
            rmsNormEps: configuration.rmsNormEps,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings
        )
        let inspection = try GGUFModelLoader.inspect(fileURL: fileURL)
        let modelDescriptors = inspection.tensors.filter { $0.name.hasPrefix("model.") }
        guard !modelDescriptors.isEmpty else {
            throw Error.invalidCheckpoint("找不到 model.* 語言模型張量")
        }

        let layerIndices = Set(modelDescriptors.compactMap { descriptor -> Int? in
            guard descriptor.name.hasPrefix("model.layers.") else { return nil }
            let suffix = descriptor.name.dropFirst("model.layers.".count)
            let digits = suffix.prefix { $0.isNumber }
            return Int(digits)
        })
        guard layerIndices.isSuperset(of: Set(0 ..< effectiveLayerCount)) else {
            throw Error.invalidCheckpoint(
                "語言層索引缺少 0..\(effectiveLayerCount - 1)：\(layerIndices.sorted())"
            )
        }

        let expected = expectedShapes(configuration: effectiveConfiguration)
        let descriptorByName = Dictionary(uniqueKeysWithValues: modelDescriptors.map { ($0.name, $0) })
        let missing = expected.keys.filter { descriptorByName[$0] == nil }.sorted()
        guard missing.isEmpty else {
            throw Error.missingTensor(missing[0])
        }
        let shapeOverrides = try MiniMaxH3GGUFWeightLoader.comfyShapeOverrides(in: inspection)
        let data: Data
        do {
            data = try Data(contentsOf: fileURL, options: [.mappedIfSafe])
        } catch {
            throw Error.invalidCheckpoint("無法讀取 GGUF：\(error.localizedDescription)")
        }

        let tokenizer = try loadTokenizer(
            from: tokenizerDirectory,
            vocabularySize: effectiveConfiguration.vocabularySize
        )
        var tensors: [String: MLXArray] = [:]
        var quantizedWeights: [String: QuantizedWeight] = [:]
        var quantizedCount = 0
        var denseCount = 0

        for name in expected.keys.sorted() {
            let descriptor = descriptorByName[name]!
            guard let byteSize = descriptor.byteSize,
                  let offset = Int(exactly: descriptor.offset),
                  let byteCount = Int(exactly: byteSize) else {
                throw Error.invalidCheckpoint("\(name) 的 GGUF 資料範圍不正確")
            }
            let (dataStart, startOverflow) = inspection.dataOffset.addingReportingOverflow(offset)
            let (dataEnd, endOverflow) = dataStart.addingReportingOverflow(byteCount)
            guard !startOverflow, !endOverflow, dataStart >= 0, dataEnd <= data.count else {
                throw Error.invalidCheckpoint("\(name) 的 GGUF 資料範圍不正確")
            }
            let logicalShape = shapeOverrides[name] ?? descriptor.shape
            guard logicalShape == expected[name]! else {
                throw Error.invalidTensor(
                    name: name,
                    expected: expected[name]!,
                    actual: logicalShape
                )
            }
            let raw = data.subdata(in: dataStart ..< dataEnd)

            if isLinearWeight(name: name) {
                if quantizeLinear {
                    let quantized: MiniMaxH3GGUFQuantizedLoader.QuantizedTensor
                    if useMetalQuantizer,
                       MiniMaxH3GGUFMetalQuantizer.supports(typeCode: descriptor.typeCode) {
                        let metal = try MiniMaxH3GGUFMetalQuantizer.quantize(
                            raw: raw,
                            sourceType: descriptor.typeCode,
                            sourceShape: logicalShape,
                            name: name
                        )
                        quantized = MiniMaxH3GGUFQuantizedLoader.QuantizedTensor(
                            weights: metal.weights,
                            scales: metal.scales,
                            biases: metal.biases,
                            backend: .metal
                        )
                    } else {
                        let dense = try GGUFDequantizer.array(
                            raw: raw,
                            typeCode: descriptor.typeCode,
                            shape: logicalShape,
                            name: name
                        ).asType(.float32)
                        let cpu = MLX.quantized(dense, groupSize: 64, bits: 8)
                        guard let biases = cpu.biases else {
                            throw Error.invalidCheckpoint("\(name) 無法建立 affine INT8 biases")
                        }
                        MLX.eval(cpu.wq, cpu.scales, biases)
                        quantized = MiniMaxH3GGUFQuantizedLoader.QuantizedTensor(
                            weights: cpu.wq,
                            scales: cpu.scales,
                            biases: biases,
                            backend: .cpu
                        )
                    }
                    let prefix = String(name.dropLast(".weight".count))
                    quantizedWeights[prefix] = QuantizedWeight(
                        weights: quantized.weights,
                        scales: quantized.scales,
                        biases: quantized.biases
                    )
                    quantizedCount += 1
                } else {
                    let dense = try GGUFDequantizer.array(
                        raw: raw,
                        typeCode: descriptor.typeCode,
                        shape: logicalShape,
                        name: name
                    ).asType(.float32)
                    MLX.eval(dense)
                    tensors[name] = dense
                    denseCount += 1
                }
            } else {
                let dense = try GGUFDequantizer.array(
                    raw: raw,
                    typeCode: descriptor.typeCode,
                    shape: logicalShape,
                    name: name
                ).asType(.float32)
                MLX.eval(dense)
                tensors[name] = dense
                denseCount += 1
            }
        }

        let sourceCount = inspection.tensors.count
        let report = LoadReport(
            sourceTensorCount: sourceCount,
            loadedTensorCount: expected.count,
            quantizedModuleCount: quantizedCount,
            denseTensorCount: denseCount,
            sourceQuantizationCounts: inspection.quantizationCounts
        )
        return try MiniMaxH3Qwen3VLTextEncoder(
            configuration: effectiveConfiguration,
            report: report,
            tokenizer: tokenizer,
            tensors: tensors,
            quantizedWeights: quantizedWeights
        )
    }

    /// Encodes raw H3 text without a chat template or BOS/EOS insertion.
    public func tokenIDs(for prompt: String) throws -> [Int] {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [151643] }
        let ids = tokenizer.encode(text: trimmed, addSpecialTokens: false)
        guard !ids.isEmpty else { return [151643] }
        guard ids.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw Error.invalidTokenizer("token id 超過 vocab (\(vocabularySize))")
        }
        return ids
    }

    /// Returns `[textLength, 5120]` raw hidden states after all 50 retained
    /// language layers. The final Qwen RMS norm is intentionally not applied.
    public func hiddenStates(for prompt: String) throws -> MLXArray {
        let ids = try tokenIDs(for: prompt)
        let inputIDs = MLXArray(ids.map(Int32.init), [1, ids.count])
        let hidden = try forward(inputIDs: inputIDs)
        let result = hidden[0]
        MLX.eval(result)
        return result
    }

    /// Convenience method for parity and diagnostics with pre-tokenized input.
    public func hiddenStates(forTokenIDs ids: [Int]) throws -> MLXArray {
        try hiddenStates(forTokenIDs: ids, throughLayerCount: nil)
    }

    /// Returns raw hidden states after a selected number of retained layers.
    /// The layer limit is diagnostic-only; the default runs the full checkpoint.
    public func hiddenStates(
        forTokenIDs ids: [Int],
        throughLayerCount: Int?
    ) throws -> MLXArray {
        guard !ids.isEmpty, ids.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw Error.invalidPrompt("token IDs 必須非空且落在 vocab 範圍內")
        }
        let layerLimit = throughLayerCount ?? configuration.layerCount
        guard layerLimit > 0, layerLimit <= configuration.layerCount else {
            throw Error.invalidPrompt("層數必須介於 1 到 \(configuration.layerCount) 之間")
        }
        let inputIDs = MLXArray(ids.map(Int32.init), [1, ids.count])
        let hidden = try forward(inputIDs: inputIDs, layerCount: layerLimit)
        let result = hidden[0]
        MLX.eval(result)
        return result
    }

    private func forward(inputIDs: MLXArray, layerCount: Int? = nil) throws -> MLXArray {
        guard inputIDs.ndim == 2, inputIDs.shape[0] == 1, inputIDs.shape[1] > 0 else {
            throw Error.invalidPrompt("input IDs 必須是 [1, sequence]")
        }
        var hidden = tokenEmbedding[inputIDs]
        for layer in 0 ..< (layerCount ?? configuration.layerCount) {
            let prefix = "model.layers.\(layer)"
            let normalized = try rmsNorm(
                hidden,
                weight: try tensor("\(prefix).input_layernorm.weight")
            )
            let attended = try attention(normalized, prefix: "\(prefix).self_attn")
            hidden = hidden + attended

            let postNorm = try rmsNorm(
                hidden,
                weight: try tensor("\(prefix).post_attention_layernorm.weight")
            )
            let feedForwardOutput = try feedForward(postNorm, prefix: "\(prefix).mlp")
            hidden = hidden + feedForwardOutput
        }
        return hidden
    }

    private func attention(_ input: MLXArray, prefix: String) throws -> MLXArray {
        let batch = input.shape[0]
        let sequence = input.shape[1]
        let query = try linear(input, prefix: "\(prefix).q_proj")
            .reshaped(batch, sequence, configuration.attentionHeadCount, configuration.headDimension)
            .transposed(0, 2, 1, 3)
        let key = try linear(input, prefix: "\(prefix).k_proj")
            .reshaped(batch, sequence, configuration.keyValueHeadCount, configuration.headDimension)
            .transposed(0, 2, 1, 3)
        let value = try linear(input, prefix: "\(prefix).v_proj")
            .reshaped(batch, sequence, configuration.keyValueHeadCount, configuration.headDimension)
            .transposed(0, 2, 1, 3)

        let normalizedQuery = RoPE(
            try rmsNorm(query, weight: try tensor("\(prefix).q_norm.weight")),
            dimensions: configuration.headDimension,
            traditional: false,
            base: configuration.ropeTheta,
            scale: 1,
            offset: 0
        )
        let normalizedKey = RoPE(
            try rmsNorm(key, weight: try tensor("\(prefix).k_norm.weight")),
            dimensions: configuration.headDimension,
            traditional: false,
            base: configuration.ropeTheta,
            scale: 1,
            offset: 0
        )

        let attended = MLXFast.scaledDotProductAttention(
            queries: normalizedQuery,
            keys: normalizedKey,
            values: value,
            scale: 1.0 / Foundation.sqrt(Float(configuration.headDimension)),
            mask: .causal
        )
        let merged = attended.transposed(0, 2, 1, 3)
            .reshaped(batch, sequence, configuration.attentionHeadCount * configuration.headDimension)
        return try linear(merged, prefix: "\(prefix).o_proj")
    }

    private func feedForward(_ input: MLXArray, prefix: String) throws -> MLXArray {
        let gate = try linear(input, prefix: "\(prefix).gate_proj")
        let up = try linear(input, prefix: "\(prefix).up_proj")
        return try linear(MLXNN.silu(gate) * up, prefix: "\(prefix).down_proj")
    }

    private func rmsNorm(_ input: MLXArray, weight: MLXArray) throws -> MLXArray {
        MLXFast.rmsNorm(input, weight: weight.asType(input.dtype), eps: configuration.rmsNormEps)
    }

    private func tensor(_ name: String) throws -> MLXArray {
        guard let value = tensors[name] else { throw Error.missingTensor(name) }
        return value
    }

    private func linear(_ input: MLXArray, prefix: String) throws -> MLXArray {
        if let quantized = quantizedWeights[prefix] {
            return MLX.quantizedMM(
                input,
                quantized.weights,
                scales: quantized.scales,
                biases: quantized.biases,
                transpose: true,
                groupSize: 64,
                bits: 8
            )
        }
        let weight = try tensor("\(prefix).weight")
        return input.matmul(weight.transposed())
    }

    private static func isLinearWeight(name: String) -> Bool {
        name.hasSuffix(".self_attn.q_proj.weight")
            || name.hasSuffix(".self_attn.k_proj.weight")
            || name.hasSuffix(".self_attn.v_proj.weight")
            || name.hasSuffix(".self_attn.o_proj.weight")
            || name.hasSuffix(".mlp.gate_proj.weight")
            || name.hasSuffix(".mlp.up_proj.weight")
            || name.hasSuffix(".mlp.down_proj.weight")
    }

    private static func expectedShapes(configuration: Configuration) -> [String: [Int]] {
        var result: [String: [Int]] = [
            "model.embed_tokens.weight": [configuration.vocabularySize, configuration.hiddenSize]
        ]
        for layer in 0 ..< configuration.layerCount {
            let prefix = "model.layers.\(layer)"
            result["\(prefix).input_layernorm.weight"] = [configuration.hiddenSize]
            result["\(prefix).post_attention_layernorm.weight"] = [configuration.hiddenSize]
            result["\(prefix).self_attn.q_norm.weight"] = [configuration.headDimension]
            result["\(prefix).self_attn.k_norm.weight"] = [configuration.headDimension]
            result["\(prefix).self_attn.q_proj.weight"] = [
                configuration.attentionHeadCount * configuration.headDimension,
                configuration.hiddenSize
            ]
            result["\(prefix).self_attn.k_proj.weight"] = [
                configuration.keyValueHeadCount * configuration.headDimension,
                configuration.hiddenSize
            ]
            result["\(prefix).self_attn.v_proj.weight"] = [
                configuration.keyValueHeadCount * configuration.headDimension,
                configuration.hiddenSize
            ]
            result["\(prefix).self_attn.o_proj.weight"] = [
                configuration.hiddenSize,
                configuration.attentionHeadCount * configuration.headDimension
            ]
            result["\(prefix).mlp.gate_proj.weight"] = [
                configuration.intermediateSize,
                configuration.hiddenSize
            ]
            result["\(prefix).mlp.up_proj.weight"] = [
                configuration.intermediateSize,
                configuration.hiddenSize
            ]
            result["\(prefix).mlp.down_proj.weight"] = [
                configuration.hiddenSize,
                configuration.intermediateSize
            ]
        }
        return result
    }

    private static func loadTokenizer(
        from directory: URL,
        vocabularySize: Int
    ) throws -> Tokenizer {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let dataURL = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw Error.missingFile(configURL)
        }
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            throw Error.missingFile(dataURL)
        }
        do {
            let config = try JSONDecoder().decode(Config.self, from: Data(contentsOf: configURL))
            let data = try JSONDecoder().decode(Config.self, from: Data(contentsOf: dataURL))
            let tokenizer = try AutoTokenizer.from(
                tokenizerConfig: config,
                tokenizerData: data
            )
            guard tokenizer.convertTokenToId("<|endoftext|>") == 151643 else {
                throw Error.invalidTokenizer("pad token id 不是 151643")
            }
            _ = vocabularySize
            return tokenizer
        } catch let error as Error {
            throw error
        } catch {
            throw Error.invalidTokenizer(error.localizedDescription)
        }
    }
}
