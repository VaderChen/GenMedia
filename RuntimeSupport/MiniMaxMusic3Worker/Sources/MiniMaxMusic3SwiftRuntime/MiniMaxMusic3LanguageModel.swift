import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct MiniMaxMusic3LanguageModelLoadReport: Sendable {
    public let tensorCount: Int
    public let quantizedModuleCount: Int
    public let shardNames: [String]

    public init(tensorCount: Int, quantizedModuleCount: Int, shardNames: [String]) {
        self.tensorCount = tensorCount
        self.quantizedModuleCount = quantizedModuleCount
        self.shardNames = shardNames
    }
}

public enum MiniMaxMusic3LanguageModelError: LocalizedError, Sendable {
    case invalidConfiguration(URL)
    case missingFile(URL)
    case missingWeights([String])
    case unexpectedWeights([String])
    case weightShapeMismatch(name: String, expected: [Int], actual: [Int])
    case invalidInput(String)
    case unsupportedInterface(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidConfiguration(url):
            "無法解析 Qwen3 設定：\(url.path)"
        case let .missingFile(url):
            "缺少語言模型必要檔案：\(url.path)"
        case let .missingWeights(names):
            "語言模型缺少權重：\(names.joined(separator: ", "))"
        case let .unexpectedWeights(names):
            "語言模型收到重複或未使用權重：\(names.joined(separator: ", "))"
        case let .weightShapeMismatch(name, expected, actual):
            "語言模型權重 \(name) 形狀不符，預期 \(expected)，實際 \(actual)。"
        case let .invalidInput(message):
            "語言模型輸入無效：\(message)"
        case let .unsupportedInterface(message):
            "Qwen3 公開介面不支援 MiniMax Music 3 所需操作：\(message)"
        }
    }
}

public final class MiniMaxMusic3LanguageModel {
    public let model: Qwen3Model
    public let report: MiniMaxMusic3LanguageModelLoadReport
    public let hiddenSize: Int
    public let vocabularySize: Int
    public let layerCount: Int

    private let originalTokenEmbedding: Embedding
    private let tokenEmbeddingWeight: MLXArray
    private let languageHeadWeight: MLXArray

    public init(modelDirectory: URL) throws {
        let configurationURL = modelDirectory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configurationURL) else {
            throw MiniMaxMusic3LanguageModelError.missingFile(configurationURL)
        }
        let configuration: Qwen3Configuration
        do {
            configuration = try JSONDecoder().decode(Qwen3Configuration.self, from: data)
        } catch {
            throw MiniMaxMusic3LanguageModelError.invalidConfiguration(configurationURL)
        }

        let model = Qwen3Model(configuration)
        let report = try MiniMaxMusic3LanguageModelWeightLoader.load(
            model: model,
            from: modelDirectory
        )
        guard let originalTokenEmbedding = model.model.children()[unwrapping: "embed_tokens"] as? Embedding else {
            throw MiniMaxMusic3LanguageModelError.unsupportedInterface(
                "Qwen3ModelInner 沒有可替換的 embed_tokens。"
            )
        }
        let parameters = model.parameters().flattened()
        guard let tokenEmbeddingWeight = parameters.first(where: {
            $0.0 == "model.embed_tokens.weight"
        })?.1 else {
            throw MiniMaxMusic3LanguageModelError.missingWeights(["model.embed_tokens.weight"])
        }
        guard let languageHeadWeight = parameters.first(where: {
            $0.0 == "lm_head.weight"
        })?.1 else {
            throw MiniMaxMusic3LanguageModelError.missingWeights(["lm_head.weight"])
        }
        self.model = model
        self.report = report
        self.hiddenSize = tokenEmbeddingWeight.shape[1]
        self.vocabularySize = model.vocabularySize
        self.layerCount = model.kvHeads.count
        self.originalTokenEmbedding = originalTokenEmbedding
        self.tokenEmbeddingWeight = tokenEmbeddingWeight
        self.languageHeadWeight = languageHeadWeight
    }

    public func hiddenStates(
        _ inputIDs: MLXArray,
        cache: [KVCache]? = nil
    ) throws -> MLXArray {
        guard inputIDs.ndim == 2, inputIDs.shape[0] > 0, inputIDs.shape[1] > 0 else {
            throw MiniMaxMusic3LanguageModelError.invalidInput(
                "input IDs 必須是非空的 [batch, sequence]。"
            )
        }
        return model.model(inputIDs, cache: cache)
    }

    public func hiddenStates(
        inputEmbeddings: MLXArray,
        cache: [KVCache]? = nil
    ) throws -> MLXArray {
        guard inputEmbeddings.ndim == 3,
              inputEmbeddings.shape[0] > 0,
              inputEmbeddings.shape[1] > 0,
              inputEmbeddings.shape[2] == hiddenSize else {
            throw MiniMaxMusic3LanguageModelError.invalidInput(
                "input embeddings 必須是非空的 [batch, sequence, \(hiddenSize)]。"
            )
        }

        let batch = inputEmbeddings.shape[0]
        let length = inputEmbeddings.shape[1]
        let injectedIDs = MLXArray(
            Array(0..<(batch * length)),
            [batch, length]
        )
        let injectedEmbedding = Embedding(
            weight: inputEmbeddings.reshaped(batch * length, hiddenSize)
        )
        model.model.update(
            modules: ModuleChildren.unflattened(["embed_tokens": injectedEmbedding])
        )
        defer {
            model.model.update(
                modules: ModuleChildren.unflattened([
                    "embed_tokens": originalTokenEmbedding
                ])
            )
        }
        return model.model(injectedIDs, cache: cache)
    }

    public func tokenEmbeddings(for tokenIDs: MLXArray) throws -> MLXArray {
        guard tokenIDs.ndim > 0, tokenIDs.size > 0 else {
            throw MiniMaxMusic3LanguageModelError.invalidInput(
                "token IDs 必須是非空陣列。"
            )
        }
        return tokenEmbeddingWeight[tokenIDs]
    }

    public func newCache() -> [KVCache] {
        model.newCache(parameters: nil)
    }

    public func logits(
        for inputIDs: MLXArray,
        cache: [KVCache]? = nil
    ) throws -> MLXArray {
        guard inputIDs.ndim == 2, inputIDs.shape[0] > 0, inputIDs.shape[1] > 0 else {
            throw MiniMaxMusic3LanguageModelError.invalidInput(
                "input IDs 必須是非空的 [batch, sequence]。"
            )
        }
        return model(inputIDs, cache: cache)
    }

    public func logits(forHiddenStates hiddenStates: MLXArray) throws -> MLXArray {
        guard hiddenStates.ndim > 0,
              hiddenStates.size > 0,
              hiddenStates.shape.last == hiddenSize else {
            throw MiniMaxMusic3LanguageModelError.invalidInput(
                "hidden states 的最後維度必須是 \(hiddenSize)。"
            )
        }
        return matmul(hiddenStates, languageHeadWeight.T)
    }
}

public enum MiniMaxMusic3LanguageModelWeightLoader {
    public static func load(
        model: Qwen3Model,
        from modelDirectory: URL
    ) throws -> MiniMaxMusic3LanguageModelLoadReport {
        var quantizedPaths = Set<String>()
        quantize(
            model: model,
            groupSize: 64,
            bits: 4,
            mode: .affine,
            filter: { path, module in
                guard module is Linear else { return false }
                guard path.hasPrefix("model.layers.") else { return false }
                guard path.contains(".self_attn.q_proj")
                    || path.contains(".self_attn.k_proj")
                    || path.contains(".self_attn.v_proj")
                    || path.contains(".self_attn.o_proj")
                    || path.contains(".mlp.gate_proj")
                    || path.contains(".mlp.up_proj")
                    || path.contains(".mlp.down_proj") else {
                    return false
                }
                quantizedPaths.insert(path)
                return true
            }
        )

        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        let shardNames = try languageModelShardNames(from: modelDirectory)
        var converted: [String: MLXArray] = [:]

        for shardName in shardNames {
            let shardURL = modelDirectory.appendingPathComponent(shardName)
            let arrays = try MLX.loadArrays(url: shardURL)
            for (sourceKey, sourceValue) in arrays where sourceKey.hasPrefix("language_model.") {
                let key = String(sourceKey.dropFirst("language_model.".count))
                guard expected[key] != nil else { continue }
                guard converted[key] == nil else {
                    throw MiniMaxMusic3LanguageModelError.unexpectedWeights([key])
                }
                guard sourceValue.shape == expected[key]!.shape else {
                    throw MiniMaxMusic3LanguageModelError.weightShapeMismatch(
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
            throw MiniMaxMusic3LanguageModelError.missingWeights(missing)
        }
        let unexpected = convertedKeys.subtracting(expectedKeys).sorted()
        guard unexpected.isEmpty else {
            throw MiniMaxMusic3LanguageModelError.unexpectedWeights(unexpected)
        }

        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
        return MiniMaxMusic3LanguageModelLoadReport(
            tensorCount: converted.count,
            quantizedModuleCount: quantizedPaths.count,
            shardNames: shardNames
        )
    }

    private static func languageModelShardNames(from modelDirectory: URL) throws -> [String] {
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
                throw MiniMaxMusic3LanguageModelError.missingFile(indexURL)
            }
            return names
        }

        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        let names = Set(
            index.weightMap.compactMap { key, file in
                key.hasPrefix("language_model.") ? file : nil
            }
        ).sorted()
        guard !names.isEmpty else {
            throw MiniMaxMusic3LanguageModelError.missingWeights(["language_model.*"])
        }
        for name in names {
            let url = modelDirectory.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MiniMaxMusic3LanguageModelError.missingFile(url)
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
