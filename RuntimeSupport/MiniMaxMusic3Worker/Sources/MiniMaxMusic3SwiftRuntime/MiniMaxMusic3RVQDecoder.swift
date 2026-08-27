import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3RVQDecoderConfiguration: Equatable, Sendable {
    public let hiddenSize: Int
    public let numberOfLayers: Int
    public let numberOfAttentionHeads: Int
    public let intermediateSize: Int
    public let audioVocabularySize: Int
    public let numberOfCodebooks: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEpsilon: Float

    public init(
        hiddenSize: Int = 4_096,
        numberOfLayers: Int = 4,
        numberOfAttentionHeads: Int = 16,
        intermediateSize: Int = 6_144,
        audioVocabularySize: Int = 1_024,
        numberOfCodebooks: Int = 8,
        maxPositionEmbeddings: Int = 16,
        rmsNormEpsilon: Float = 1e-6
    ) {
        self.hiddenSize = hiddenSize
        self.numberOfLayers = numberOfLayers
        self.numberOfAttentionHeads = numberOfAttentionHeads
        self.intermediateSize = intermediateSize
        self.audioVocabularySize = audioVocabularySize
        self.numberOfCodebooks = numberOfCodebooks
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEpsilon = rmsNormEpsilon
    }

    public static let music3 = Self()

    public func validate() throws {
        guard hiddenSize > 0,
              numberOfLayers > 0,
              numberOfAttentionHeads > 0,
              intermediateSize > 0,
              audioVocabularySize > 0,
              numberOfCodebooks > 1,
              maxPositionEmbeddings > 0,
              hiddenSize.isMultiple(of: numberOfAttentionHeads),
              rmsNormEpsilon > 0 else {
            throw MiniMaxMusic3RVQDecoderError.invalidConfiguration
        }
    }

    public static func load(from modelDirectory: URL) throws -> Self {
        let url = modelDirectory.appendingPathComponent("config.json")
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw MiniMaxMusic3RVQDecoderError.missingFile(url)
        }
        do {
            let source = try JSONDecoder().decode(ModelConfigurationFile.self, from: data)
            let configuration = Self(
                hiddenSize: source.hiddenSize,
                numberOfLayers: source.numberOfLayers,
                numberOfAttentionHeads: source.numberOfAttentionHeads,
                intermediateSize: source.intermediateSize,
                audioVocabularySize: source.audioVocabularySize,
                numberOfCodebooks: source.numberOfCodebooks,
                maxPositionEmbeddings: source.maxPositionEmbeddings,
                rmsNormEpsilon: source.rmsNormEpsilon
            )
            try configuration.validate()
            return configuration
        } catch let error as MiniMaxMusic3RVQDecoderError {
            throw error
        } catch {
            throw MiniMaxMusic3RVQDecoderError.invalidConfigurationFile(url)
        }
    }

    private struct ModelConfigurationFile: Decodable {
        let hiddenSize: Int
        let numberOfLayers: Int
        let numberOfAttentionHeads: Int
        let intermediateSize: Int
        let audioVocabularySize: Int
        let numberOfCodebooks: Int
        let maxPositionEmbeddings: Int
        let rmsNormEpsilon: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numberOfLayers = "depth_num_layers"
            case numberOfAttentionHeads = "depth_num_heads"
            case intermediateSize = "depth_intermediate_size"
            case audioVocabularySize = "audio_vocab_size"
            case numberOfCodebooks = "num_codebooks"
            case maxPositionEmbeddings = "depth_max_position_embeddings"
            case rmsNormEpsilon = "rms_norm_eps"
        }
    }
}

public enum MiniMaxMusic3RVQDecoderError: LocalizedError, Sendable {
    case invalidConfiguration
    case invalidConfigurationFile(URL)
    case missingFile(URL)
    case invalidInput(String)
    case missingWeights([String])
    case unexpectedWeights([String])
    case weightShapeMismatch(name: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            "MiniMax Music 3 RVQ decoder 設定無效。"
        case let .invalidConfigurationFile(url):
            "無法解析 RVQ decoder 設定：\(url.path)"
        case let .missingFile(url):
            "缺少 RVQ decoder 必要檔案：\(url.path)"
        case let .invalidInput(message):
            "RVQ decoder 輸入無效：\(message)"
        case let .missingWeights(names):
            "RVQ decoder 缺少權重：\(names.joined(separator: ", "))"
        case let .unexpectedWeights(names):
            "RVQ decoder 收到重複或未使用權重：\(names.joined(separator: ", "))"
        case let .weightShapeMismatch(name, expected, actual):
            "RVQ decoder 權重 \(name) 形狀不符，預期 \(expected)，實際 \(actual)。"
        }
    }
}

private final class MiniMaxMusic3RVQDepthAttention: Module {
    @ModuleInfo(key: "to_q") private var query: Linear
    @ModuleInfo(key: "to_k") private var key: Linear
    @ModuleInfo(key: "to_v") private var value: Linear
    @ModuleInfo(key: "to_out") private var output: Linear

    private let numberOfHeads: Int
    private let headDimension: Int
    private let hiddenSize: Int

    init(configuration: MiniMaxMusic3RVQDecoderConfiguration) {
        self.numberOfHeads = configuration.numberOfAttentionHeads
        self.headDimension = configuration.hiddenSize / configuration.numberOfAttentionHeads
        self.hiddenSize = configuration.hiddenSize
        self._query.wrappedValue = QuantizedLinear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: false,
            groupSize: 64,
            bits: 4
        )
        self._key.wrappedValue = QuantizedLinear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: false,
            groupSize: 64,
            bits: 4
        )
        self._value.wrappedValue = QuantizedLinear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: false,
            groupSize: 64,
            bits: 4
        )
        self._output.wrappedValue = QuantizedLinear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: false,
            groupSize: 64,
            bits: 4
        )
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let batch = hiddenStates.shape[0]
        let length = hiddenStates.shape[1]
        let shape = [batch, length, numberOfHeads, headDimension]
        let queries = query(hiddenStates)
            .reshaped(shape)
            .transposed(0, 2, 1, 3)
        let keys = key(hiddenStates)
            .reshaped(shape)
            .transposed(0, 2, 1, 3)
        let values = value(hiddenStates)
            .reshaped(shape)
            .transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: pow(Float(headDimension), -0.5),
            mask: .causal
        )
        return output(
            attended.transposed(0, 2, 1, 3).reshaped(batch, length, hiddenSize)
        )
    }
}

private final class MiniMaxMusic3RVQDecoderBlock: Module {
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "attn") private var attention: MiniMaxMusic3RVQDepthAttention
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(configuration: MiniMaxMusic3RVQDecoderConfiguration) {
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        self._attention.wrappedValue = MiniMaxMusic3RVQDepthAttention(
            configuration: configuration
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        self._gateProjection.wrappedValue = QuantizedLinear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: false,
            groupSize: 64,
            bits: 4
        )
        self._upProjection.wrappedValue = QuantizedLinear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: false,
            groupSize: 64,
            bits: 4
        )
        self._downProjection.wrappedValue = QuantizedLinear(
            configuration.intermediateSize,
            configuration.hiddenSize,
            bias: false,
            groupSize: 64,
            bits: 4
        )
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let attended = hiddenStates + attention(inputLayerNorm(hiddenStates))
        let normalized = postAttentionLayerNorm(attended)
        let gated = silu(gateProjection(normalized)) * upProjection(normalized)
        return attended + downProjection(gated)
    }
}

public enum MiniMaxMusic3RVQCodebookLayout {
    public static func residualEmbeddingIDs(
        for frameCodes: [[Int32]],
        audioVocabularySize: Int = 1_024
    ) throws -> [[Int32]] {
        guard audioVocabularySize > 0,
              !frameCodes.isEmpty,
              frameCodes.allSatisfy({ $0.count > 1 }) else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "frame codes 必須是非空且包含 semantic code 與 residual codebooks。"
            )
        }
        let residualCodebookCount = frameCodes[0].count - 1
        guard frameCodes.allSatisfy({ $0.count == residualCodebookCount + 1 }) else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "frame codes 的 codebook 數量必須一致。"
            )
        }
        return frameCodes.map { codes in
            codes.dropFirst().enumerated().map { index, code in
                code + Int32(index * audioVocabularySize)
            }
        }
    }
}

public enum MiniMaxMusic3RVQEmbedding {
    public static func combine(
        semanticEmbedding: MLXArray,
        residualEmbeddings: MLXArray,
        numberOfCodebooks: Int
    ) throws -> MLXArray {
        guard numberOfCodebooks > 1,
              semanticEmbedding.ndim == 3,
              residualEmbeddings.ndim == 3,
              semanticEmbedding.shape[1] == 1,
              residualEmbeddings.shape[1] == numberOfCodebooks - 1,
              semanticEmbedding.shape[0] == residualEmbeddings.shape[0],
              semanticEmbedding.shape[2] == residualEmbeddings.shape[2] else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "semantic 與 residual embedding shape 不符合 [batch, 1/7, hidden]。"
            )
        }
        return (
            semanticEmbedding + residualEmbeddings.sum(axis: 1, keepDims: true)
        ) * pow(Float(numberOfCodebooks), -0.5)
    }
}

public final class MiniMaxMusic3RVQDepthDecoder: Module {
    public let configuration: MiniMaxMusic3RVQDecoderConfiguration

    @ModuleInfo(key: "audio_embeddings") private var audioEmbeddings: Embedding
    @ModuleInfo(key: "projection") private var projection: Linear
    @ModuleInfo(key: "pos_embedding") private var positionEmbedding: Embedding
    @ModuleInfo(key: "layers") private var layers: [MiniMaxMusic3RVQDecoderBlock]
    @ModuleInfo(key: "norm") private var norm: RMSNorm
    @ModuleInfo(key: "audio_heads") private var audioHeads: [Linear]

    public init(
        configuration: MiniMaxMusic3RVQDecoderConfiguration = .music3
    ) {
        precondition((try? configuration.validate()) != nil)
        self.configuration = configuration
        self._audioEmbeddings.wrappedValue = Embedding(
            embeddingCount: configuration.audioVocabularySize
                * (configuration.numberOfCodebooks - 1),
            dimensions: configuration.hiddenSize
        )
        self._projection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: false
        )
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: configuration.maxPositionEmbeddings,
            dimensions: configuration.hiddenSize
        )
        self._layers.wrappedValue = (0..<configuration.numberOfLayers).map { _ in
            MiniMaxMusic3RVQDecoderBlock(configuration: configuration)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEpsilon
        )
        self._audioHeads.wrappedValue = (0..<(configuration.numberOfCodebooks - 1)).map { _ in
            Linear(configuration.hiddenSize, configuration.audioVocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputsEmbeds: MLXArray) throws -> MLXArray {
        guard inputsEmbeds.ndim == 3,
              inputsEmbeds.shape[2] == configuration.hiddenSize else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "inputs_embeds 必須是 [batch, length, \(configuration.hiddenSize)]。"
            )
        }
        let length = inputsEmbeds.shape[1]
        guard length > 0, length <= configuration.maxPositionEmbeddings else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "depth sequence 超過 max_position_embeddings。"
            )
        }
        let positions = MLXArray.arange(length, dtype: .int32)
        var hiddenStates = inputsEmbeds + positionEmbedding(positions)[.newAxis, 0..., 0...]
        for layer in layers {
            hiddenStates = layer(hiddenStates)
        }
        return norm(hiddenStates)
    }

    public func logits(for hiddenStates: MLXArray) throws -> [MLXArray] {
        guard hiddenStates.ndim == 3,
              hiddenStates.shape[2] == configuration.hiddenSize else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "hidden states 必須是 [batch, length, \(configuration.hiddenSize)]。"
            )
        }
        return audioHeads.map { $0(hiddenStates).asType(.float32) }
    }

    public func project(_ hiddenStates: MLXArray) throws -> MLXArray {
        guard hiddenStates.ndim > 0,
              hiddenStates.shape.last == configuration.hiddenSize else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "projection 輸入的最後維度必須是 \(configuration.hiddenSize)。"
            )
        }
        return projection(hiddenStates)
    }

    public func residualEmbedding(
        for codes: MLXArray,
        codebookIndex: Int
    ) throws -> MLXArray {
        guard codes.ndim == 1,
              codes.shape[0] > 0,
              codebookIndex >= 0,
              codebookIndex < configuration.numberOfCodebooks - 1 else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "residual code 與 codebook index 不符合設定。"
            )
        }
        let offset = Int32(codebookIndex * configuration.audioVocabularySize)
        return audioEmbeddings(codes + offset)
    }

    public func embedAudioFrame(
        semanticEmbedding: MLXArray,
        residualCodes: MLXArray
    ) throws -> MLXArray {
        guard residualCodes.ndim == 2,
              residualCodes.shape[1] == configuration.numberOfCodebooks - 1,
              semanticEmbedding.ndim == 3,
              semanticEmbedding.shape[1] == 1,
              semanticEmbedding.shape[0] == residualCodes.shape[0],
              semanticEmbedding.shape[2] == configuration.hiddenSize else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "semantic embedding 或 residual codes shape 不符合。"
            )
        }
        let offsets = MLXArray(
            (0..<(configuration.numberOfCodebooks - 1)).map {
                Int32($0 * configuration.audioVocabularySize)
            },
            [1, configuration.numberOfCodebooks - 1]
        )
        let residualEmbeddings = audioEmbeddings(residualCodes + offsets)
        return try MiniMaxMusic3RVQEmbedding.combine(
            semanticEmbedding: semanticEmbedding,
            residualEmbeddings: residualEmbeddings,
            numberOfCodebooks: configuration.numberOfCodebooks
        )
    }

    public func residualEmbeddings(for residualCodes: MLXArray) throws -> MLXArray {
        guard residualCodes.ndim == 2,
              residualCodes.shape[1] == configuration.numberOfCodebooks - 1 else {
            throw MiniMaxMusic3RVQDecoderError.invalidInput(
                "residual codes shape 不符合。"
            )
        }
        let offsets = MLXArray(
            (0..<(configuration.numberOfCodebooks - 1)).map {
                Int32($0 * configuration.audioVocabularySize)
            },
            [1, configuration.numberOfCodebooks - 1]
        )
        return audioEmbeddings(residualCodes + offsets)
    }
}
