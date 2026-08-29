import Foundation
import MLX
import MLXNN

public struct LTXGemma3TextConfiguration: Decodable, Sendable, Equatable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let headDimension: Int
    public let rmsNormEpsilon: Float
    public let vocabularySize: Int
    public let keyValueHeads: Int
    public let ropeTheta: Float
    public let localRopeBaseFrequency: Float
    public let queryPreAttentionScalar: Float
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int

    public init(
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        headDimension: Int,
        rmsNormEpsilon: Float,
        vocabularySize: Int,
        keyValueHeads: Int,
        ropeTheta: Float = 1_000_000,
        localRopeBaseFrequency: Float = 10_000,
        queryPreAttentionScalar: Float = 256,
        slidingWindow: Int = 512,
        slidingWindowPattern: Int = 6,
        maxPositionEmbeddings: Int = 32_768
    ) throws {
        guard hiddenSize > 0,
              hiddenLayers > 0,
              intermediateSize > 0,
              attentionHeads > 0,
              headDimension > 0,
              rmsNormEpsilon > 0,
              vocabularySize > 0,
              keyValueHeads > 0,
              attentionHeads.isMultiple(of: keyValueHeads),
              slidingWindow > 0,
              slidingWindowPattern > 0,
              maxPositionEmbeddings > 0 else {
            throw LTXGemma3TextEncoderError.invalidConfiguration(
                "Gemma3 維度、head 設定或位置設定無效。"
            )
        }
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimension = headDimension
        self.rmsNormEpsilon = rmsNormEpsilon
        self.vocabularySize = vocabularySize
        self.keyValueHeads = keyValueHeads
        self.ropeTheta = ropeTheta
        self.localRopeBaseFrequency = localRopeBaseFrequency
        self.queryPreAttentionScalar = queryPreAttentionScalar
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.maxPositionEmbeddings = maxPositionEmbeddings
    }

    public static func load(from modelDirectory: URL) throws -> Self {
        let url = modelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LTXGemma3TextEncoderError.missingFile(url)
        }
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimension = "head_dim"
        case rmsNormEpsilon = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case keyValueHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case localRopeBaseFrequency = "rope_local_base_freq"
        case queryPreAttentionScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: CodingKeys.self)
        let container: KeyedDecodingContainer<CodingKeys>
        if outer.contains(.textConfig) {
            container = try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            container = outer
        }
        try self.init(
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            headDimension: try container.decode(Int.self, forKey: .headDimension),
            rmsNormEpsilon: try container.decodeIfPresent(Float.self, forKey: .rmsNormEpsilon) ?? 1e-6,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            keyValueHeads: try container.decode(Int.self, forKey: .keyValueHeads),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000,
            localRopeBaseFrequency: try container.decodeIfPresent(Float.self, forKey: .localRopeBaseFrequency) ?? 10_000,
            queryPreAttentionScalar: try container.decodeIfPresent(Float.self, forKey: .queryPreAttentionScalar) ?? 256,
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512,
            slidingWindowPattern: try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6,
            maxPositionEmbeddings: try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32_768
        )
    }
}

public enum LTXGemma3TextEncoderError: LocalizedError, Sendable, Equatable {
    case invalidConfiguration(String)
    case invalidInputShape([Int])
    case missingFile(URL)
    case missingWeights([String])
    case duplicateWeight(String)
    case weightShapeMismatch(name: String, expected: [Int], actual: [Int])
    case unsupportedWeightIndex(URL)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "LTX Gemma3 設定無效：\(message)"
        case .invalidInputShape(let shape):
            return "LTX Gemma3 token 輸入 shape 無效：\(shape)"
        case .missingFile(let url):
            return "找不到 LTX Gemma3 檔案：\(url.path)"
        case .missingWeights(let keys):
            return "LTX Gemma3 缺少權重：\(keys.prefix(12).joined(separator: "、"))"
        case .duplicateWeight(let key):
            return "LTX Gemma3 權重重複：\(key)"
        case .weightShapeMismatch(let name, let expected, let actual):
            return "LTX Gemma3 權重 \(name) shape 不一致：預期 \(expected)，實際 \(actual)。"
        case .unsupportedWeightIndex(let url):
            return "無法讀取 Gemma3 權重索引：\(url.path)"
        }
    }
}

final class LTXGemmaRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let epsilon: Float

    init(dimensions: Int, epsilon: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.epsilon = epsilon
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(input, weight: 1 + weight, eps: epsilon)
    }
}

final class LTXGemma3Attention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: LTXGemmaRMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: LTXGemmaRMSNorm

    let headCount: Int
    let keyValueHeadCount: Int
    let headDimension: Int
    let scale: Float
    let rope: RoPE

    init(configuration: LTXGemma3TextConfiguration, layerIndex: Int) {
        let hiddenSize = configuration.hiddenSize
        self.headCount = configuration.attentionHeads
        self.keyValueHeadCount = configuration.keyValueHeads
        self.headDimension = configuration.headDimension
        self.scale = pow(configuration.queryPreAttentionScalar, -0.5)
        let isSliding = (layerIndex + 1).isMultiple(of: configuration.slidingWindowPattern) == false
        self.rope = RoPE(
            dimensions: configuration.headDimension,
            traditional: false,
            base: isSliding ? configuration.localRopeBaseFrequency : configuration.ropeTheta
        )
        self._queryProjection = ModuleInfo(
            wrappedValue: Linear(hiddenSize, headCount * headDimension, bias: false),
            key: "q_proj"
        )
        self._keyProjection = ModuleInfo(
            wrappedValue: Linear(hiddenSize, keyValueHeadCount * headDimension, bias: false),
            key: "k_proj"
        )
        self._valueProjection = ModuleInfo(
            wrappedValue: Linear(hiddenSize, keyValueHeadCount * headDimension, bias: false),
            key: "v_proj"
        )
        self._outputProjection = ModuleInfo(
            wrappedValue: Linear(headCount * headDimension, hiddenSize, bias: false),
            key: "o_proj"
        )
        self._queryNorm = ModuleInfo(
            wrappedValue: LTXGemmaRMSNorm(
                dimensions: headDimension,
                epsilon: configuration.rmsNormEpsilon
            ),
            key: "q_norm"
        )
        self._keyNorm = ModuleInfo(
            wrappedValue: LTXGemmaRMSNorm(
                dimensions: headDimension,
                epsilon: configuration.rmsNormEpsilon
            ),
            key: "k_norm"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray?) -> MLXArray {
        let batch = input.shape[0]
        let sequenceLength = input.shape[1]
        var queries = queryProjection(input)
            .reshaped(batch, sequenceLength, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(input)
            .reshaped(batch, sequenceLength, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batch, sequenceLength, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)
        queries = rope(queryNorm(queries))
        keys = rope(keyNorm(keys))
        let attention = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask.map { .array($0) } ?? .none
        )
        return outputProjection(
            attention.transposed(0, 2, 1, 3).reshaped(batch, sequenceLength, headCount * headDimension)
        )
    }
}

final class LTXGemma3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear

    init(configuration: LTXGemma3TextConfiguration) {
        self._gateProjection = ModuleInfo(
            wrappedValue: Linear(
                configuration.hiddenSize,
                configuration.intermediateSize,
                bias: false
            ),
            key: "gate_proj"
        )
        self._downProjection = ModuleInfo(
            wrappedValue: Linear(
                configuration.intermediateSize,
                configuration.hiddenSize,
                bias: false
            ),
            key: "down_proj"
        )
        self._upProjection = ModuleInfo(
            wrappedValue: Linear(
                configuration.hiddenSize,
                configuration.intermediateSize,
                bias: false
            ),
            key: "up_proj"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(geluApproximate(gateProjection(input)) * upProjection(input))
    }
}

final class LTXGemma3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: LTXGemma3Attention
    @ModuleInfo var mlp: LTXGemma3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LTXGemmaRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LTXGemmaRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: LTXGemmaRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: LTXGemmaRMSNorm

    init(configuration: LTXGemma3TextConfiguration, layerIndex: Int) {
        self._attention = ModuleInfo(
            wrappedValue: LTXGemma3Attention(configuration: configuration, layerIndex: layerIndex),
            key: "self_attn"
        )
        self._mlp = ModuleInfo(
            wrappedValue: LTXGemma3MLP(configuration: configuration),
            key: "mlp"
        )
        self._inputLayerNorm = ModuleInfo(
            wrappedValue: LTXGemmaRMSNorm(
                dimensions: configuration.hiddenSize,
                epsilon: configuration.rmsNormEpsilon
            ),
            key: "input_layernorm"
        )
        self._postAttentionLayerNorm = ModuleInfo(
            wrappedValue: LTXGemmaRMSNorm(
                dimensions: configuration.hiddenSize,
                epsilon: configuration.rmsNormEpsilon
            ),
            key: "post_attention_layernorm"
        )
        self._preFeedforwardLayerNorm = ModuleInfo(
            wrappedValue: LTXGemmaRMSNorm(
                dimensions: configuration.hiddenSize,
                epsilon: configuration.rmsNormEpsilon
            ),
            key: "pre_feedforward_layernorm"
        )
        self._postFeedforwardLayerNorm = ModuleInfo(
            wrappedValue: LTXGemmaRMSNorm(
                dimensions: configuration.hiddenSize,
                epsilon: configuration.rmsNormEpsilon
            ),
            key: "post_feedforward_layernorm"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray?) -> MLXArray {
        let attentionOutput = attention(inputLayerNorm(input), mask: mask)
        let afterAttention = input + postAttentionLayerNorm(attentionOutput)
        let feedForwardOutput = mlp(preFeedforwardLayerNorm(afterAttention))
        return afterAttention + postFeedforwardLayerNorm(feedForwardOutput)
    }
}

public final class LTXGemma3AllLayerModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [LTXGemma3TransformerBlock]
    @ModuleInfo var norm: LTXGemmaRMSNorm

    public let configuration: LTXGemma3TextConfiguration

    public init(configuration: LTXGemma3TextConfiguration) {
        self.configuration = configuration
        self._embedTokens = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: configuration.vocabularySize,
                dimensions: configuration.hiddenSize
            ),
            key: "embed_tokens"
        )
        self._layers = ModuleInfo(
            wrappedValue: (0..<configuration.hiddenLayers).map { index in
                LTXGemma3TransformerBlock(configuration: configuration, layerIndex: index)
            },
            key: "layers"
        )
        self._norm = ModuleInfo(
            wrappedValue: LTXGemmaRMSNorm(
                dimensions: configuration.hiddenSize,
                epsilon: configuration.rmsNormEpsilon
            ),
            key: "norm"
        )
        super.init()
    }

    public func allHiddenStates(
        tokenIDs: MLXArray,
        attentionMask: MLXArray? = nil
    ) throws -> [MLXArray] {
        guard tokenIDs.ndim == 2 else {
            throw LTXGemma3TextEncoderError.invalidInputShape(tokenIDs.shape)
        }
        if let attentionMask,
           attentionMask.ndim != 2
                || attentionMask.shape[0] != tokenIDs.shape[0]
                || attentionMask.shape[1] != tokenIDs.shape[1] {
            throw LTXGemma3TextEncoderError.invalidInputShape(attentionMask.shape)
        }
        let sequenceLength = tokenIDs.shape[1]
        var hidden = embedTokens(tokenIDs)
        let scale = MLXArray(sqrt(Float(configuration.hiddenSize)), dtype: .bfloat16)
        hidden = hidden * scale.asType(hidden.dtype)

        let causal = triu(
            MLXArray.full(
                [sequenceLength, sequenceLength],
                values: MLXArray(-1e9),
                dtype: .bfloat16
            ),
            k: 1
        )
        let padding = attentionMask.map {
            (1 - $0[0..., .newAxis, .newAxis, 0...].asType(.bfloat16)) * -1e9
        }
        let mask = causal[.newAxis, .newAxis, 0..., 0...] + (padding ?? MLXArray(0))

        var states = [MLXArray]()
        states.reserveCapacity(layers.count + 1)
        states.append(hidden)
        for layer in layers {
            hidden = layer(hidden, mask: mask)
            states.append(hidden)
        }
        return states
    }
}
