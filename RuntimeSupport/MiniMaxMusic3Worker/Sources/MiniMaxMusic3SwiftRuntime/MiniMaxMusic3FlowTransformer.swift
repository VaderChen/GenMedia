import Foundation
import MLX
import MLXNN

public struct MiniMaxMusic3FlowTransformerConfiguration: Equatable, Sendable {
    public let inChannels: Int
    public let conditionDim: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let attentionHeadDim: Int
    public let ffInnerDim: Int
    public let rotaryDim: Int
    public let fourierEmbeddingDim: Int

    public init(
        inChannels: Int = 128,
        conditionDim: Int = 2048,
        numLayers: Int = 36,
        numAttentionHeads: Int = 32,
        attentionHeadDim: Int = 64,
        ffInnerDim: Int = 8192,
        rotaryDim: Int = 32,
        fourierEmbeddingDim: Int = 256
    ) {
        self.inChannels = inChannels
        self.conditionDim = conditionDim
        self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads
        self.attentionHeadDim = attentionHeadDim
        self.ffInnerDim = ffInnerDim
        self.rotaryDim = rotaryDim
        self.fourierEmbeddingDim = fourierEmbeddingDim
    }

    public static let music3 = Self()

    public var innerDim: Int {
        numAttentionHeads * attentionHeadDim
    }

    public var concatChannels: Int {
        2 * inChannels + conditionDim
    }

    public func validate() throws {
        guard inChannels > 0,
              conditionDim > 0,
              numLayers > 0,
              numAttentionHeads > 0,
              attentionHeadDim > 0,
              ffInnerDim > 0 else {
            throw MiniMaxMusic3FlowTransformerError.invalidConfiguration
        }
        guard rotaryDim >= 0,
              rotaryDim <= attentionHeadDim,
              rotaryDim.isMultiple(of: 2),
              fourierEmbeddingDim.isMultiple(of: 2) else {
            throw MiniMaxMusic3FlowTransformerError.invalidConfiguration
        }
    }

    public static func load(from modelDirectory: URL) throws -> Self {
        let url = modelDirectory.appendingPathComponent("config.json")
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw MiniMaxMusic3FlowTransformerError.missingFile(url)
        }
        let source: ModelConfigurationFile
        do {
            source = try JSONDecoder().decode(ModelConfigurationFile.self, from: data)
        } catch {
            throw MiniMaxMusic3FlowTransformerError.invalidConfigurationFile(url)
        }
        let configuration = Self(
            inChannels: source.inChannels,
            conditionDim: source.conditionDim,
            numLayers: source.numLayers,
            numAttentionHeads: source.numAttentionHeads,
            attentionHeadDim: source.attentionHeadDim,
            ffInnerDim: source.ffInnerDim,
            rotaryDim: source.rotaryDim,
            fourierEmbeddingDim: source.fourierEmbeddingDim
        )
        try configuration.validate()
        return configuration
    }

    private struct ModelConfigurationFile: Decodable {
        let inChannels: Int
        let conditionDim: Int
        let numLayers: Int
        let numAttentionHeads: Int
        let attentionHeadDim: Int
        let ffInnerDim: Int
        let rotaryDim: Int
        let fourierEmbeddingDim: Int

        enum CodingKeys: String, CodingKey {
            case inChannels = "dit_in_channels"
            case conditionDim = "condition_out_dim"
            case numLayers = "dit_num_layers"
            case numAttentionHeads = "dit_num_heads"
            case attentionHeadDim = "dit_head_dim"
            case ffInnerDim = "dit_ff_inner_dim"
            case rotaryDim = "dit_rotary_dim"
            case fourierEmbeddingDim = "dit_fourier_dim"
        }
    }
}

public enum MiniMaxMusic3FlowTransformerError: LocalizedError, Sendable {
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
            "MiniMax Music 3 Flow Transformer 設定無效。"
        case let .invalidConfigurationFile(url):
            "無法解析 Flow Transformer 設定：\(url.path)"
        case let .missingFile(url):
            "缺少 Flow Transformer 必要檔案：\(url.path)"
        case let .invalidInput(message):
            "Flow Transformer 輸入無效：\(message)"
        case let .missingWeights(names):
            "Flow Transformer 缺少權重：\(names.joined(separator: ", "))"
        case let .unexpectedWeights(names):
            "Flow Transformer 收到重複或未使用權重：\(names.joined(separator: ", "))"
        case let .weightShapeMismatch(name, expected, actual):
            "Flow Transformer 權重 \(name) 形狀不符，預期 \(expected)，實際 \(actual)。"
        }
    }
}

public func miniMaxMusic3RotaryFrequencies(
    sequenceLength: Int,
    rotaryDim: Int,
    theta: Float = 10_000
) throws -> (cosine: MLXArray, sine: MLXArray) {
    guard sequenceLength > 0,
          rotaryDim >= 0,
          rotaryDim.isMultiple(of: 2),
          theta > 0 else {
        throw MiniMaxMusic3FlowTransformerError.invalidConfiguration
    }
    if rotaryDim == 0 {
        let empty = MLXArray.zeros([sequenceLength, 0], dtype: .float32)
        return (empty, empty)
    }
    let indices = MLXArray.arange(0, rotaryDim, step: 2, dtype: .float32)
    let inverseFrequency = 1.0 / MLX.pow(
        MLXArray(theta),
        indices / Float(rotaryDim)
    )
    let steps = MLXArray.arange(sequenceLength, dtype: .float32)
    let frequencies = MLX.concatenated(
        [
            steps[0..., .newAxis] * inverseFrequency[.newAxis, 0...],
            steps[0..., .newAxis] * inverseFrequency[.newAxis, 0...]
        ],
        axis: -1
    )
    return (MLX.cos(frequencies), MLX.sin(frequencies))
}

public func miniMaxMusic3ApplyPartialRotary(
    _ hiddenStates: MLXArray,
    rotary: (cosine: MLXArray, sine: MLXArray)
) throws -> MLXArray {
    guard hiddenStates.ndim == 4 else {
        throw MiniMaxMusic3FlowTransformerError.invalidInput(
            "hiddenStates 必須是 [batch, length, heads, dimension]。"
        )
    }
    let cosine = rotary.cosine
    let sine = rotary.sine
    let rotaryDim = cosine.shape.last ?? 0
    guard cosine.shape == sine.shape,
          cosine.shape.first == hiddenStates.shape[1],
          rotaryDim <= hiddenStates.shape[3],
          rotaryDim.isMultiple(of: 2) else {
        throw MiniMaxMusic3FlowTransformerError.invalidInput("RoPE shape 不一致。")
    }
    guard rotaryDim > 0 else { return hiddenStates }

    let rotated = hiddenStates[0..., 0..., 0..., 0..<rotaryDim]
    let half = rotaryDim / 2
    let rotateHalf = MLX.concatenated(
        [
            -rotated[0..., 0..., 0..., half..<rotaryDim],
            rotated[0..., 0..., 0..., 0..<half]
        ],
        axis: -1
    )
    let expandedCosine = cosine[.newAxis, 0..., .newAxis, 0...].asType(hiddenStates.dtype)
    let expandedSine = sine[.newAxis, 0..., .newAxis, 0...].asType(hiddenStates.dtype)
    let leading = rotated * expandedCosine + rotateHalf * expandedSine
    return MLX.concatenated(
        [leading, hiddenStates[0..., 0..., 0..., rotaryDim..<hiddenStates.shape[3]]],
        axis: -1
    )
}

private final class MiniMaxMusic3FourierEmbedding: Module {
    @ParameterInfo(key: "weight") private var weight: MLXArray

    init(embeddingDimension: Int) {
        self._weight.wrappedValue = MLXRandom.normal([embeddingDimension / 2, 1])
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let angles = (timestep[0..., .newAxis] * (2 * Float.pi)).matmul(weight.T)
        return MLX.concatenated([MLX.cos(angles), MLX.sin(angles)], axis: -1)
    }
}

private final class MiniMaxMusic3TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") private var linear1: Linear
    @ModuleInfo(key: "linear_2") private var linear2: Linear

    init(inputDimension: Int, outputDimension: Int) {
        self._linear1.wrappedValue = Linear(inputDimension, outputDimension)
        self._linear2.wrappedValue = Linear(outputDimension, outputDimension)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        linear2(silu(linear1(hiddenStates)))
    }
}

private final class MiniMaxMusic3FlowAttention: Module {
    @ModuleInfo(key: "to_q") private var query: Linear
    @ModuleInfo(key: "to_k") private var key: Linear
    @ModuleInfo(key: "to_v") private var value: Linear
    @ModuleInfo(key: "to_out") private var output: [Linear]

    private let heads: Int
    private let headDimension: Int
    private let innerDimension: Int

    init(
        configuration: MiniMaxMusic3FlowTransformerConfiguration,
        quantizationBits: Int?
    ) {
        heads = configuration.numAttentionHeads
        headDimension = configuration.attentionHeadDim
        innerDimension = configuration.innerDim
        self._query.wrappedValue = miniMaxMusic3Linear(
            configuration.innerDim,
            configuration.innerDim,
            bias: false,
            quantizationBits: quantizationBits
        )
        self._key.wrappedValue = miniMaxMusic3Linear(
            configuration.innerDim,
            configuration.innerDim,
            bias: false,
            quantizationBits: quantizationBits
        )
        self._value.wrappedValue = miniMaxMusic3Linear(
            configuration.innerDim,
            configuration.innerDim,
            bias: false,
            quantizationBits: quantizationBits
        )
        self._output.wrappedValue = [
            miniMaxMusic3Linear(
                configuration.innerDim,
                configuration.innerDim,
                bias: false,
                quantizationBits: quantizationBits
            )
        ]
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        rotary: (cosine: MLXArray, sine: MLXArray)
    ) throws -> MLXArray {
        let batch = hiddenStates.shape[0]
        let length = hiddenStates.shape[1]
        let shape = [batch, length, heads, headDimension]
        let queries = try miniMaxMusic3ApplyPartialRotary(
            query(hiddenStates).reshaped(shape),
            rotary: rotary
        ).transposed(0, 2, 1, 3)
        let keys = try miniMaxMusic3ApplyPartialRotary(
            key(hiddenStates).reshaped(shape),
            rotary: rotary
        ).transposed(0, 2, 1, 3)
        let values = value(hiddenStates)
            .reshaped(shape)
            .transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: pow(Float(headDimension), -0.5),
            mask: nil
        )
        let merged = attended.transposed(0, 2, 1, 3)
            .reshaped(batch, length, innerDimension)
        return output[0](merged)
    }
}

private final class MiniMaxMusic3FlowTransformerBlock: Module {
    @ModuleInfo(key: "norm1") private var norm1: LayerNorm
    @ModuleInfo(key: "attn") private var attention: MiniMaxMusic3FlowAttention
    @ModuleInfo(key: "norm2") private var norm2: LayerNorm
    @ModuleInfo(key: "ff_in") private var feedForwardIn: Linear
    @ModuleInfo(key: "ff_out") private var feedForwardOut: Linear

    init(
        configuration: MiniMaxMusic3FlowTransformerConfiguration,
        quantizationBits: Int?
    ) {
        self._norm1.wrappedValue = LayerNorm(dimensions: configuration.innerDim)
        self._attention.wrappedValue = MiniMaxMusic3FlowAttention(
            configuration: configuration,
            quantizationBits: quantizationBits
        )
        self._norm2.wrappedValue = LayerNorm(dimensions: configuration.innerDim)
        self._feedForwardIn.wrappedValue = miniMaxMusic3Linear(
            configuration.innerDim,
            configuration.ffInnerDim * 2,
            quantizationBits: quantizationBits
        )
        self._feedForwardOut.wrappedValue = miniMaxMusic3Linear(
            configuration.ffInnerDim,
            configuration.innerDim,
            quantizationBits: quantizationBits
        )
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        rotary: (cosine: MLXArray, sine: MLXArray)
    ) throws -> MLXArray {
        var output = hiddenStates + (try attention(norm1(hiddenStates), rotary: rotary))
        let gates = feedForwardIn(norm2(output)).split(parts: 2, axis: -1)
        output = output + feedForwardOut(gates[0] * silu(gates[1]))
        return output
    }
}

public final class MiniMaxMusic3FlowTransformer: Module {
    public let configuration: MiniMaxMusic3FlowTransformerConfiguration

    @ModuleInfo(key: "time_proj") private var timeProjection: MiniMaxMusic3FourierEmbedding
    @ModuleInfo(key: "time_embed") private var timeEmbedding: MiniMaxMusic3TimestepEmbedding
    @ModuleInfo(key: "preprocess_conv") private var preprocessConvolution: Conv1d
    @ModuleInfo(key: "proj_in") private var inputProjection: Linear
    @ModuleInfo(key: "transformer_blocks") private var blocks: [MiniMaxMusic3FlowTransformerBlock]
    @ModuleInfo(key: "proj_out") private var outputProjection: Linear
    @ModuleInfo(key: "postprocess_conv") private var postprocessConvolution: Conv1d

    public init(
        configuration: MiniMaxMusic3FlowTransformerConfiguration = .music3,
        quantizationBits: Int? = 4
    ) {
        precondition((try? configuration.validate()) != nil)
        self.configuration = configuration
        self._timeProjection.wrappedValue = MiniMaxMusic3FourierEmbedding(
            embeddingDimension: configuration.fourierEmbeddingDim
        )
        self._timeEmbedding.wrappedValue = MiniMaxMusic3TimestepEmbedding(
            inputDimension: configuration.fourierEmbeddingDim,
            outputDimension: configuration.innerDim
        )
        self._preprocessConvolution.wrappedValue = Conv1d(
            inputChannels: configuration.concatChannels,
            outputChannels: configuration.concatChannels,
            kernelSize: 1,
            bias: false
        )
        self._inputProjection.wrappedValue = miniMaxMusic3Linear(
            configuration.concatChannels,
            configuration.innerDim,
            bias: false,
            quantizationBits: quantizationBits
        )
        self._blocks.wrappedValue = (0..<configuration.numLayers).map { _ in
            MiniMaxMusic3FlowTransformerBlock(
                configuration: configuration,
                quantizationBits: quantizationBits
            )
        }
        self._outputProjection.wrappedValue = miniMaxMusic3Linear(
            configuration.innerDim,
            configuration.inChannels,
            bias: false,
            quantizationBits: quantizationBits
        )
        self._postprocessConvolution.wrappedValue = Conv1d(
            inputChannels: configuration.inChannels,
            outputChannels: configuration.inChannels,
            kernelSize: 1,
            bias: false
        )
        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        timestep: MLXArray,
        encoderHiddenStates: MLXArray
    ) throws -> MLXArray {
        guard hiddenStates.ndim == 3 else {
            throw MiniMaxMusic3FlowTransformerError.invalidInput(
                "hiddenStates 必須是 [batch, length, channels]。"
            )
        }
        let batch = hiddenStates.shape[0]
        let length = hiddenStates.shape[1]
        guard hiddenStates.shape[2] == configuration.inChannels else {
            throw MiniMaxMusic3FlowTransformerError.invalidInput("latent channel 數不符。")
        }
        guard encoderHiddenStates.shape == [batch, length, configuration.conditionDim] else {
            throw MiniMaxMusic3FlowTransformerError.invalidInput(
                "condition 必須是 [(batch), (length), (configuration.conditionDim)]。"
            )
        }
        guard timestep.shape == [batch] else {
            throw MiniMaxMusic3FlowTransformerError.invalidInput("timestep 必須是 [batch]。")
        }

        let zeros = MLXArray.zeros(like: hiddenStates)
        var output = MLX.concatenated(
            [hiddenStates, zeros, encoderHiddenStates],
            axis: -1
        )
        output = preprocessConvolution(output) + output
        let time = timeEmbedding(timeProjection(timestep))
        output = inputProjection(output)
        output = MLX.concatenated([time[0..., .newAxis, 0...], output], axis: 1)
        let rotary = try miniMaxMusic3RotaryFrequencies(
            sequenceLength: output.shape[1],
            rotaryDim: configuration.rotaryDim
        )
        for block in blocks {
            output = try block(output, rotary: rotary)
        }
        output = outputProjection(output[0..., 1..<output.shape[1], 0...])
        return postprocessConvolution(output) + output
    }
}
