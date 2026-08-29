import Foundation
import MLX
import MLXNN

public struct LTXGemmaConnectorConfiguration: Sendable, Equatable {
    public let captionChannels: Int
    public let gemmaLayerCount: Int
    public let videoDimension: Int
    public let audioDimension: Int
    public let attentionHeads: Int
    public let videoHeadDimension: Int
    public let audioHeadDimension: Int
    public let layerCount: Int
    public let registerCount: Int
    public let feedForwardMultiplier: Float
    public let maxPosition: Int
    public let normalizeOutput: Bool
    public let gatedAttention: Bool

    public init(
        captionChannels: Int = 3840,
        gemmaLayerCount: Int = 49,
        videoDimension: Int = 4096,
        audioDimension: Int = 2048,
        attentionHeads: Int = 32,
        videoHeadDimension: Int = 128,
        audioHeadDimension: Int = 64,
        layerCount: Int = 8,
        registerCount: Int = 128,
        feedForwardMultiplier: Float = 4,
        maxPosition: Int = 4096,
        normalizeOutput: Bool = true,
        gatedAttention: Bool = true
    ) throws {
        guard captionChannels > 0,
              gemmaLayerCount > 0,
              videoDimension > 0,
              audioDimension > 0,
              attentionHeads > 0,
              videoHeadDimension > 0,
              audioHeadDimension > 0,
              layerCount > 0,
              registerCount >= 0,
              feedForwardMultiplier > 0,
              maxPosition > 0,
              videoDimension.isMultiple(of: attentionHeads),
              audioDimension.isMultiple(of: attentionHeads) else {
            throw LTXGemmaConnectorRuntimeError.invalidConfiguration(
                "Gemma connector 維度、layer、register 或位置設定無效。"
            )
        }
        self.captionChannels = captionChannels
        self.gemmaLayerCount = gemmaLayerCount
        self.videoDimension = videoDimension
        self.audioDimension = audioDimension
        self.attentionHeads = attentionHeads
        self.videoHeadDimension = videoHeadDimension
        self.audioHeadDimension = audioHeadDimension
        self.layerCount = layerCount
        self.registerCount = registerCount
        self.feedForwardMultiplier = feedForwardMultiplier
        self.maxPosition = maxPosition
        self.normalizeOutput = normalizeOutput
        self.gatedAttention = gatedAttention
    }

    public var projectionInputDimension: Int {
        gemmaLayerCount * captionChannels
    }
}

public enum LTXGemmaConnectorRuntimeError: LocalizedError, Sendable, Equatable {
    case invalidConfiguration(String)
    case invalidInputShape([Int])
    case missingWeightsFile(URL)
    case missingWeights([String])
    case duplicateWeight(String)
    case weightShapeMismatch(name: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "LTX Gemma connector 設定無效：\(message)"
        case .invalidInputShape(let shape):
            return "LTX Gemma connector 輸入 shape 無效：\(shape)"
        case .missingWeightsFile(let url):
            return "找不到 LTX Gemma connector 權重：\(url.path)"
        case .missingWeights(let keys):
            return "LTX Gemma connector 缺少權重：\(keys.prefix(12).joined(separator: "、"))"
        case .duplicateWeight(let key):
            return "LTX Gemma connector 權重重複：\(key)"
        case .weightShapeMismatch(let name, let expected, let actual):
            return "LTX Gemma connector 權重 \(name) shape 不一致：預期 \(expected)，實際 \(actual)。"
        }
    }
}

public final class LTXGemmaTextEmbeddingProjection: Module {
    @ModuleInfo(key: "video_aggregate_embed") public var videoAggregateEmbed: Linear
    @ModuleInfo(key: "audio_aggregate_embed") public var audioAggregateEmbed: Linear

    private let embeddingDimension: Int

    public init(
        inputDimension: Int,
        videoDimension: Int,
        audioDimension: Int,
        embeddingDimension: Int
    ) {
        self.embeddingDimension = embeddingDimension
        self._videoAggregateEmbed = ModuleInfo(
            wrappedValue: Linear(inputDimension, videoDimension),
            key: "video_aggregate_embed"
        )
        self._audioAggregateEmbed = ModuleInfo(
            wrappedValue: Linear(inputDimension, audioDimension),
            key: "audio_aggregate_embed"
        )
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> (video: MLXArray, audio: MLXArray) {
        let videoScale = Foundation.sqrt(Float(videoAggregateEmbed.weight.shape[0]) / Float(embeddingDimension))
        let audioScale = Foundation.sqrt(Float(audioAggregateEmbed.weight.shape[0]) / Float(embeddingDimension))
        return (
            videoAggregateEmbed(hiddenStates * videoScale),
            audioAggregateEmbed(hiddenStates * audioScale)
        )
    }
}

public final class LTXGemmaConnectorAttention: Module {
    @ModuleInfo(key: "to_q") public var toQ: Linear
    @ModuleInfo(key: "to_k") public var toK: Linear
    @ModuleInfo(key: "to_v") public var toV: Linear
    @ModuleInfo(key: "to_out") public var toOut: [Linear]
    @ModuleInfo(key: "to_gate_logits") public var toGateLogits: Linear?
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    public let headCount: Int
    public let headDimension: Int
    public let scale: Float

    public init(
        dimension: Int,
        headCount: Int,
        headDimension: Int,
        gated: Bool
    ) {
        self.headCount = headCount
        self.headDimension = headDimension
        self.scale = 1 / Foundation.sqrt(Float(headDimension))
        let innerDimension = headCount * headDimension
        self._toQ = ModuleInfo(wrappedValue: Linear(dimension, innerDimension), key: "to_q")
        self._toK = ModuleInfo(wrappedValue: Linear(dimension, innerDimension), key: "to_k")
        self._toV = ModuleInfo(wrappedValue: Linear(dimension, innerDimension), key: "to_v")
        self._toOut = ModuleInfo(
            wrappedValue: [Linear(innerDimension, dimension)],
            key: "to_out"
        )
        self._toGateLogits = ModuleInfo(
            wrappedValue: gated ? Linear(dimension, headCount) : nil,
            key: "to_gate_logits"
        )
        self._qNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: innerDimension, eps: 1e-5),
            key: "q_norm"
        )
        self._kNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: innerDimension, eps: 1e-5),
            key: "k_norm"
        )
        super.init()
    }

    public func callAsFunction(
        _ input: MLXArray,
        rope: LTXRoPEFrequencies? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let batch = input.shape[0]
        let sequenceLength = input.shape[1]
        var query = qNorm(toQ(input))
        var key = kNorm(toK(input))
        var value = toV(input)

        query = query.reshaped(batch, sequenceLength, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        key = key.reshaped(batch, sequenceLength, headCount, headDimension)
            .transposed(0, 2, 1, 3)
        value = value.reshaped(batch, sequenceLength, headCount, headDimension)
            .transposed(0, 2, 1, 3)

        if let rope {
            query = LTXTransformerOps.applyRoPE(query, frequencies: rope)
            key = LTXTransformerOps.applyRoPE(key, frequencies: rope)
        }

        var scores = matmul(query, key.transposed(0, 1, 3, 2)) * scale
        if let attentionMask {
            if attentionMask.ndim == 2 {
                let invalid = (1 - attentionMask.asType(scores.dtype)) * -1e9
                scores = scores + invalid[0..., .newAxis, .newAxis, 0...]
            } else {
                scores = scores + attentionMask
            }
        }
        var output = matmul(softmax(scores, axis: -1), value)

        if let toGateLogits {
            let gate = (2 * sigmoid(toGateLogits(input)))
                .transposed(0, 2, 1)
                .reshaped(batch, headCount, sequenceLength, 1)
            output = output * gate
        }

        output = output.transposed(0, 2, 1, 3)
            .reshaped(batch, sequenceLength, headCount * headDimension)
        return toOut[0](output)
    }
}

public final class LTXGemmaConnectorGELUProjection: Module {
    @ModuleInfo public var proj: Linear

    public init(inputDimension: Int, outputDimension: Int) {
        self._proj = ModuleInfo(
            wrappedValue: Linear(inputDimension, outputDimension),
            key: "proj"
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        geluApproximate(proj(input))
    }
}

public final class LTXGemmaConnectorIdentity: Module {
    public override init() {
        super.init()
    }
}

public final class LTXGemmaConnectorFeedForward: Module {
    @ModuleInfo(key: "net") public var net: (
        LTXGemmaConnectorGELUProjection,
        LTXGemmaConnectorIdentity,
        Linear
    )

    public init(dimension: Int, multiplier: Float) {
        let hiddenDimension = Int(Float(dimension) * multiplier)
        self._net = ModuleInfo(
            wrappedValue: (
                LTXGemmaConnectorGELUProjection(
                    inputDimension: dimension,
                    outputDimension: hiddenDimension
                ),
                LTXGemmaConnectorIdentity(),
                Linear(hiddenDimension, dimension)
            ),
            key: "net"
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        net.2(net.0(input))
    }
}

public final class LTXGemmaConnectorTransformerBlock: Module {
    @ModuleInfo(key: "attn1") public var attention: LTXGemmaConnectorAttention
    @ModuleInfo(key: "ff") public var feedForward: LTXGemmaConnectorFeedForward

    public init(
        dimension: Int,
        headCount: Int,
        headDimension: Int,
        multiplier: Float,
        gated: Bool
    ) {
        self._attention = ModuleInfo(
            wrappedValue: LTXGemmaConnectorAttention(
                dimension: dimension,
                headCount: headCount,
                headDimension: headDimension,
                gated: gated
            ),
            key: "attn1"
        )
        self._feedForward = ModuleInfo(
            wrappedValue: LTXGemmaConnectorFeedForward(
                dimension: dimension,
                multiplier: multiplier
            ),
            key: "ff"
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray, rope: LTXRoPEFrequencies?) -> MLXArray {
        let normalized = LTXTransformerOps.rmsNorm(input, eps: 1e-6)
        let attentionOutput = attention(normalized, rope: rope)
        let afterAttention = input + attentionOutput
        return afterAttention + feedForward(LTXTransformerOps.rmsNorm(afterAttention, eps: 1e-6))
    }
}

public final class LTXGemmaEmbeddings1DConnector: Module {
    @ParameterInfo(key: "learnable_registers") public var learnableRegisters: MLXArray
    @ModuleInfo(key: "transformer_1d_blocks") public var transformer1DBlocks: [LTXGemmaConnectorTransformerBlock]

    public let dimension: Int
    public let registerCount: Int
    public let maxPosition: Int
    public let headDimension: Int

    public init(
        dimension: Int,
        headCount: Int,
        headDimension: Int,
        layerCount: Int,
        registerCount: Int,
        multiplier: Float,
        maxPosition: Int,
        gated: Bool
    ) {
        self.dimension = dimension
        self.registerCount = registerCount
        self.maxPosition = maxPosition
        self.headDimension = headDimension
        self._learnableRegisters = ParameterInfo(
            wrappedValue: MLXArray.zeros([registerCount, dimension]),
            key: "learnable_registers"
        )
        self._transformer1DBlocks = ModuleInfo(
            wrappedValue: (0..<layerCount).map { _ in
                LTXGemmaConnectorTransformerBlock(
                    dimension: dimension,
                    headCount: headCount,
                    headDimension: headDimension,
                    multiplier: multiplier,
                    gated: gated
                )
            },
            key: "transformer_1d_blocks"
        )
        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let batch = hiddenStates.shape[0]
        let sequenceLength = hiddenStates.shape[1]
        var states = hiddenStates

        if registerCount > 0 {
            let tileCount = max(1, (sequenceLength + registerCount - 1) / registerCount)
            let registers = tiled(
                learnableRegisters[.newAxis, 0..., 0...],
                repetitions: [batch, tileCount, 1]
            )[0..., ..<sequenceLength, 0...]

            if let attentionMask {
                let maskValues = attentionMask.asType(.int32).asArray(Int32.self)
                var results: [MLXArray] = []
                results.reserveCapacity(batch)
                for batchIndex in 0..<batch {
                    let validCount = Int(maskValues[batchIndex * sequenceLength..<((batchIndex + 1) * sequenceLength)].reduce(0, +))
                    let valid = validCount > 0
                        ? states[batchIndex, (sequenceLength - validCount)..., 0...]
                        : MLXArray.zeros([0, dimension], dtype: states.dtype)
                    let padding = validCount < sequenceLength
                        ? MLXArray.zeros([sequenceLength - validCount, dimension], dtype: states.dtype)
                        : MLXArray.zeros([0, dimension], dtype: states.dtype)
                    let adjusted = concatenated([valid, padding], axis: 0)
                    let validMask = concatenated([
                        MLXArray.ones([validCount, 1], dtype: states.dtype),
                        MLXArray.zeros([sequenceLength - validCount, 1], dtype: states.dtype)
                    ], axis: 0)
                    results.append(
                        validMask * adjusted
                            + (1 - validMask) * registers[batchIndex]
                    )
                }
                states = stacked(results, axis: 0)
            } else {
                states = concatenated([states, registers], axis: 1)
            }
        }

        let positions = MLXArray(0..<states.shape[1])
            .asType(.float32)
            .reshaped(1, states.shape[1], 1)
        let rope = LTXTransformerOps.precomputeRoPE(
            positions: positions,
            numHeads: dimension / headDimension,
            headDimension: headDimension,
            theta: 10_000,
            maxPositions: [maxPosition],
            type: .split
        )
        for block in transformer1DBlocks {
            states = block(states, rope: rope)
            if ProcessInfo.processInfo.environment["LTX2_GEMMA_EVAL_EVERY"] != "0" {
                MLX.eval(states)
            }
        }
        return LTXTransformerOps.rmsNorm(states, eps: 1e-6)
    }
}

public final class LTXGemmaTextEncoderConnector: Module {
    @ModuleInfo(key: "text_embedding_projection") public var textEmbeddingProjection: LTXGemmaTextEmbeddingProjection
    @ModuleInfo(key: "video_embeddings_connector") public var videoEmbeddingsConnector: LTXGemmaEmbeddings1DConnector
    @ModuleInfo(key: "audio_embeddings_connector") public var audioEmbeddingsConnector: LTXGemmaEmbeddings1DConnector

    public let configuration: LTXGemmaConnectorConfiguration

    public init(configuration: LTXGemmaConnectorConfiguration = try! .init()) {
        self.configuration = configuration
        self._textEmbeddingProjection = ModuleInfo(
            wrappedValue: LTXGemmaTextEmbeddingProjection(
                inputDimension: configuration.projectionInputDimension,
                videoDimension: configuration.videoDimension,
                audioDimension: configuration.audioDimension,
                embeddingDimension: configuration.captionChannels
            ),
            key: "text_embedding_projection"
        )
        self._videoEmbeddingsConnector = ModuleInfo(
            wrappedValue: LTXGemmaEmbeddings1DConnector(
                dimension: configuration.videoDimension,
                headCount: configuration.attentionHeads,
                headDimension: configuration.videoHeadDimension,
                layerCount: configuration.layerCount,
                registerCount: configuration.registerCount,
                multiplier: configuration.feedForwardMultiplier,
                maxPosition: configuration.maxPosition,
                gated: configuration.gatedAttention
            ),
            key: "video_embeddings_connector"
        )
        self._audioEmbeddingsConnector = ModuleInfo(
            wrappedValue: LTXGemmaEmbeddings1DConnector(
                dimension: configuration.audioDimension,
                headCount: configuration.attentionHeads,
                headDimension: configuration.audioHeadDimension,
                layerCount: configuration.layerCount,
                registerCount: configuration.registerCount,
                multiplier: configuration.feedForwardMultiplier,
                maxPosition: configuration.maxPosition,
                gated: configuration.gatedAttention
            ),
            key: "audio_embeddings_connector"
        )
        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        let projected = textEmbeddingProjection(hiddenStates)
        return (
            videoEmbeddingsConnector(projected.video, attentionMask: attentionMask),
            audioEmbeddingsConnector(projected.audio, attentionMask: attentionMask)
        )
    }
}
