import Foundation
import MLX
import MLXNN

public struct LTXVideo096TransformerConfiguration: Sendable, Equatable {
    public let numLayers: Int
    public let dimension: Int
    public let numHeads: Int
    public let headDimension: Int
    public let latentChannels: Int
    public let captionChannels: Int
    public let timestepDimension: Int
    public let timestepScaleMultiplier: Float
    public let ropeTheta: Float
    public let ropeMaxPositions: [Int]
    public let normEps: Float

    public init(
        numLayers: Int = 28,
        dimension: Int = 2048,
        numHeads: Int = 32,
        headDimension: Int = 64,
        latentChannels: Int = 128,
        captionChannels: Int = 4096,
        timestepDimension: Int = 256,
        timestepScaleMultiplier: Float = 1000,
        ropeTheta: Float = 10000,
        ropeMaxPositions: [Int] = [20, 2048, 2048],
        normEps: Float = 1e-6
    ) throws {
        guard numLayers > 0, dimension > 0, numHeads > 0,
              headDimension > 0, numHeads * headDimension == dimension,
              latentChannels > 0, captionChannels > 0, timestepDimension > 0,
              ropeMaxPositions.count == 3,
              ropeMaxPositions.allSatisfy({ $0 > 0 }),
              ropeTheta > 1, normEps > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "LTX-Video 0.9.6 Transformer 設定無效。"
            )
        }
        self.numLayers = numLayers
        self.dimension = dimension
        self.numHeads = numHeads
        self.headDimension = headDimension
        self.latentChannels = latentChannels
        self.captionChannels = captionChannels
        self.timestepDimension = timestepDimension
        self.timestepScaleMultiplier = timestepScaleMultiplier
        self.ropeTheta = ropeTheta
        self.ropeMaxPositions = ropeMaxPositions
        self.normEps = normEps
    }
}

final class LTXVideo096TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDimension: Int, outputDimension: Int) {
        self._linear1 = ModuleInfo(
            wrappedValue: Linear(inputDimension, outputDimension), key: "linear_1"
        )
        self._linear2 = ModuleInfo(
            wrappedValue: Linear(outputDimension, outputDimension), key: "linear_2"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear2(silu(linear1(input)))
    }
}

final class LTXVideo096TimestepEmbeddingContainer: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: LTXVideo096TimestepEmbedding

    init(inputDimension: Int, outputDimension: Int) {
        self._timestepEmbedder = ModuleInfo(
            wrappedValue: LTXVideo096TimestepEmbedding(
                inputDimension: inputDimension,
                outputDimension: outputDimension
            ),
            key: "timestep_embedder"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        timestepEmbedder(input)
    }
}

final class LTXVideo096AdaLayerNormSingle: Module {
    @ModuleInfo(key: "emb") var embedding: LTXVideo096TimestepEmbeddingContainer
    @ModuleInfo(key: "linear") var linear: Linear

    init(dimension: Int, parameterCount: Int, timestepDimension: Int) {
        self._embedding = ModuleInfo(
            wrappedValue: LTXVideo096TimestepEmbeddingContainer(
                inputDimension: timestepDimension,
                outputDimension: dimension
            ),
            key: "emb"
        )
        self._linear = ModuleInfo(
            wrappedValue: Linear(dimension, parameterCount * dimension), key: "linear"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> (parameters: MLXArray, embedded: MLXArray) {
        let embedded = embedding(input)
        return (linear(silu(embedded)), embedded)
    }
}

final class LTXVideo096Attention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    private let numHeads: Int
    private let headDimension: Int
    private let useRoPE: Bool
    private let scale: Float
    private let normEps: Float

    init(configuration: LTXVideo096TransformerConfiguration, useRoPE: Bool) {
        let dimension = configuration.dimension
        self.numHeads = configuration.numHeads
        self.headDimension = configuration.headDimension
        self.useRoPE = useRoPE
        self.scale = Float(1 / Foundation.sqrt(Double(configuration.headDimension)))
        self.normEps = configuration.normEps
        self._toQ = ModuleInfo(wrappedValue: Linear(dimension, dimension), key: "to_q")
        self._toK = ModuleInfo(wrappedValue: Linear(dimension, dimension), key: "to_k")
        self._toV = ModuleInfo(wrappedValue: Linear(dimension, dimension), key: "to_v")
        self._toOut = ModuleInfo(
            wrappedValue: Linear(dimension, dimension), key: "to_out"
        )
        self._qNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: dimension, eps: configuration.normEps),
            key: "q_norm"
        )
        self._kNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: dimension, eps: configuration.normEps),
            key: "k_norm"
        )
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        encoderHiddenStates: MLXArray? = nil,
        rope: LTXRoPEFrequencies? = nil
    ) -> MLXArray {
        let batch = input.shape[0]
        let context = encoderHiddenStates ?? input
        var query = qNorm(toQ(input))
        var key = kNorm(toK(context))
        var value = toV(context)

        query = query.reshaped(batch, -1, numHeads, headDimension)
            .transposed(0, 2, 1, 3)
        key = key.reshaped(batch, -1, numHeads, headDimension)
            .transposed(0, 2, 1, 3)
        value = value.reshaped(batch, -1, numHeads, headDimension)
            .transposed(0, 2, 1, 3)

        if useRoPE, let rope {
            query = LTXTransformerOps.applyRoPE(query, frequencies: rope)
            key = LTXTransformerOps.applyRoPE(key, frequencies: rope)
        }
        let attention = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: scale,
            mask: nil
        )
        let output = attention.transposed(0, 2, 1, 3)
            .reshaped(batch, -1, numHeads * headDimension)
        return toOut(output)
    }
}

final class LTXVideo096FeedForward: Module {
    @ModuleInfo(key: "proj_in") var projectionIn: Linear
    @ModuleInfo(key: "proj_out") var projectionOut: Linear

    init(dimension: Int) {
        self._projectionIn = ModuleInfo(
            wrappedValue: Linear(dimension, dimension * 4), key: "proj_in"
        )
        self._projectionOut = ModuleInfo(
            wrappedValue: Linear(dimension * 4, dimension), key: "proj_out"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        projectionOut(geluApproximate(projectionIn(input)))
    }
}

final class LTXVideo096TransformerBlock: Module {
    @ModuleInfo(key: "attn1") var selfAttention: LTXVideo096Attention
    @ModuleInfo(key: "attn2") var crossAttention: LTXVideo096Attention
    @ModuleInfo(key: "ff") var feedForward: LTXVideo096FeedForward
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    private let dimension: Int
    private let normEps: Float

    init(configuration: LTXVideo096TransformerConfiguration) {
        self.dimension = configuration.dimension
        self.normEps = configuration.normEps
        self._selfAttention = ModuleInfo(
            wrappedValue: LTXVideo096Attention(configuration: configuration, useRoPE: true),
            key: "attn1"
        )
        self._crossAttention = ModuleInfo(
            wrappedValue: LTXVideo096Attention(configuration: configuration, useRoPE: false),
            key: "attn2"
        )
        self._feedForward = ModuleInfo(
            wrappedValue: LTXVideo096FeedForward(dimension: configuration.dimension),
            key: "ff"
        )
        self._scaleShiftTable = ParameterInfo(
            wrappedValue: MLXArray.zeros([6, configuration.dimension]),
            key: "scale_shift_table"
        )
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        timestepParameters: MLXArray,
        encoderHiddenStates: MLXArray,
        rope: LTXRoPEFrequencies?
    ) -> MLXArray {
        let batch = input.shape[0]
        let values = scaleShiftTable[.newAxis, .newAxis, 0...] + timestepParameters
            .reshaped(batch, 1, 6, dimension)
        let shiftSelf = values[0..., 0..., 0, 0...]
        let scaleSelf = values[0..., 0..., 1, 0...]
        let gateSelf = values[0..., 0..., 2, 0...]
        let shiftFF = values[0..., 0..., 3, 0...]
        let scaleFF = values[0..., 0..., 4, 0...]
        let gateFF = values[0..., 0..., 5, 0...]

        let normalized = LTXTransformerOps.rmsNorm(input, eps: normEps)
            * (1 + scaleSelf) + shiftSelf
        var hidden = input + selfAttention(normalized, rope: rope) * gateSelf
        hidden += crossAttention(hidden, encoderHiddenStates: encoderHiddenStates)
        let feedInput = LTXTransformerOps.rmsNorm(hidden, eps: normEps)
            * (1 + scaleFF) + shiftFF
        hidden += feedForward(feedInput) * gateFF
        return hidden
    }
}

public final class LTXVideo096Transformer: Module {
    @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear
    @ModuleInfo(key: "caption_projection") var captionProjection: LTXVideo096CaptionProjection
    @ModuleInfo(key: "adaln_single") var adalnSingle: LTXVideo096AdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [LTXVideo096TransformerBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    public let configuration: LTXVideo096TransformerConfiguration

    public init(configuration: LTXVideo096TransformerConfiguration) {
        self.configuration = configuration
        self._patchifyProj = ModuleInfo(
            wrappedValue: Linear(configuration.latentChannels, configuration.dimension),
            key: "patchify_proj"
        )
        self._captionProjection = ModuleInfo(
            wrappedValue: LTXVideo096CaptionProjection(
                inputDimension: configuration.captionChannels,
                dimension: configuration.dimension
            ),
            key: "caption_projection"
        )
        self._adalnSingle = ModuleInfo(
            wrappedValue: LTXVideo096AdaLayerNormSingle(
                dimension: configuration.dimension,
                parameterCount: 6,
                timestepDimension: configuration.timestepDimension
            ),
            key: "adaln_single"
        )
        self._transformerBlocks = ModuleInfo(
            wrappedValue: (0..<configuration.numLayers).map { _ in
                LTXVideo096TransformerBlock(configuration: configuration)
            },
            key: "transformer_blocks"
        )
        self._projOut = ModuleInfo(
            wrappedValue: Linear(configuration.dimension, configuration.latentChannels),
            key: "proj_out"
        )
        self._scaleShiftTable = ParameterInfo(
            wrappedValue: MLXArray.zeros([2, configuration.dimension]),
            key: "scale_shift_table"
        )
        super.init()
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        indicesGrid: MLXArray,
        encoderHiddenStates: MLXArray,
        timestep: MLXArray
    ) -> MLXArray {
        let inputDType = hiddenStates.dtype
        let hidden = hiddenStates.asType(inputDType)
        let projected = patchifyProj(hidden)
        let scaledTimestep = timestep.asType(projected.dtype)
            * configuration.timestepScaleMultiplier
        let timestepInput = LTXVideo096SinusoidalEmbedding.make(
            scaledTimestep.reshaped(-1), dimension: configuration.timestepDimension
        )
        let timeValues = adalnSingle(timestepInput)
        let batch = projected.shape[0]
        let timestepParameters = timeValues.parameters.reshaped(batch, 1, -1)
        let caption = captionProjection(encoderHiddenStates.asType(projected.dtype))
        let positions = indicesGrid.ndim == 3 && indicesGrid.shape[1] == 3
            ? indicesGrid.transposed(0, 2, 1)
            : indicesGrid
        let rope = LTXTransformerOps.precomputeRoPE(
            positions: positions,
            numHeads: configuration.numHeads,
            headDimension: configuration.headDimension,
            theta: configuration.ropeTheta,
            maxPositions: configuration.ropeMaxPositions,
            type: .interleaved
        )

        var current = projected
        for block in transformerBlocks {
            current = block(
                current,
                timestepParameters: timestepParameters,
                encoderHiddenStates: caption,
                rope: rope
            )
        }

        let finalValues = scaleShiftTable[.newAxis, .newAxis, 0...]
            + timeValues.embedded.asType(current.dtype)
                .reshaped(batch, 1, 1, configuration.dimension)
        let shift = finalValues[0..., 0..., 0, 0...]
        let scale = finalValues[0..., 0..., 1, 0...]
        return projOut(
            LTXTransformerOps.rmsNorm(current, eps: configuration.normEps)
                * (1 + scale) + shift
        )
    }
}

public final class LTXVideo096CaptionProjection: Module {
    @ModuleInfo(key: "linear_1") public var linear1: Linear
    @ModuleInfo(key: "linear_2") public var linear2: Linear

    init(inputDimension: Int, dimension: Int) {
        self._linear1 = ModuleInfo(
            wrappedValue: Linear(inputDimension, dimension), key: "linear_1"
        )
        self._linear2 = ModuleInfo(
            wrappedValue: Linear(dimension, dimension), key: "linear_2"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear2(silu(linear1(input)))
    }
}

enum LTXVideo096SinusoidalEmbedding {
    static func make(_ timesteps: MLXArray, dimension: Int) -> MLXArray {
        let half = dimension / 2
        let exponent = -Foundation.log(10_000 as Float)
            * MLXArray(0..<half).asType(.float32) / Float(max(1, half))
        let angles = timesteps[.ellipsis, .newAxis].asType(.float32)
            * exp(exponent)[.newAxis, .ellipsis]
        return concatenated([sin(angles), cos(angles)], axis: -1)
    }
}
