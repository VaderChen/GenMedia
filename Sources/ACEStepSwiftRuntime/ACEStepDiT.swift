import Darwin
import MLX
import MLXNN
import Foundation

enum ACEStepDiTError: LocalizedError {
    case invalidInput(String)
    case parameterMismatch(missing: [String])
    case nonFiniteOutput

    var errorDescription: String? {
        switch self {
        case let .invalidInput(reason):
            "ACE-Step DiT 輸入無效：\(reason)"
        case let .parameterMismatch(missing):
            "ACE-Step DiT 缺少權重：\(missing)"
        case .nonFiniteOutput:
            "ACE-Step DiT 輸出含有 NaN 或 Infinity"
        }
    }
}

final class ACEStepCrossAttentionCache {
    private var keys: [Int: MLXArray] = [:]
    private var values: [Int: MLXArray] = [:]

    func value(for layerIndex: Int) -> (key: MLXArray, value: MLXArray)? {
        guard let key = keys[layerIndex], let value = values[layerIndex] else {
            return nil
        }
        return (key, value)
    }

    func update(key: MLXArray, value: MLXArray, layerIndex: Int) {
        keys[layerIndex] = key
        values[layerIndex] = value
    }
}

final class ACEStepDiTAttention: Module {
    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear
    @ModuleInfo(key: "q_norm") private var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") private var keyNorm: RMSNorm

    private let hiddenSize: Int
    private let queryHeadCount: Int
    private let keyValueHeadCount: Int
    private let headDimension: Int
    private let scale: Float
    private let layerIndex: Int
    private let isCrossAttention: Bool
    private let rope: RoPE

    init(
        configuration: ACEStepDiTConfiguration,
        layerIndex: Int,
        isCrossAttention: Bool
    ) {
        self.hiddenSize = configuration.hiddenSize
        self.queryHeadCount = configuration.numAttentionHeads
        self.keyValueHeadCount = configuration.numKeyValueHeads
        self.headDimension = configuration.headDim
        self.scale = pow(Float(configuration.headDim), -0.5)
        self.layerIndex = layerIndex
        self.isCrossAttention = isCrossAttention
        self._queryProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.numAttentionHeads * configuration.headDim,
            bias: configuration.attentionBias
        )
        self._keyProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.numKeyValueHeads * configuration.headDim,
            bias: configuration.attentionBias
        )
        self._valueProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.numKeyValueHeads * configuration.headDim,
            bias: configuration.attentionBias
        )
        self._outputProjection.wrappedValue = Linear(
            configuration.numAttentionHeads * configuration.headDim,
            configuration.hiddenSize,
            bias: configuration.attentionBias
        )
        self._queryNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDim,
            eps: configuration.rmsNormEps
        )
        self._keyNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDim,
            eps: configuration.rmsNormEps
        )
        self.rope = RoPE(
            dimensions: configuration.headDim,
            traditional: false,
            base: configuration.ropeTheta,
            scale: 1
        )
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray?,
        encoderHiddenStates: MLXArray? = nil,
        cache: ACEStepCrossAttentionCache? = nil,
        useCache: Bool = false
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let sequenceLength = hiddenStates.dim(1)
        var queries = queryNorm(
            queryProjection(hiddenStates).reshaped(
                batchSize,
                sequenceLength,
                queryHeadCount,
                headDimension
            )
        ).transposed(0, 2, 1, 3)

        var keys: MLXArray
        var values: MLXArray
        if isCrossAttention, let encoderHiddenStates {
            if let cached = cache?.value(for: layerIndex) {
                keys = cached.key
                values = cached.value
            } else {
                let encoderLength = encoderHiddenStates.dim(1)
                keys = keyNorm(
                    keyProjection(encoderHiddenStates).reshaped(
                        batchSize,
                        encoderLength,
                        keyValueHeadCount,
                        headDimension
                    )
                ).transposed(0, 2, 1, 3)
                values = valueProjection(encoderHiddenStates)
                    .reshaped(batchSize, encoderLength, keyValueHeadCount, headDimension)
                    .transposed(0, 2, 1, 3)
                if useCache {
                    cache?.update(key: keys, value: values, layerIndex: layerIndex)
                }
            }
        } else {
            keys = keyNorm(
                keyProjection(hiddenStates).reshaped(
                    batchSize,
                    sequenceLength,
                    keyValueHeadCount,
                    headDimension
                )
            ).transposed(0, 2, 1, 3)
            values = valueProjection(hiddenStates)
                .reshaped(batchSize, sequenceLength, keyValueHeadCount, headDimension)
                .transposed(0, 2, 1, 3)
            queries = rope(queries)
            keys = rope(keys)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: attentionMask
        )
        output = output.transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, hiddenSize)
        return outputProjection(output)
    }
}

final class ACEStepDiTMLP: Module {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(configuration: ACEStepDiTConfiguration) {
        self._gateProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: false
        )
        self._downProjection.wrappedValue = Linear(
            configuration.intermediateSize,
            configuration.hiddenSize,
            bias: false
        )
        self._upProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

final class ACEStepDiTLayer: Module {
    @ModuleInfo(key: "self_attn_norm") private var selfAttentionNorm: RMSNorm
    @ModuleInfo(key: "self_attn") private var selfAttention: ACEStepDiTAttention
    @ModuleInfo(key: "cross_attn_norm") private var crossAttentionNorm: RMSNorm
    @ModuleInfo(key: "cross_attn") private var crossAttention: ACEStepDiTAttention
    @ModuleInfo(key: "mlp_norm") private var mlpNorm: RMSNorm
    @ModuleInfo(key: "mlp") private var mlp: ACEStepDiTMLP
    @ParameterInfo(key: "scale_shift_table") private var scaleShiftTable: MLXArray

    let attentionType: String

    init(configuration: ACEStepDiTConfiguration, layerIndex: Int) {
        self._selfAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self._selfAttention.wrappedValue = ACEStepDiTAttention(
            configuration: configuration,
            layerIndex: layerIndex,
            isCrossAttention: false
        )
        self._crossAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self._crossAttention.wrappedValue = ACEStepDiTAttention(
            configuration: configuration,
            layerIndex: layerIndex,
            isCrossAttention: true
        )
        self._mlpNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self._mlp.wrappedValue = ACEStepDiTMLP(configuration: configuration)
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([1, 6, configuration.hiddenSize])
        self.attentionType = configuration.layerTypes[layerIndex]
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        timestepProjection: MLXArray,
        selfAttentionMask: MLXArray?,
        encoderHiddenStates: MLXArray,
        cache: ACEStepCrossAttentionCache?,
        useCache: Bool
    ) -> MLXArray {
        let modulation = scaleShiftTable + timestepProjection
        let shiftSelfAttention = modulation[0..., 0..<1, 0...]
        let scaleSelfAttention = modulation[0..., 1..<2, 0...]
        let gateSelfAttention = modulation[0..., 2..<3, 0...]
        let shiftMLP = modulation[0..., 3..<4, 0...]
        let scaleMLP = modulation[0..., 4..<5, 0...]
        let gateMLP = modulation[0..., 5..<6, 0...]

        var output = hiddenStates
        var normalized = selfAttentionNorm(output)
        normalized = normalized * (1 + scaleSelfAttention) + shiftSelfAttention
        output = output + selfAttention(
            normalized,
            attentionMask: selfAttentionMask
        ) * gateSelfAttention

        normalized = crossAttentionNorm(output)
        output = output + crossAttention(
            normalized,
            attentionMask: nil,
            encoderHiddenStates: encoderHiddenStates,
            cache: cache,
            useCache: useCache
        )

        normalized = mlpNorm(output)
        normalized = normalized * (1 + scaleMLP) + shiftMLP
        return output + mlp(normalized) * gateMLP
    }
}

final class ACEStepTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") private var inputProjection: Linear
    @ModuleInfo(key: "linear_2") private var hiddenProjection: Linear
    @ModuleInfo(key: "time_proj") private var modulationProjection: Linear

    private let inputDimensions: Int
    private let scale: Float

    init(inputDimensions: Int, hiddenDimensions: Int, scale: Float = 1_000) {
        self.inputDimensions = inputDimensions
        self.scale = scale
        self._inputProjection.wrappedValue = Linear(
            inputDimensions,
            hiddenDimensions,
            bias: true
        )
        self._hiddenProjection.wrappedValue = Linear(
            hiddenDimensions,
            hiddenDimensions,
            bias: true
        )
        self._modulationProjection.wrappedValue = Linear(
            hiddenDimensions,
            hiddenDimensions * 6,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> (embedding: MLXArray, projection: MLXArray) {
        let frequencyEmbedding = sinusoidalEmbedding(timestep)
        var embedding = inputProjection(frequencyEmbedding.asType(timestep.dtype))
        embedding = hiddenProjection(silu(embedding))
        let projection = modulationProjection(silu(embedding))
            .reshaped(timestep.dim(0), 6, -1)
        return (embedding, projection)
    }

    private func sinusoidalEmbedding(_ timestep: MLXArray) -> MLXArray {
        let halfDimensions = inputDimensions / 2
        let frequencies = MLX.exp(
            -Float(Darwin.log(Double(10_000)))
                * MLXArray(0..<halfDimensions).asType(.float32)
                / Float(halfDimensions)
        )
        let arguments = (timestep * scale).asType(.float32)[0..., .newAxis]
            * frequencies[.newAxis, 0...]
        var result = MLX.concatenated([MLX.cos(arguments), MLX.sin(arguments)], axis: -1)
        if inputDimensions.isMultiple(of: 2) == false {
            result = MLX.concatenated([
                result,
                MLXArray.zeros([timestep.dim(0), 1], dtype: .float32)
            ], axis: -1)
        }
        return result
    }
}

final class ACEStepDiTDecoder: Module {
    @ModuleInfo(key: "proj_in") private var inputProjection: Conv1d
    @ModuleInfo(key: "time_embed") private var timestepEmbedding: ACEStepTimestepEmbedding
    @ModuleInfo(key: "time_embed_r") private var referenceTimestepEmbedding: ACEStepTimestepEmbedding
    @ModuleInfo(key: "condition_embedder") private var conditionEmbedder: Linear
    @ModuleInfo(key: "layers") private var layers: [ACEStepDiTLayer]
    @ModuleInfo(key: "norm_out") private var outputNorm: RMSNorm
    @ModuleInfo(key: "proj_out") private var outputProjection: ConvTransposed1d
    @ParameterInfo(key: "scale_shift_table") private var scaleShiftTable: MLXArray

    private let configuration: ACEStepDiTConfiguration

    init(configuration: ACEStepDiTConfiguration) throws {
        guard configuration.layerTypes.count >= configuration.numHiddenLayers else {
            throw ACEStepDiTError.invalidInput("layer_types 少於 DiT 層數")
        }
        self.configuration = configuration
        self._inputProjection.wrappedValue = Conv1d(
            inputChannels: configuration.inChannels,
            outputChannels: configuration.hiddenSize,
            kernelSize: configuration.patchSize,
            stride: configuration.patchSize
        )
        self._timestepEmbedding.wrappedValue = ACEStepTimestepEmbedding(
            inputDimensions: 256,
            hiddenDimensions: configuration.hiddenSize
        )
        self._referenceTimestepEmbedding.wrappedValue = ACEStepTimestepEmbedding(
            inputDimensions: 256,
            hiddenDimensions: configuration.hiddenSize
        )
        self._conditionEmbedder.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.hiddenSize,
            bias: true
        )
        self._layers.wrappedValue = (0..<configuration.numHiddenLayers).map {
            ACEStepDiTLayer(configuration: configuration, layerIndex: $0)
        }
        self._outputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self._outputProjection.wrappedValue = ConvTransposed1d(
            inputChannels: configuration.hiddenSize,
            outputChannels: configuration.audioAcousticHiddenDim,
            kernelSize: configuration.patchSize,
            stride: configuration.patchSize
        )
        self._scaleShiftTable.wrappedValue = MLXArray.zeros([1, 2, configuration.hiddenSize])
        super.init()
    }

    func callAsFunction(
        hiddenStates inputHiddenStates: MLXArray,
        timestep: MLXArray,
        referenceTimestep: MLXArray,
        encoderHiddenStates inputEncoderHiddenStates: MLXArray,
        contextLatents: MLXArray,
        cache: ACEStepCrossAttentionCache? = nil,
        useCache: Bool = false
    ) throws -> MLXArray {
        guard inputHiddenStates.ndim == 3,
              inputHiddenStates.dim(0) == contextLatents.dim(0),
              inputHiddenStates.dim(1) == contextLatents.dim(1),
              inputHiddenStates.dim(2) == configuration.audioAcousticHiddenDim,
              contextLatents.dim(2) + inputHiddenStates.dim(2) == configuration.inChannels else {
            throw ACEStepDiTError.invalidInput(
                "hidden=\(inputHiddenStates.shape)、context=\(contextLatents.shape)"
            )
        }
        let timestepResult = timestepEmbedding(timestep)
        let referenceResult = referenceTimestepEmbedding(timestep - referenceTimestep)
        let timestepEmbedding = timestepResult.embedding + referenceResult.embedding
        let timestepProjection = timestepResult.projection + referenceResult.projection

        var hiddenStates = MLX.concatenated([contextLatents, inputHiddenStates], axis: -1)
        let originalSequenceLength = hiddenStates.dim(1)
        let remainder = originalSequenceLength % configuration.patchSize
        if remainder != 0 {
            let paddingLength = configuration.patchSize - remainder
            hiddenStates = MLX.concatenated([
                hiddenStates,
                MLXArray.zeros(
                    [hiddenStates.dim(0), paddingLength, hiddenStates.dim(2)],
                    dtype: hiddenStates.dtype
                )
            ], axis: 1)
        }
        hiddenStates = inputProjection(hiddenStates)
        let encoderHiddenStates = conditionEmbedder(inputEncoderHiddenStates)
        let slidingMask = makeSlidingMask(
            sequenceLength: hiddenStates.dim(1),
            windowSize: configuration.slidingWindow
        )

        for layer in layers {
            hiddenStates = layer(
                hiddenStates,
                timestepProjection: timestepProjection,
                selfAttentionMask: layer.attentionType == "sliding_attention" ? slidingMask : nil,
                encoderHiddenStates: encoderHiddenStates,
                cache: cache,
                useCache: useCache
            )
        }

        let outputModulation = scaleShiftTable
            + MLX.expandedDimensions(timestepEmbedding, axis: 1)
        let shift = outputModulation[0..., 0..<1, 0...]
        let scale = outputModulation[0..., 1..<2, 0...]
        hiddenStates = outputNorm(hiddenStates) * (1 + scale) + shift
        hiddenStates = outputProjection(hiddenStates)
        return hiddenStates[0..., 0..<originalSequenceLength, 0...]
    }

    private func makeSlidingMask(sequenceLength: Int, windowSize: Int) -> MLXArray {
        let positions = MLXArray(0..<sequenceLength)
        let rows = positions.reshaped(sequenceLength, 1)
        let columns = positions.reshaped(1, sequenceLength)
        return (MLX.abs(rows - columns) .<= windowSize)
            .reshaped(1, 1, sequenceLength, sequenceLength)
    }
}

enum ACEStepDiTWeightLoader {
    static func load(
        model: ACEStepDiTDecoder,
        from url: URL,
        dtype: DType = .bfloat16
    ) throws {
        let source = try MLX.loadArrays(url: url)
        let expectedKeys = model.parameters().flattened().map(\.0)
        var converted: [String: MLXArray] = [:]
        var missing: [String] = []
        for key in expectedKeys {
            let sourceKey: String
            if key.hasPrefix("proj_in.") {
                sourceKey = "decoder.proj_in.1." + key.dropFirst("proj_in.".count)
            } else if key.hasPrefix("proj_out.") {
                sourceKey = "decoder.proj_out.1." + key.dropFirst("proj_out.".count)
            } else {
                sourceKey = "decoder." + key
            }
            guard var value = source[sourceKey] else {
                missing.append(sourceKey)
                continue
            }
            if key == "proj_in.weight" {
                value = value.transposed(0, 2, 1)
            } else if key == "proj_out.weight" {
                value = value.transposed(1, 2, 0)
            }
            converted[key] = value.asType(dtype)
        }
        guard missing.isEmpty else {
            throw ACEStepDiTError.parameterMismatch(missing: missing.sorted())
        }
        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model)
    }
}
