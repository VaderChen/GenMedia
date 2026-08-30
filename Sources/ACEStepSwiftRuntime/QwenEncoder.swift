import MLX
import MLXFast
import MLXNN

final class QwenAttention: Module {
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let numKeyValueGroups: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(configuration: QwenTextEncoderConfiguration) {
        numAttentionHeads = configuration.numAttentionHeads
        numKeyValueHeads = configuration.numKeyValueHeads
        headDim = configuration.headDim
        numKeyValueGroups = configuration.numAttentionHeads / configuration.numKeyValueHeads
        scale = 1 / Float(configuration.headDim).squareRoot()

        _qProj.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.numAttentionHeads * configuration.headDim,
            bias: false
        )
        _kProj.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.numKeyValueHeads * configuration.headDim,
            bias: false
        )
        _vProj.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.numKeyValueHeads * configuration.headDim,
            bias: false
        )
        _oProj.wrappedValue = Linear(
            configuration.numAttentionHeads * configuration.headDim,
            configuration.hiddenSize,
            bias: false
        )
        _qNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDim,
            eps: configuration.rmsNormEps
        )
        _kNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDim,
            eps: configuration.rmsNormEps
        )
        rope = RoPE(
            dimensions: configuration.headDim,
            traditional: false,
            base: configuration.ropeTheta,
            scale: 1.0
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)

        var queries = qProj(input)
        var keys = kProj(input)
        var values = vProj(input)
        queries = qNorm(
            queries.reshaped(batchSize, sequenceLength, numAttentionHeads, headDim)
        ).transposed(0, 2, 1, 3)
        keys = kNorm(
            keys.reshaped(batchSize, sequenceLength, numKeyValueHeads, headDim)
        ).transposed(0, 2, 1, 3)
        values = values
            .reshaped(batchSize, sequenceLength, numKeyValueHeads, headDim)
            .transposed(0, 2, 1, 3)

        queries = rope(queries)
        keys = rope(keys)
        if numKeyValueHeads != numAttentionHeads {
            keys = expandKeyValue(keys, repeats: numKeyValueGroups)
            values = expandKeyValue(values, repeats: numKeyValueGroups)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        return oProj(
            output.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, -1)
        )
    }

    private func expandKeyValue(_ input: MLXArray, repeats: Int) -> MLXArray {
        guard repeats > 1 else { return input }
        let expanded = MLX.repeated(
            MLX.expandedDimensions(input, axis: 2),
            count: repeats,
            axis: 2
        )
        return expanded.reshaped(
            input.dim(0),
            input.dim(1) * repeats,
            input.dim(2),
            input.dim(3)
        )
    }
}

final class QwenMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProj(silu(gateProj(input)) * upProj(input))
    }
}

final class QwenEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: QwenAttention
    let mlp: QwenMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(configuration: QwenTextEncoderConfiguration) {
        _selfAttention.wrappedValue = QwenAttention(configuration: configuration)
        mlp = QwenMLP(
            dimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.intermediateSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let attentionOutput = selfAttention(inputLayerNorm(input), mask: mask)
        let residual = input + attentionOutput
        return residual + mlp(postAttentionLayerNorm(residual))
    }
}

final class QwenEncoder: Module {
    let configuration: QwenTextEncoderConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [QwenEncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(configuration: QwenTextEncoderConfiguration) {
        self.configuration = configuration
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabSize,
            dimensions: configuration.hiddenSize
        )
        _layers.wrappedValue = (0..<configuration.numHiddenLayers).map { _ in
            QwenEncoderLayer(configuration: configuration)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
    }

    func callAsFunction(
        inputIds: MLXArray,
        attentionMask: MLXArray?
    ) -> MLXArray {
        forward(inputIds: inputIds, attentionMask: attentionMask).lastHiddenState
    }

    func forward(
        inputIds: MLXArray,
        attentionMask: MLXArray?,
        outputHiddenStates: Bool = false
    ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
        let tokenIDs = inputIds.dtype == .int32 ? inputIds : inputIds.asType(.int32)
        var hidden = embedTokens(tokenIDs)
        let mask = createAttentionMask(hidden: hidden, attentionMask: attentionMask)
        var allHiddenStates: [MLXArray]? = outputHiddenStates ? [hidden] : nil

        for layer in layers {
            hidden = layer(hidden, mask: mask)
            if outputHiddenStates {
                allHiddenStates?.append(hidden)
            }
        }

        hidden = norm(hidden)
        if outputHiddenStates, var states = allHiddenStates, !states.isEmpty {
            states[states.count - 1] = hidden
            allHiddenStates = states
        }
        return (hidden, allHiddenStates)
    }

    func embed(inputIds: MLXArray) -> MLXArray {
        let tokenIDs = inputIds.dtype == .int32 ? inputIds : inputIds.asType(.int32)
        return embedTokens(tokenIDs)
    }

    private func createAttentionMask(
        hidden: MLXArray,
        attentionMask: MLXArray?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let sequenceLength = hidden.dim(1)
        guard let attentionMask else { return .causal }

        let paddingKeepMask = attentionMask
            .asType(.bool)
            .reshaped(attentionMask.dim(0), 1, 1, sequenceLength)
        let indices = MLXArray(0..<sequenceLength)
        let rows = indices.reshaped(sequenceLength, 1)
        let columns = indices.reshaped(1, sequenceLength)
        let causalKeepMask = (columns .<= rows).reshaped(1, 1, sequenceLength, sequenceLength)
        return .array(causalKeepMask .&& paddingKeepMask)
    }
}
