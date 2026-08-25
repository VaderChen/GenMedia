import MLX
import MLXNN
import Foundation

struct ACEStepDiTConfiguration: Decodable {
    let attentionBias: Bool
    let audioAcousticHiddenDim: Int
    let headDim: Int
    let hiddenSize: Int
    let inChannels: Int
    let intermediateSize: Int
    let layerTypes: [String]
    let maxPositionEmbeddings: Int
    let numAttentionHeads: Int
    let numHiddenLayers: Int
    let numKeyValueHeads: Int
    let numLyricEncoderHiddenLayers: Int
    let numTimbreEncoderHiddenLayers: Int
    let patchSize: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let slidingWindow: Int
    let textHiddenDim: Int
    let timbreFixFrame: Int
    let timbreHiddenDim: Int
    let useSlidingWindow: Bool

    enum CodingKeys: String, CodingKey {
        case attentionBias = "attention_bias"
        case audioAcousticHiddenDim = "audio_acoustic_hidden_dim"
        case headDim = "head_dim"
        case hiddenSize = "hidden_size"
        case inChannels = "in_channels"
        case intermediateSize = "intermediate_size"
        case layerTypes = "layer_types"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case numLyricEncoderHiddenLayers = "num_lyric_encoder_hidden_layers"
        case numTimbreEncoderHiddenLayers = "num_timbre_encoder_hidden_layers"
        case patchSize = "patch_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case textHiddenDim = "text_hidden_dim"
        case timbreFixFrame = "timbre_fix_frame"
        case timbreHiddenDim = "timbre_hidden_dim"
        case useSlidingWindow = "use_sliding_window"
    }

    static func load(from url: URL) throws -> ACEStepDiTConfiguration {
        try JSONDecoder().decode(ACEStepDiTConfiguration.self, from: Data(contentsOf: url))
    }
}

enum ACEStepConditionEncoderError: LocalizedError {
    case invalidConfiguration(String)
    case invalidInput(String)
    case parameterMismatch(missing: [String])
    case missingNullCondition

    var errorDescription: String? {
        switch self {
        case let .invalidConfiguration(reason):
            "ACE-Step condition encoder 設定無效：\(reason)"
        case let .invalidInput(reason):
            "ACE-Step condition encoder 輸入無效：\(reason)"
        case let .parameterMismatch(missing):
            "ACE-Step condition encoder 缺少權重：\(missing)"
        case .missingNullCondition:
            "ACE-Step checkpoint 缺少 null_condition_emb"
        }
    }
}

final class ACEStepConditionAttention: Module {
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
    private let rope: RoPE

    init(configuration: ACEStepDiTConfiguration) {
        self.hiddenSize = configuration.hiddenSize
        self.queryHeadCount = configuration.numAttentionHeads
        self.keyValueHeadCount = configuration.numKeyValueHeads
        self.headDimension = configuration.headDim
        self.scale = pow(Float(configuration.headDim), -0.5)
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

    func callAsFunction(_ input: MLXArray, mask: MLXArray?) -> MLXArray {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)
        var queries = queryProjection(input)
            .reshaped(batchSize, sequenceLength, queryHeadCount, headDimension)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(input)
            .reshaped(batchSize, sequenceLength, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batchSize, sequenceLength, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)

        queries = rope(queryNorm(queries))
        keys = rope(keyNorm(keys))
        var output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        output = output.transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, hiddenSize)
        return outputProjection(output)
    }
}

final class ACEStepConditionMLP: Module {
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

final class ACEStepConditionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") private var attention: ACEStepConditionAttention
    @ModuleInfo(key: "mlp") private var mlp: ACEStepConditionMLP
    @ModuleInfo(key: "input_layernorm") private var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionNorm: RMSNorm

    let attentionType: String

    init(configuration: ACEStepDiTConfiguration, layerIndex: Int) {
        self._attention.wrappedValue = ACEStepConditionAttention(configuration: configuration)
        self._mlp.wrappedValue = ACEStepConditionMLP(configuration: configuration)
        self._inputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self._postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self.attentionType = configuration.layerTypes[layerIndex]
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray?) -> MLXArray {
        let attentionOutput = input + attention(inputNorm(input), mask: mask)
        return attentionOutput + mlp(postAttentionNorm(attentionOutput))
    }
}

final class ACEStepLyricConditionEncoder: Module {
    @ModuleInfo(key: "embed_tokens") private var inputProjection: Linear
    @ModuleInfo(key: "layers") private var layers: [ACEStepConditionEncoderLayer]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    private let configuration: ACEStepDiTConfiguration

    init(configuration: ACEStepDiTConfiguration) {
        self.configuration = configuration
        self._inputProjection.wrappedValue = Linear(
            configuration.textHiddenDim,
            configuration.hiddenSize,
            bias: true
        )
        self._layers.wrappedValue = (0..<configuration.numLyricEncoderHiddenLayers).map {
            ACEStepConditionEncoderLayer(configuration: configuration, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, attentionMask: MLXArray) -> MLXArray {
        var hiddenStates = inputProjection(input)
        for layer in layers {
            let mask = ACEStepAttentionMask.make(
                batchSize: hiddenStates.dim(0),
                sequenceLength: hiddenStates.dim(1),
                paddingMask: attentionMask,
                slidingWindow: layer.attentionType == "sliding_attention"
                    ? configuration.slidingWindow : nil
            )
            hiddenStates = layer(hiddenStates, mask: mask)
        }
        return norm(hiddenStates)
    }
}

final class ACEStepTimbreConditionEncoder: Module {
    @ModuleInfo(key: "embed_tokens") private var inputProjection: Linear
    @ModuleInfo(key: "layers") private var layers: [ACEStepConditionEncoderLayer]
    @ModuleInfo(key: "norm") private var norm: RMSNorm
    @ParameterInfo(key: "special_token") private var specialToken: MLXArray

    private let configuration: ACEStepDiTConfiguration

    init(configuration: ACEStepDiTConfiguration) {
        self.configuration = configuration
        self._inputProjection.wrappedValue = Linear(
            configuration.timbreHiddenDim,
            configuration.hiddenSize,
            bias: true
        )
        self._layers.wrappedValue = (0..<configuration.numTimbreEncoderHiddenLayers).map {
            ACEStepConditionEncoderLayer(configuration: configuration, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        self._specialToken.wrappedValue = MLXArray.zeros([1, 1, configuration.hiddenSize])
        super.init()
    }

    func encode(
        _ input: MLXArray,
        referenceOrder: MLXArray
    ) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        guard input.ndim == 3, input.dim(0) > 0, input.dim(1) > 0 else {
            throw ACEStepConditionEncoderError.invalidInput(
                "timbre 輸入必須為非空 [N, T, D]，實際 \(input.shape)"
            )
        }
        var hiddenStates = inputProjection(input)
        for layer in layers {
            let mask = ACEStepAttentionMask.make(
                batchSize: hiddenStates.dim(0),
                sequenceLength: hiddenStates.dim(1),
                paddingMask: nil,
                slidingWindow: layer.attentionType == "sliding_attention"
                    ? configuration.slidingWindow : nil
            )
            hiddenStates = layer(hiddenStates, mask: mask)
        }
        hiddenStates = norm(hiddenStates)[0..., 0, 0...]
        return try unpack(hiddenStates, referenceOrder: referenceOrder)
    }

    private func unpack(
        _ packed: MLXArray,
        referenceOrder: MLXArray
    ) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        MLX.eval(referenceOrder)
        let order = referenceOrder.asType(.int32).asArray(Int32.self).map(Int.init)
        guard order.count == packed.dim(0), let maximumBatchIndex = order.max(),
              order.allSatisfy({ $0 >= 0 }) else {
            throw ACEStepConditionEncoderError.invalidInput(
                "reference order 與 timbre 數量不一致"
            )
        }
        let batchSize = maximumBatchIndex + 1
        let groups = (0..<batchSize).map { batchIndex in
            order.indices.filter { order[$0] == batchIndex }
        }
        guard groups.allSatisfy({ !$0.isEmpty }) else {
            throw ACEStepConditionEncoderError.invalidInput("reference order 的 batch index 不連續")
        }
        let maximumCount = groups.map(\.count).max() ?? 0
        var rows: [MLXArray] = []
        var masks: [MLXArray] = []
        for indices in groups {
            let gathered = MLX.take(
                packed,
                MLXArray(indices.map(Int32.init)),
                axis: 0
            )
            let paddingCount = maximumCount - indices.count
            let row = paddingCount > 0
                ? MLX.concatenated([
                    gathered,
                    MLXArray.zeros([paddingCount, packed.dim(1)], dtype: packed.dtype)
                ], axis: 0)
                : gathered
            rows.append(row)
            masks.append(MLX.concatenated([
                MLXArray.ones([indices.count], dtype: .int32),
                MLXArray.zeros([paddingCount], dtype: .int32)
            ], axis: 0))
        }
        return (MLX.stacked(rows), MLX.stacked(masks))
    }
}

final class ACEStepConditionEncoder: Module {
    @ModuleInfo(key: "text_projector") private var textProjector: Linear
    @ModuleInfo(key: "lyric_encoder") private var lyricEncoder: ACEStepLyricConditionEncoder
    @ModuleInfo(key: "timbre_encoder") private var timbreEncoder: ACEStepTimbreConditionEncoder

    init(configuration: ACEStepDiTConfiguration) throws {
        let requiredLayers = max(
            configuration.numLyricEncoderHiddenLayers,
            configuration.numTimbreEncoderHiddenLayers
        )
        guard configuration.layerTypes.count >= requiredLayers else {
            throw ACEStepConditionEncoderError.invalidConfiguration(
                "layer_types 僅有 \(configuration.layerTypes.count) 層，至少需要 \(requiredLayers) 層"
            )
        }
        guard configuration.hiddenSize == configuration.numAttentionHeads * configuration.headDim,
              configuration.numAttentionHeads.isMultiple(of: configuration.numKeyValueHeads) else {
            throw ACEStepConditionEncoderError.invalidConfiguration("Attention heads 與 hidden size 不相容")
        }
        self._textProjector.wrappedValue = Linear(
            configuration.textHiddenDim,
            configuration.hiddenSize,
            bias: false
        )
        self._lyricEncoder.wrappedValue = ACEStepLyricConditionEncoder(configuration: configuration)
        self._timbreEncoder.wrappedValue = ACEStepTimbreConditionEncoder(configuration: configuration)
        super.init()
    }

    func encode(
        textHiddenStates: MLXArray,
        textAttentionMask: MLXArray,
        lyricHiddenStates: MLXArray,
        lyricAttentionMask: MLXArray,
        timbreHiddenStates: MLXArray,
        timbreReferenceOrder: MLXArray
    ) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        let projectedText = textProjector(textHiddenStates)
        let encodedLyrics = lyricEncoder(
            lyricHiddenStates,
            attentionMask: lyricAttentionMask
        )
        let encodedTimbre = try timbreEncoder.encode(
            timbreHiddenStates,
            referenceOrder: timbreReferenceOrder
        )
        let lyricAndTimbre = try ACEStepSequencePacker.pack(
            encodedLyrics,
            encodedTimbre.hiddenStates,
            lyricAttentionMask,
            encodedTimbre.attentionMask
        )
        return try ACEStepSequencePacker.pack(
            lyricAndTimbre.hiddenStates,
            projectedText,
            lyricAndTimbre.attentionMask,
            textAttentionMask
        )
    }
}

enum ACEStepConditionWeightLoader {
    static func load(
        model: ACEStepConditionEncoder,
        from url: URL
    ) throws -> MLXArray {
        let source = try MLX.loadArrays(url: url)
        let expectedKeys = model.parameters().flattened().map(\.0)
        var converted: [String: MLXArray] = [:]
        var missing: [String] = []
        for key in expectedKeys {
            let sourceKey = "encoder." + key
            guard let value = source[sourceKey] else {
                missing.append(sourceKey)
                continue
            }
            converted[key] = value
        }
        guard missing.isEmpty else {
            throw ACEStepConditionEncoderError.parameterMismatch(missing: missing.sorted())
        }
        guard let nullCondition = source["null_condition_emb"] else {
            throw ACEStepConditionEncoderError.missingNullCondition
        }
        try model.update(
            parameters: ModuleParameters.unflattened(converted),
            verify: .all
        )
        MLX.eval(model, nullCondition)
        return nullCondition
    }
}

private enum ACEStepAttentionMask {
    static func make(
        batchSize: Int,
        sequenceLength: Int,
        paddingMask: MLXArray?,
        slidingWindow: Int?
    ) -> MLXArray? {
        var result: MLXArray?
        if let slidingWindow {
            let positions = MLXArray(0..<sequenceLength)
            let rows = positions.reshaped(sequenceLength, 1)
            let columns = positions.reshaped(1, sequenceLength)
            result = (MLX.abs(rows - columns) .<= slidingWindow)
                .reshaped(1, 1, sequenceLength, sequenceLength)
        }
        if let paddingMask {
            let keyMask = paddingMask.asType(.bool)
                .reshaped(batchSize, 1, 1, sequenceLength)
            result = result.map { $0 .&& keyMask } ?? keyMask
        }
        return result
    }
}

private enum ACEStepSequencePacker {
    static func pack(
        _ firstHiddenStates: MLXArray,
        _ secondHiddenStates: MLXArray,
        _ firstMask: MLXArray,
        _ secondMask: MLXArray
    ) throws -> (hiddenStates: MLXArray, attentionMask: MLXArray) {
        let batchSize = firstHiddenStates.dim(0)
        guard secondHiddenStates.dim(0) == batchSize,
              firstMask.dim(0) == batchSize,
              secondMask.dim(0) == batchSize,
              firstHiddenStates.dim(2) == secondHiddenStates.dim(2) else {
            throw ACEStepConditionEncoderError.invalidInput("pack_sequences 維度不相容")
        }

        let firstLength = firstHiddenStates.dim(1)
        let secondLength = secondHiddenStates.dim(1)
        MLX.eval(firstMask, secondMask)
        let firstValues = firstMask.asType(.int32).asArray(Int32.self)
        let secondValues = secondMask.asType(.int32).asArray(Int32.self)
        var packedRows: [MLXArray] = []
        var packedMasks: [MLXArray] = []
        for batchIndex in 0..<batchSize {
            let row = MLX.concatenated([
                firstHiddenStates[batchIndex, 0..., 0...],
                secondHiddenStates[batchIndex, 0..., 0...]
            ], axis: 0)
            let firstBase = batchIndex * firstLength
            let secondBase = batchIndex * secondLength
            let mask = Array(firstValues[firstBase..<(firstBase + firstLength)])
                + Array(secondValues[secondBase..<(secondBase + secondLength)])
            let valid = mask.indices.filter { mask[$0] != 0 }
            let invalid = mask.indices.filter { mask[$0] == 0 }
            let indices = (valid + invalid).map(Int32.init)
            packedRows.append(MLX.take(row, MLXArray(indices), axis: 0))
            packedMasks.append(MLX.concatenated([
                MLXArray.ones([valid.count], dtype: .int32),
                MLXArray.zeros([invalid.count], dtype: .int32)
            ], axis: 0))
        }
        return (MLX.stacked(packedRows), MLX.stacked(packedMasks))
    }
}
