import Foundation
import MLX
import MLXNN

public struct LTXVideo096T5Configuration: Sendable, Equatable {
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHeads: Int
    public let headDimension: Int
    public let numLayers: Int
    public let relativeAttentionBuckets: Int
    public let relativeAttentionMaxDistance: Int
    public let normEps: Float

    public init(
        vocabularySize: Int = 32128,
        hiddenSize: Int = 4096,
        intermediateSize: Int = 10240,
        numHeads: Int = 64,
        headDimension: Int = 64,
        numLayers: Int = 24,
        relativeAttentionBuckets: Int = 32,
        relativeAttentionMaxDistance: Int = 128,
        normEps: Float = 1e-6
    ) throws {
        guard vocabularySize > 0, hiddenSize > 0, intermediateSize > 0,
              numHeads > 0, headDimension > 0,
              numHeads * headDimension == hiddenSize,
              numLayers > 0, relativeAttentionBuckets > 0,
              relativeAttentionBuckets.isMultiple(of: 2),
              relativeAttentionMaxDistance > 0, normEps > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration("T5 v1.1 設定無效。")
        }
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHeads = numHeads
        self.headDimension = headDimension
        self.numLayers = numLayers
        self.relativeAttentionBuckets = relativeAttentionBuckets
        self.relativeAttentionMaxDistance = relativeAttentionMaxDistance
        self.normEps = normEps
    }
}

final class LTXVideo096T5Attention: Module {
    @ModuleInfo(key: "attn_q") var query: Linear
    @ModuleInfo(key: "attn_k") var key: Linear
    @ModuleInfo(key: "attn_v") var value: Linear
    @ModuleInfo(key: "attn_o") var output: Linear

    private let configuration: LTXVideo096T5Configuration

    init(configuration: LTXVideo096T5Configuration) {
        self.configuration = configuration
        self._query = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false),
            key: "attn_q"
        )
        self._key = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false),
            key: "attn_k"
        )
        self._value = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false),
            key: "attn_v"
        )
        self._output = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false),
            key: "attn_o"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, relativeBias: MLXArray?) -> MLXArray {
        let batch = input.shape[0]
        let query = query(input)
            .reshaped(batch, -1, configuration.numHeads, configuration.headDimension)
            .transposed(0, 2, 1, 3)
        let key = key(input)
            .reshaped(batch, -1, configuration.numHeads, configuration.headDimension)
            .transposed(0, 2, 1, 3)
        let value = value(input)
            .reshaped(batch, -1, configuration.numHeads, configuration.headDimension)
            .transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: 1,
            mask: relativeBias
        )
        return output(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, -1, configuration.hiddenSize)
        )
    }
}

final class LTXVideo096T5FeedForward: Module {
    @ModuleInfo(key: "ffn_gate") var gate: Linear
    @ModuleInfo(key: "ffn_up") var up: Linear
    @ModuleInfo(key: "ffn_down") var down: Linear

    init(configuration: LTXVideo096T5Configuration) {
        self._gate = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.intermediateSize, bias: false),
            key: "ffn_gate"
        )
        self._up = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.intermediateSize, bias: false),
            key: "ffn_up"
        )
        self._down = ModuleInfo(
            wrappedValue: Linear(configuration.intermediateSize, configuration.hiddenSize, bias: false),
            key: "ffn_down"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        down(geluApproximate(gate(input)) * up(input))
    }
}

final class LTXVideo096T5Block: Module {
    @ModuleInfo(key: "attn_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attn_q") var query: Linear
    @ModuleInfo(key: "attn_k") var key: Linear
    @ModuleInfo(key: "attn_v") var value: Linear
    @ModuleInfo(key: "attn_o") var output: Linear
    @ModuleInfo(key: "ffn_norm") var feedForwardNorm: RMSNorm
    @ModuleInfo(key: "ffn_gate") var gate: Linear
    @ModuleInfo(key: "ffn_up") var up: Linear
    @ModuleInfo(key: "ffn_down") var down: Linear
    @ParameterInfo(key: "attn_rel_b") var relativeAttentionBias: MLXArray

    private let normEps: Float

    init(configuration: LTXVideo096T5Configuration) {
        self.normEps = configuration.normEps
        self._attentionNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.normEps),
            key: "attn_norm"
        )
        self._query = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false),
            key: "attn_q"
        )
        self._key = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false),
            key: "attn_k"
        )
        self._value = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false),
            key: "attn_v"
        )
        self._output = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.hiddenSize, bias: false),
            key: "attn_o"
        )
        self._feedForwardNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.normEps),
            key: "ffn_norm"
        )
        self._gate = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.intermediateSize, bias: false),
            key: "ffn_gate"
        )
        self._up = ModuleInfo(
            wrappedValue: Linear(configuration.hiddenSize, configuration.intermediateSize, bias: false),
            key: "ffn_up"
        )
        self._down = ModuleInfo(
            wrappedValue: Linear(configuration.intermediateSize, configuration.hiddenSize, bias: false),
            key: "ffn_down"
        )
        self._relativeAttentionBias = ParameterInfo(
            wrappedValue: MLXArray.zeros(
                [configuration.relativeAttentionBuckets, configuration.numHeads],
                dtype: .float32
            ),
            key: "attn_rel_b"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, relativeBias: MLXArray?) -> MLXArray {
        let normalized = attentionNorm(input)
        let batch = normalized.shape[0]
        let queries = query(normalized)
            .reshaped(batch, -1, 64, normalized.shape[2] / 64)
            .transposed(0, 2, 1, 3)
        let keys = key(normalized)
            .reshaped(batch, -1, 64, normalized.shape[2] / 64)
            .transposed(0, 2, 1, 3)
        let values = value(normalized)
            .reshaped(batch, -1, 64, normalized.shape[2] / 64)
            .transposed(0, 2, 1, 3)
        let attentionOutput = output(
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: 1,
                mask: relativeBias
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, -1, normalized.shape[2])
        )
        let afterAttention = input + attentionOutput
        let feedInput = feedForwardNorm(afterAttention)
        return afterAttention + down(geluApproximate(gate(feedInput)) * up(feedInput))
    }
}

final class LTXVideo096T5EncoderStack: Module {
    @ModuleInfo(key: "blk") var blocks: [LTXVideo096T5Block]
    @ModuleInfo(key: "output_norm") var outputNorm: RMSNorm

    let configuration: LTXVideo096T5Configuration

    init(configuration: LTXVideo096T5Configuration) {
        self.configuration = configuration
        self._blocks = ModuleInfo(
            wrappedValue: (0..<configuration.numLayers).map { _ in
                LTXVideo096T5Block(configuration: configuration)
            },
            key: "blk"
        )
        self._outputNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.normEps),
            key: "output_norm"
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let sequenceLength = input.shape[1]
        let buckets = LTXVideo096T5RelativePositionBuckets.make(
            sequenceLength: sequenceLength,
            numberOfBuckets: configuration.relativeAttentionBuckets,
            maxDistance: configuration.relativeAttentionMaxDistance
        )
        let bucketArray = MLXArray(buckets.map(Int32.init), [sequenceLength, sequenceLength])
        let firstBias = MLX.take(
            blocks[0].relativeAttentionBias,
            bucketArray,
            axis: 0
        ).transposed(2, 0, 1).reshaped(
            1, configuration.numHeads, sequenceLength, sequenceLength
        )

        var value = input
        for (index, block) in blocks.enumerated() {
            value = block(
                value,
                relativeBias: index == 0 ? firstBias.asType(value.dtype) : nil
            )
        }
        return outputNorm(value)
    }
}

public final class LTXVideo096T5Encoder: Module {
    @ModuleInfo(key: "token_embd") var tokenEmbedding: Embedding
    @ModuleInfo(key: "enc") var encoder: LTXVideo096T5EncoderStack

    public let configuration: LTXVideo096T5Configuration

    public init(configuration: LTXVideo096T5Configuration = try! .init()) {
        self.configuration = configuration
        self._tokenEmbedding = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: configuration.vocabularySize,
                dimensions: configuration.hiddenSize
            ),
            key: "token_embd"
        )
        self._encoder = ModuleInfo(
            wrappedValue: LTXVideo096T5EncoderStack(configuration: configuration),
            key: "enc"
        )
        super.init()
    }

    public func callAsFunction(_ tokenIDs: MLXArray) throws -> MLXArray {
        guard tokenIDs.ndim == 2,
              tokenIDs.shape[0] > 0,
              tokenIDs.shape[1] > 0,
              tokenIDs.shape[1] <= 512 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "T5 token IDs 必須是非空的 [B,N]，且 N 不得超過 512。"
            )
        }
        return encoder(tokenEmbedding(tokenIDs.asType(.int32)))
    }
}

public enum LTXVideo096T5WeightLoader {
    public static func load(
        from weightsURL: URL,
        configuration: LTXVideo096T5Configuration = try! .init()
    ) throws -> (model: LTXVideo096T5Encoder, report: LTXVideo096GGUFLoadReport) {
        let file = try LTXVideo096GGUFFile(url: weightsURL)
        let model = LTXVideo096T5Encoder(configuration: configuration)
        let q4Names = Set(file.tensors.filter { ["Q3_K", "Q4_K"].contains($0.typeName) }
            .map { parameterPath($0.name) })
        let q8Names = Set(file.tensors.filter { ["Q5_K", "Q6_K", "IQ4_XS"].contains($0.typeName) }
            .map { parameterPath($0.name) })
        if !q4Names.isEmpty || !q8Names.isEmpty {
            ltxQuantizeSorted(model: model, groupSize: 64) { path, module in
                guard module is Linear else { return nil }
                if q4Names.contains(path) { return 4 }
                if q8Names.contains(path) { return 8 }
                return nil
            }
        }

        let expected = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) }
        )
        var converted: [String: MLXArray] = [:]
        var sourceTypes: [String: Int] = [:]
        var quantizedCount = 0
        for tensor in file.tensors {
            sourceTypes[tensor.typeName, default: 0] += 1
            let key = canonicalKey(tensor.name)
            guard expected[key] != nil else { continue }
            let source = try file.array(for: tensor)
            if let bits = LTXVideo096GGUFQuantization.targetBits(for: tensor.typeName),
               key != "token_embd.weight" {
                let quantized = MLX.quantized(
                    source.asType(.float32), groupSize: 64, bits: bits, mode: .affine
                )
                try insert(quantized.wq, name: key, expected: expected, into: &converted)
                let prefix = String(key.dropLast(".weight".count))
                try insert(quantized.scales, name: prefix + ".scales", expected: expected, into: &converted)
                if let biases = quantized.biases {
                    try insert(biases, name: prefix + ".biases", expected: expected, into: &converted)
                }
                quantizedCount += 1
            } else {
                guard source.shape == expected[key]!.shape else {
                    throw LTXVideo096GGUFError.weightShapeMismatch(
                        name: key,
                        expected: expected[key]!.shape,
                        actual: source.shape
                    )
                }
                let value = source.dtype == .uint32
                    ? source
                    : source.asType(.bfloat16)
                try insert(value, name: key, expected: expected, into: &converted)
            }
        }

        if let sharedBias = converted["enc.blk.0.attn_rel_b"] {
            for index in 1..<configuration.numLayers {
                let key = "enc.blk.\(index).attn_rel_b"
                try insert(sharedBias, name: key, expected: expected, into: &converted)
            }
        }

        let missing = Set(expected.keys).subtracting(converted.keys).sorted()
        guard missing.isEmpty else { throw LTXVideo096GGUFError.missingWeights(missing) }
        let parameterTree = ModuleParameters.unflattened(
            converted.sorted { $0.key < $1.key }
        )
        try model.update(parameters: parameterTree, verify: .all)
        MLX.eval(model)
        return (
            model,
            LTXVideo096GGUFLoadReport(
                sourceTensorCount: file.tensors.count,
                loadedParameterCount: converted.count,
                quantizedTensorCount: quantizedCount,
                sourceTypes: sourceTypes
            )
        )
    }

    private static func insert(
        _ value: MLXArray,
        name: String,
        expected: [String: MLXArray],
        into converted: inout [String: MLXArray]
    ) throws {
        guard expected[name] != nil else { return }
        guard converted[name] == nil else { throw LTXVideo096GGUFError.duplicateWeight(name) }
        guard value.shape == expected[name]!.shape else {
            throw LTXVideo096GGUFError.weightShapeMismatch(
                name: name,
                expected: expected[name]!.shape,
                actual: value.shape
            )
        }
        converted[name] = value
    }

    private static func parameterPath(_ name: String) -> String {
        name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count))
            : name
    }

    private static func canonicalKey(_ name: String) -> String {
        name.hasSuffix(".attn_rel_b.weight")
            ? String(name.dropLast(".weight".count))
            : name
    }
}

public enum LTXVideo096T5RelativePositionBuckets {
    public static func make(
        sequenceLength: Int,
        numberOfBuckets: Int,
        maxDistance: Int
    ) -> [Int] {
        precondition(sequenceLength > 0)
        precondition(numberOfBuckets > 0 && numberOfBuckets.isMultiple(of: 2))
        let halfBuckets = numberOfBuckets / 2
        let maxExact = halfBuckets / 2
        let logarithmicDenominator = Foundation.log(
            Double(maxDistance) / Double(maxExact)
        )
        var result: [Int] = []
        result.reserveCapacity(sequenceLength * sequenceLength)
        for query in 0..<sequenceLength {
            for key in 0..<sequenceLength {
                let relative = key - query
                let direction = relative < 0 ? halfBuckets : 0
                let distance = abs(relative)
                if distance < maxExact {
                    result.append(direction + distance)
                } else {
                    let logarithmic = maxExact + Int(
                        Foundation.log(Double(max(distance, 1) / maxExact))
                            / logarithmicDenominator
                            * Double(halfBuckets - maxExact)
                    )
                    result.append(direction + min(halfBuckets - 1, logarithmic))
                }
            }
        }
        return result
    }
}
