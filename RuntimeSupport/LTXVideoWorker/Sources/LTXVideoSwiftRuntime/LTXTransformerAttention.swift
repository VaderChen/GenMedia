import Foundation
import MLX
import MLXNN

public final class LTXAttention: Module {
    @ModuleInfo(key: "to_q") public var toQ: Linear
    @ModuleInfo(key: "to_k") public var toK: Linear
    @ModuleInfo(key: "to_v") public var toV: Linear
    @ModuleInfo(key: "to_out") public var toOut: Linear
    @ModuleInfo(key: "to_gate_logits") public var toGateLogits: Linear?
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    public let numHeads: Int
    public let headDimension: Int
    public let useRoPE: Bool
    public let scale: Float

    public init(
        queryDimension: Int,
        keyValueDimension: Int? = nil,
        outputDimension: Int? = nil,
        numHeads: Int,
        headDimension: Int,
        useRoPE: Bool,
        normEps: Float,
        gated: Bool = true
    ) {
        let keyValueDimension = keyValueDimension ?? queryDimension
        let outputDimension = outputDimension ?? queryDimension
        self.numHeads = numHeads
        self.headDimension = headDimension
        self.useRoPE = useRoPE
        self.scale = Float(1 / Foundation.sqrt(Double(headDimension)))
        self._toQ = ModuleInfo(
            wrappedValue: Linear(queryDimension, numHeads * headDimension), key: "to_q"
        )
        self._toK = ModuleInfo(
            wrappedValue: Linear(keyValueDimension, numHeads * headDimension), key: "to_k"
        )
        self._toV = ModuleInfo(
            wrappedValue: Linear(keyValueDimension, numHeads * headDimension), key: "to_v"
        )
        self._toOut = ModuleInfo(
            wrappedValue: Linear(numHeads * headDimension, outputDimension), key: "to_out"
        )
        self._toGateLogits = ModuleInfo(
            wrappedValue: gated ? Linear(queryDimension, numHeads) : nil,
            key: "to_gate_logits"
        )
        self._qNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: numHeads * headDimension, eps: normEps),
            key: "q_norm"
        )
        self._kNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: numHeads * headDimension, eps: normEps),
            key: "k_norm"
        )
        super.init()
    }

    public func callAsFunction(
        _ input: MLXArray,
        encoderHiddenStates: MLXArray? = nil,
        rope: LTXRoPEFrequencies? = nil,
        keyRoPE: LTXRoPEFrequencies? = nil,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let batch = input.shape[0]
        let keyValue = encoderHiddenStates ?? input
        var query = qNorm(toQ(input))
        var key = kNorm(toK(keyValue))
        var value = toV(keyValue)

        query = query.reshaped(batch, -1, numHeads, headDimension)
            .transposed(0, 2, 1, 3)
        key = key.reshaped(batch, -1, numHeads, headDimension)
            .transposed(0, 2, 1, 3)
        value = value.reshaped(batch, -1, numHeads, headDimension)
            .transposed(0, 2, 1, 3)

        if useRoPE, let rope {
            query = LTXTransformerOps.applyRoPE(query, frequencies: rope)
            key = LTXTransformerOps.applyRoPE(key, frequencies: keyRoPE ?? rope)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: scale,
            mask: mask
        )
        if let toGateLogits {
            let gate = (2 * sigmoid(toGateLogits(input)))
                .transposed(0, 2, 1)
                .reshaped(batch, numHeads, input.shape[1], 1)
            output = output * gate
        }
        return toOut(output.transposed(0, 2, 1, 3).reshaped(batch, -1, numHeads * headDimension))
    }
}

public final class LTXFeedForward: Module {
    @ModuleInfo(key: "proj_in") public var projIn: Linear
    @ModuleInfo(key: "proj_out") public var projOut: Linear

    public init(dimension: Int, multiplier: Float) {
        self._projIn = ModuleInfo(
            wrappedValue: Linear(dimension, Int(Float(dimension) * multiplier)), key: "proj_in"
        )
        self._projOut = ModuleInfo(
            wrappedValue: Linear(Int(Float(dimension) * multiplier), dimension), key: "proj_out"
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        projOut(geluApproximate(projIn(input)))
    }
}
