import Foundation

/// Geometry and normalization constants for the MiniMax H3 video VAE.
///
/// Values mirror `MiniMaxH3VideoVAE.__init__` and `ViT3DDecoder.__init__` in the
/// ComfyUI reference (`vae.py`), and are re-checked against the real checkpoint
/// by `validate(against:)`.
public struct MiniMaxH3VideoVAEConfiguration: Sendable, Equatable {
    public var latentChannels: Int
    public var outputChannels: Int
    public var layerCount: Int
    public var headCount: Int
    public var headDim: Int
    /// Feed-forward multiplier; `w1` emits `2 * mult * dim` because the FFN is
    /// SiLU-gated.
    public var ffnMultiplier: Int
    public var registerTokenCount: Int
    /// Spatial downsampling factor, `prod(space_down)` = 16.
    public var spatialRatio: Int
    /// Temporal downsampling factor, `prod(time_down)` = 4.
    public var temporalRatio: Int
    public var ropeTheta: Float
    /// Fraction of each head's width that carries rotary position, 0.75.
    public var ropeDimRatio: Float
    public var eps: Float

    public var hiddenSize: Int { headCount * headDim }
    /// Width of the rotary section of a head, `headDim * ropeDimRatio` = 48.
    public var ropeDim: Int { Int(Float(headDim) * ropeDimRatio) }
    /// Number of rotation pairs, 24 — half the rotary width, since the rope is
    /// split-half rather than interleaved.
    public var ropePairCount: Int { ropeDim / 2 }
    /// Elements each token decodes to: `3 * 4 * 16 * 16`.
    public var patchOutputDim: Int {
        outputChannels * temporalRatio * spatialRatio * spatialRatio
    }
    /// Register tokens plus the single zero token appended after them.
    public var suffixTokenCount: Int { registerTokenCount + 1 }

    public static let `default` = MiniMaxH3VideoVAEConfiguration(
        latentChannels: 24,
        outputChannels: 3,
        layerCount: 36,
        headCount: 32,
        headDim: 64,
        ffnMultiplier: 4,
        registerTokenCount: 4,
        spatialRatio: 16,
        temporalRatio: 4,
        ropeTheta: 100.0,
        ropeDimRatio: 0.75,
        eps: 1e-5
    )

    public init(
        latentChannels: Int,
        outputChannels: Int,
        layerCount: Int,
        headCount: Int,
        headDim: Int,
        ffnMultiplier: Int,
        registerTokenCount: Int,
        spatialRatio: Int,
        temporalRatio: Int,
        ropeTheta: Float,
        ropeDimRatio: Float,
        eps: Float
    ) {
        self.latentChannels = latentChannels
        self.outputChannels = outputChannels
        self.layerCount = layerCount
        self.headCount = headCount
        self.headDim = headDim
        self.ffnMultiplier = ffnMultiplier
        self.registerTokenCount = registerTokenCount
        self.spatialRatio = spatialRatio
        self.temporalRatio = temporalRatio
        self.ropeTheta = ropeTheta
        self.ropeDimRatio = ropeDimRatio
        self.eps = eps
    }

    /// Per-channel latent statistics, applied before decoding.
    public static let latentsMean: [Float] = [
        0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
        -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
        -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
        -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
        -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
        -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
    ]

    public static let latentsStd: [Float] = [
        1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
        1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
        0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
        0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
        2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
        3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525
    ]

    /// ImageNet statistics used to map decoder output back to [0, 1] pixels.
    public static let pixelMean: [Float] = [0.485, 0.456, 0.406]
    public static let pixelStd: [Float] = [0.229, 0.224, 0.225]

    /// Every decoder tensor and the shape it must have.
    public var expectedDecoderShapes: [String: [Int]] {
        var expected: [String: [Int]] = [
            // A 1x1x1 Conv3d applied to the latent before the ViT decoder;
            // effectively a learned channel mix, and required for the decoder
            // to receive its expected input basis.
            "post_quant_conv.weight": [latentChannels, latentChannels, 1, 1, 1],
            "post_quant_conv.bias": [latentChannels],
            "decoder.x_embedder.weight": [hiddenSize, latentChannels],
            "decoder.x_embedder.bias": [hiddenSize],
            "decoder.register_tokens": [1, registerTokenCount, hiddenSize],
            "decoder.mask_token": [1, 1, hiddenSize],
            "decoder.norm_out.weight": [hiddenSize],
            "decoder.norm_out.bias": [hiddenSize],
            "decoder.proj_out.weight": [patchOutputDim, hiddenSize],
            "decoder.proj_out.bias": [patchOutputDim]
        ]
        let ffnInner = hiddenSize * ffnMultiplier
        for layer in 0 ..< layerCount {
            let prefix = "decoder.transformer_blocks.\(layer)"
            expected["\(prefix).attn.to_qkv.weight"] = [hiddenSize * 3, hiddenSize]
            expected["\(prefix).attn.to_qkv.bias"] = [hiddenSize * 3]
            expected["\(prefix).attn.to_out.weight"] = [hiddenSize, hiddenSize]
            expected["\(prefix).attn.to_out.bias"] = [hiddenSize]
            expected["\(prefix).ff.w1.weight"] = [ffnInner * 2, hiddenSize]
            expected["\(prefix).ff.w1.bias"] = [ffnInner * 2]
            expected["\(prefix).ff.w2.weight"] = [hiddenSize, ffnInner]
            expected["\(prefix).ff.w2.bias"] = [hiddenSize]
            expected["\(prefix).norm1.weight"] = [hiddenSize]
            expected["\(prefix).norm2.weight"] = [hiddenSize]
            expected["\(prefix).scale1"] = [hiddenSize]
            expected["\(prefix).scale2"] = [hiddenSize]
        }
        return expected
    }

    /// Verify a checkpoint matches this configuration before any weights are used.
    public func validate(against shapes: [String: [Int]]) throws {
        for (name, expectedShape) in expectedDecoderShapes.sorted(by: { $0.key < $1.key }) {
            guard let actual = shapes[name] else {
                throw MiniMaxH3WeightError.missingTensor(name)
            }
            guard actual == expectedShape else {
                throw MiniMaxH3WeightError.unexpectedShape(
                    name, expected: expectedShape, actual: actual
                )
            }
        }
    }
}
