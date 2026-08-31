import Foundation

/// Architecture parameters for the MiniMax H3 FL2VA transformer.
///
/// Every value here is cross-checked against two independent sources: the
/// tensor shapes in the real GGUF, and the module definitions in the ComfyUI
/// reference (`model.py`). `validate(against:)` re-checks them at load time so
/// a different checkpoint cannot be silently misinterpreted.
public struct MiniMaxH3Configuration: Sendable, Equatable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var attentionHeadCount: Int
    public var attentionHeadDim: Int
    /// Feed-forward width. `fc1` projects to `2 * ffnHiddenSize` because the
    /// MLP is SwiGLU-gated (`model.py` uses `linear_input_act(..., "swiglu")`).
    public var ffnHiddenSize: Int
    public var timeEmbedDim: Int
    /// Pruned checkpoints replace the timestep MLP with a 1025-point table.
    /// Zero means that the full timestep embedder is present.
    public var adalnCurveGrid: Int
    /// `AdalnProj(t_dim, hidden, 6, 3)` in the reference: six modulation
    /// parameters across three modalities.
    public var adalnExpand: Int
    public var adalnModalities: Int
    /// `FinalLayer` uses `AdalnProj(t_dim, hidden, 2, 1)`.
    public var finalAdalnExpand: Int
    public var finalAdalnModalities: Int
    public var videoLatentChannels: Int
    public var audioLatentChannels: Int
    public var patchSize: (frames: Int, height: Int, width: Int)
    /// Width of the text encoder hidden states fed into `condition_proj`.
    public var conditionInputDim: Int
    public var ropeInvFreqLength: Int
    public var tokenRefinerLayerCount: Int
    public var normEps: Float
    public var qkNormEps: Float

    /// Video and audio use different flow-matching shift values.
    public var videoShift: Float
    public var audioShift: Float

    /// Total attention width, `heads * headDim`. Note this is *not* equal to
    /// `hiddenSize` for H3 (7168 vs 5376), so the projections are not square.
    public var attentionInnerDim: Int { attentionHeadCount * attentionHeadDim }

    /// Elements per video patch token, i.e. the `video_patch_proj` input width.
    public var videoPatchDim: Int {
        videoLatentChannels * patchSize.frames * patchSize.height * patchSize.width
    }

    public var adalnOutputDim: Int { adalnExpand * adalnModalities * hiddenSize }
    public var finalAdalnOutputDim: Int {
        finalAdalnExpand * finalAdalnModalities * hiddenSize
    }

    public static let fl2va = MiniMaxH3Configuration(
        hiddenSize: 5376,
        layerCount: 50,
        attentionHeadCount: 56,
        attentionHeadDim: 128,
        ffnHiddenSize: 14336,
        timeEmbedDim: 2688,
        adalnCurveGrid: 0,
        adalnExpand: 6,
        adalnModalities: 3,
        finalAdalnExpand: 2,
        finalAdalnModalities: 1,
        videoLatentChannels: 24,
        audioLatentChannels: 32,
        patchSize: (frames: 1, height: 2, width: 2),
        conditionInputDim: 5120,
        ropeInvFreqLength: 16,
        tokenRefinerLayerCount: 2,
        normEps: 1e-5,
        qkNormEps: 1e-5,
        videoShift: 12.0,
        audioShift: 3.0
    )

    public init(
        hiddenSize: Int,
        layerCount: Int,
        attentionHeadCount: Int,
        attentionHeadDim: Int,
        ffnHiddenSize: Int,
        timeEmbedDim: Int,
        adalnCurveGrid: Int = 0,
        adalnExpand: Int,
        adalnModalities: Int,
        finalAdalnExpand: Int,
        finalAdalnModalities: Int,
        videoLatentChannels: Int,
        audioLatentChannels: Int,
        patchSize: (frames: Int, height: Int, width: Int),
        conditionInputDim: Int,
        ropeInvFreqLength: Int,
        tokenRefinerLayerCount: Int,
        normEps: Float,
        qkNormEps: Float,
        videoShift: Float,
        audioShift: Float
    ) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.attentionHeadCount = attentionHeadCount
        self.attentionHeadDim = attentionHeadDim
        self.ffnHiddenSize = ffnHiddenSize
        self.timeEmbedDim = timeEmbedDim
        self.adalnCurveGrid = adalnCurveGrid
        self.adalnExpand = adalnExpand
        self.adalnModalities = adalnModalities
        self.finalAdalnExpand = finalAdalnExpand
        self.finalAdalnModalities = finalAdalnModalities
        self.videoLatentChannels = videoLatentChannels
        self.audioLatentChannels = audioLatentChannels
        self.patchSize = patchSize
        self.conditionInputDim = conditionInputDim
        self.ropeInvFreqLength = ropeInvFreqLength
        self.tokenRefinerLayerCount = tokenRefinerLayerCount
        self.normEps = normEps
        self.qkNormEps = qkNormEps
        self.videoShift = videoShift
        self.audioShift = audioShift
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.hiddenSize == rhs.hiddenSize
            && lhs.layerCount == rhs.layerCount
            && lhs.attentionHeadCount == rhs.attentionHeadCount
            && lhs.attentionHeadDim == rhs.attentionHeadDim
            && lhs.ffnHiddenSize == rhs.ffnHiddenSize
            && lhs.timeEmbedDim == rhs.timeEmbedDim
            && lhs.adalnCurveGrid == rhs.adalnCurveGrid
            && lhs.adalnExpand == rhs.adalnExpand
            && lhs.adalnModalities == rhs.adalnModalities
            && lhs.finalAdalnExpand == rhs.finalAdalnExpand
            && lhs.finalAdalnModalities == rhs.finalAdalnModalities
            && lhs.videoLatentChannels == rhs.videoLatentChannels
            && lhs.audioLatentChannels == rhs.audioLatentChannels
            && lhs.patchSize == rhs.patchSize
            && lhs.conditionInputDim == rhs.conditionInputDim
            && lhs.ropeInvFreqLength == rhs.ropeInvFreqLength
            && lhs.tokenRefinerLayerCount == rhs.tokenRefinerLayerCount
            && lhs.videoShift == rhs.videoShift
            && lhs.audioShift == rhs.audioShift
    }
}

public extension MiniMaxH3Configuration {
    /// The Pruned GGUF stores a compact timestep curve instead of the full
    /// timestep MLP. Its remaining transformer weights use the same shapes and
    /// block logic as the regular FL2VA checkpoint.
    static let fl2vaPruned = MiniMaxH3Configuration(
        hiddenSize: 5376,
        layerCount: 50,
        attentionHeadCount: 56,
        attentionHeadDim: 128,
        ffnHiddenSize: 14336,
        timeEmbedDim: 8,
        adalnCurveGrid: 1025,
        adalnExpand: 6,
        adalnModalities: 3,
        finalAdalnExpand: 2,
        finalAdalnModalities: 1,
        videoLatentChannels: 24,
        audioLatentChannels: 32,
        patchSize: (frames: 1, height: 2, width: 2),
        conditionInputDim: 5120,
        ropeInvFreqLength: 16,
        tokenRefinerLayerCount: 2,
        normEps: 1e-5,
        qkNormEps: 1e-5,
        videoShift: 12.0,
        audioShift: 3.0
    )

    public var usesAdalnCurves: Bool { adalnCurveGrid > 0 }

    /// Every tensor the transformer requires, with the shape it must have.
    ///
    /// Shapes are row-major `[out, in]` for linear weights, matching what
    /// `GGUFModelLoader` produces once ComfyUI overrides are applied.
    var expectedTensorShapes: [String: [Int]] {
        var expected: [String: [Int]] = [
            // Patch projections read *into* the hidden width
            // (`Linear(video_patch_dim, hidden)`), whereas `final_layer.*_out`
            // projects back out of it. The two are transposes of each other, so
            // the direction matters.
            "video_patch_proj.weight": [hiddenSize, videoPatchDim],
            "video_patch_proj.bias": [hiddenSize],
            "audio_patch_proj.weight": [hiddenSize, audioLatentChannels],
            "audio_patch_proj.bias": [hiddenSize],
            "condition_proj.weight": [hiddenSize, conditionInputDim],
            "condition_proj.bias": [hiddenSize],
            "rope.inv_freq": [ropeInvFreqLength],
            "final_layer.norm.weight": [hiddenSize],
            "final_layer.adaln_proj.linear.weight": [finalAdalnOutputDim, timeEmbedDim],
            "final_layer.adaln_proj.linear.bias": [finalAdalnOutputDim],
            "final_layer.video_out.weight": [videoPatchDim, hiddenSize],
            "final_layer.video_out.bias": [videoPatchDim],
            "final_layer.audio_out.weight": [audioLatentChannels, hiddenSize],
            "final_layer.audio_out.bias": [audioLatentChannels],
            "token_refiner.final_norm.weight": [hiddenSize]
        ]

        if usesAdalnCurves {
            expected["adaln_t_table"] = [adalnCurveGrid, timeEmbedDim]
        } else {
            expected["time_embedder.proj_in.weight"] = [hiddenSize, 256]
            expected["time_embedder.proj_in.bias"] = [hiddenSize]
            expected["time_embedder.proj_out.weight"] = [timeEmbedDim, hiddenSize]
            expected["time_embedder.proj_out.bias"] = [timeEmbedDim]
        }

        for layer in 0 ..< tokenRefinerLayerCount {
            let prefix = "token_refiner.blocks.\(layer)"
            expected["\(prefix).attn.qkv_proj.weight"] =
                [attentionInnerDim * 3, hiddenSize]
            expected["\(prefix).attn.out_proj.weight"] =
                [hiddenSize, attentionInnerDim]
            expected["\(prefix).attn.q_norm.weight"] = [attentionHeadDim]
            expected["\(prefix).attn.k_norm.weight"] = [attentionHeadDim]
            expected["\(prefix).mlp.fc1.weight"] = [ffnHiddenSize * 2, hiddenSize]
            expected["\(prefix).mlp.fc2.weight"] = [hiddenSize, ffnHiddenSize]
            expected["\(prefix).norm1.weight"] = [hiddenSize]
            expected["\(prefix).norm2.weight"] = [hiddenSize]
        }

        for layer in 0 ..< layerCount {
            let prefix = "blocks.\(layer)"
            expected["\(prefix).attn.qkv_proj.weight"] =
                [attentionInnerDim * 3, hiddenSize]
            expected["\(prefix).attn.out_proj.weight"] =
                [hiddenSize, attentionInnerDim]
            expected["\(prefix).attn.q_norm.weight"] = [attentionHeadDim]
            expected["\(prefix).attn.k_norm.weight"] = [attentionHeadDim]
            expected["\(prefix).mlp.fc1.weight"] = [ffnHiddenSize * 2, hiddenSize]
            expected["\(prefix).mlp.fc2.weight"] = [hiddenSize, ffnHiddenSize]
            expected["\(prefix).norm1.weight"] = [hiddenSize]
            expected["\(prefix).norm2.weight"] = [hiddenSize]
            expected["\(prefix).adaln_proj.linear.weight"] =
                [adalnOutputDim, timeEmbedDim]
            expected["\(prefix).adaln_proj.linear.bias"] = [adalnOutputDim]
        }

        return expected
    }

    /// Check a real checkpoint against this configuration.
    ///
    /// Throws on the first missing tensor or shape disagreement, so a
    /// checkpoint that does not match cannot be loaded and quietly produce
    /// wrong output.
    func validate(against inventory: MiniMaxH3WeightInventory) throws {
        let actual = inventory.entries.reduce(into: [String: [Int]]()) {
            $0[$1.name] = $1.logicalShape
        }
        for (name, expectedShape) in expectedTensorShapes.sorted(by: { $0.key < $1.key }) {
            guard let actualShape = actual[name] else {
                throw MiniMaxH3WeightError.missingTensor(name)
            }
            guard actualShape == expectedShape else {
                throw MiniMaxH3WeightError.unexpectedShape(
                    name,
                    expected: expectedShape,
                    actual: actualShape
                )
            }
        }
    }

    /// Detect the checkpoint family from its header inventory and validate it
    /// before any large tensor is materialized.
    public static func forInventory(
        _ inventory: MiniMaxH3WeightInventory
    ) throws -> MiniMaxH3Configuration {
        let configuration = inventory.entry(named: "adaln_t_table") == nil
            ? Self.fl2va
            : Self.fl2vaPruned
        try configuration.validate(against: inventory)
        return configuration
    }
}
