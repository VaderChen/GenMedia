import Foundation
import MLX
import MLXNN
import MLXRandom

/// The MiniMax H3 DiT.
///
/// Ported from `MiniMaxH3Model` in the ComfyUI reference (`model.py`), covering
/// the base text-to-video-audio path: one packed sequence of
/// `[text, audio, video]`, three AdaLN modalities, 3-axis split-half RoPE, a
/// two-layer token refiner and 50 DiT blocks.
///
/// Not modelled yet: Ref2VA reference blocks, denoise masks, and the PDD head
/// bank (this checkpoint has a single head, so `n == 1`).
public final class MiniMaxH3Transformer {
    public let configuration: MiniMaxH3Configuration
    private let weights: [String: MLXArray]
    private let quantizedPrefixes: Set<String>
    private let computeDType: DType

    public init(
        configuration: MiniMaxH3Configuration = .fl2va,
        weights: [String: MLXArray],
        quantizedPrefixes: Set<String> = [],
        computeDType: DType = .float32
    ) {
        self.configuration = configuration
        self.weights = weights
        self.quantizedPrefixes = quantizedPrefixes
        self.computeDType = computeDType
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let value = weights[name] else {
            throw MiniMaxH3WeightError.missingTensor(name)
        }
        return value
    }

    /// `y = x W^T (+ b)`, transparently handling quantized weights.
    private func linear(
        _ input: MLXArray,
        _ prefix: String,
        bias hasBias: Bool = false
    ) throws -> MLXArray {
        var output: MLXArray
        if quantizedPrefixes.contains(prefix) {
            output = MLX.quantizedMatmul(
                input,
                try weight("\(prefix).weight"),
                scales: try weight("\(prefix).scales"),
                biases: weights["\(prefix).biases"],
                transpose: true,
                groupSize: MiniMaxH3GGUFQuantizedLoader.groupSize,
                bits: MiniMaxH3GGUFQuantizedLoader.targetBits
            )
        } else {
            let matrix = try weight("\(prefix).weight").asType(input.dtype)
            output = input.matmul(matrix.transposed())
        }
        if hasBias {
            output = output + (try weight("\(prefix).bias")).asType(output.dtype)
        }
        return output
    }

    // MARK: - Primitives

    private func rmsNorm(_ input: MLXArray, weightName: String) throws -> MLXArray {
        let scale = try weight(weightName).asType(input.dtype)
        return rmsNorm(input) * scale
    }

    private func rmsNorm(_ input: MLXArray) -> MLXArray {
        let meanSquare = MLX.mean(input.asType(.float32) * input.asType(.float32),
                                  axis: -1, keepDims: true)
        let normalized = input.asType(.float32) * MLX.rsqrt(meanSquare + configuration.normEps)
        return normalized.asType(input.dtype)
    }

    /// Sinusoidal timestep features. The reference emits **cos before sin**.
    func timestepFeatures(_ timesteps: MLXArray) -> MLXArray {
        let half = 256 / 2
        let scale = -Foundation.log(10000.0) / Double(half)
        let frequencies = MLXArray(
            (0 ..< half).map { Float(Foundation.exp(Double($0) * scale)) },
            [1, half]
        )
        let arguments = timesteps.asType(.float32).reshaped(-1, 1) * frequencies
        return MLX.concatenated([MLX.cos(arguments), MLX.sin(arguments)], axis: -1)
    }

    /// `proj_out(silu(proj_in(features)))`.
    private func timeEmbedding(_ timesteps: MLXArray) throws -> MLXArray {
        let features = timestepFeatures(timesteps)
        let hidden = MLXNN.silu(try linear(features, "time_embedder.proj_in", bias: true))
        return try linear(hidden, "time_embedder.proj_out", bias: true)
    }

    /// Interpolate the compact AdaLN timestep table used by Pruned GGUF
    /// checkpoints. The reference clamps the lower index to the final pair of
    /// rows, so an exact timestep of 1.0 still selects the last row exactly.
    private func adalnCurveEmbedding(_ timesteps: [Float]) throws -> MLXArray {
        let table = try weight("adaln_t_table")
        let grid = configuration.adalnCurveGrid
        guard grid >= 2, table.shape == [grid, configuration.timeEmbedDim] else {
            throw MiniMaxH3WeightError.unexpectedShape(
                "adaln_t_table",
                expected: [grid, configuration.timeEmbedDim],
                actual: table.shape
            )
        }

        var lowerIndices: [Int32] = []
        var upperIndices: [Int32] = []
        var fractions: [Float] = []
        lowerIndices.reserveCapacity(timesteps.count)
        upperIndices.reserveCapacity(timesteps.count)
        fractions.reserveCapacity(timesteps.count)

        for timestep in timesteps {
            let position = min(max(Double(timestep), 0), 1) * Double(grid - 1)
            let lower = min(Int(floor(position)), grid - 2)
            lowerIndices.append(Int32(lower))
            upperIndices.append(Int32(lower + 1))
            fractions.append(Float(position - Double(lower)))
        }

        let lower = table.take(MLXArray(lowerIndices), axis: 0)
        let upper = table.take(MLXArray(upperIndices), axis: 0)
        let fraction = MLXArray(fractions, [timesteps.count, 1])
        return lower + (upper - lower) * fraction
    }

    /// AdaLN parameters, returned as `expand` tensors of `[M * modalities, hidden]`.
    private func adalnParameters(
        _ timeEmbedding: MLXArray,
        prefix: String,
        expand: Int,
        modalities: Int
    ) throws -> [MLXArray] {
        let projected = try linear(
            MLXNN.silu(timeEmbedding), "\(prefix).linear", bias: true
        )
        let rows = projected.shape[0] * modalities
        let reshaped = projected.reshaped(rows, expand * configuration.hiddenSize)
        return (0 ..< expand).map { index in
            reshaped[0..., (index * configuration.hiddenSize)
                ..< ((index + 1) * configuration.hiddenSize)]
        }
    }

    // MARK: - RoPE

    /// Per-token rotation angles, `[S, ropePairs]`.
    ///
    /// The reference builds `[S, 96]` by concatenating the 16 inverse
    /// frequencies for each of t/h/w and then duplicating the halves; the
    /// rotation table only consumes the first half, so the unique angles are
    /// the `[S, 48]` prefix.
    func ropeAngles(positionIDs: MLXArray) throws -> MLXArray {
        let inverseFrequency = try weight("rope.inv_freq").asType(.float32)
        let sequence = positionIDs.shape[0]
        let perAxis = positionIDs.asType(.float32).reshaped(sequence, 3, 1)
            * inverseFrequency.reshaped(1, 1, inverseFrequency.shape[0])
        return perAxis.reshaped(sequence, 3 * inverseFrequency.shape[0])
    }

    /// Rotate the leading `2 * pairs` channels of each head, split-half.
    func applyRoPE(_ input: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let pairs = cos.shape[1]
        let rotated = pairs * 2
        let headDim = input.shape[input.shape.count - 1]
        let cosine = cos.reshaped(1, cos.shape[0], 1, pairs).asType(input.dtype)
        let sine = sin.reshaped(1, sin.shape[0], 1, pairs).asType(input.dtype)
        let first = input[.ellipsis, 0 ..< pairs]
        let second = input[.ellipsis, pairs ..< rotated]
        let outFirst = first * cosine - second * sine
        let outSecond = first * sine + second * cosine
        if rotated == headDim {
            return MLX.concatenated([outFirst, outSecond], axis: -1)
        }
        return MLX.concatenated(
            [outFirst, outSecond, input[.ellipsis, rotated ..< headDim]], axis: -1
        )
    }

    // MARK: - Blocks

    private func attention(
        _ input: MLXArray,
        prefix: String,
        rope: (cos: MLXArray, sin: MLXArray)?
    ) throws -> MLXArray {
        let sequence = input.shape[0]
        let heads = configuration.attentionHeadCount
        let headDim = configuration.attentionHeadDim
        let inner = heads * headDim

        let qkv = try linear(input, "\(prefix).qkv_proj")
        // Contiguous q | k | v blocks, unlike the video VAE's per-head interleave.
        var query = qkv[0..., 0 ..< inner].reshaped(1, sequence, heads, headDim)
        var key = qkv[0..., inner ..< (2 * inner)].reshaped(1, sequence, heads, headDim)
        let value = qkv[0..., (2 * inner) ..< (3 * inner)]
            .reshaped(1, sequence, heads, headDim)

        query = try rmsNorm(query, weightName: "\(prefix).q_norm.weight")
        key = try rmsNorm(key, weightName: "\(prefix).k_norm.weight")
        if let rope {
            query = applyRoPE(query, cos: rope.cos, sin: rope.sin)
            key = applyRoPE(key, cos: rope.cos, sin: rope.sin)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: query.transposed(0, 2, 1, 3),
            keys: key.transposed(0, 2, 1, 3),
            values: value.transposed(0, 2, 1, 3),
            scale: 1.0 / Foundation.sqrt(Float(headDim)),
            mask: nil
        )
        let merged = output.transposed(0, 2, 1, 3).reshaped(sequence, inner)
        return try linear(merged, "\(prefix).out_proj")
    }

    /// SwiGLU: `fc2(silu(gate) * value)` where `fc1` emits both halves.
    private func feedForward(_ input: MLXArray, prefix: String) throws -> MLXArray {
        let projected = try linear(input, "\(prefix).fc1")
        let inner = configuration.ffnHiddenSize
        let gate = projected[0..., 0 ..< inner]
        let value = projected[0..., inner ..< (2 * inner)]
        return try linear(MLXNN.silu(gate) * value, "\(prefix).fc2")
    }

    /// Apply `x * (1 + scale) + shift` per modulation segment.
    private func modulate(
        _ input: MLXArray,
        shift: MLXArray,
        scale: MLXArray,
        segments: [(start: Int, end: Int, row: Int)]
    ) -> MLXArray {
        var pieces: [MLXArray] = []
        pieces.reserveCapacity(segments.count)
        for segment in segments {
            let slice = input[segment.start ..< segment.end]
            let rowScale = scale[segment.row ..< (segment.row + 1)].asType(slice.dtype)
            let rowShift = shift[segment.row ..< (segment.row + 1)].asType(slice.dtype)
            pieces.append(slice * (1.0 + rowScale) + rowShift)
        }
        return MLX.concatenated(pieces, axis: 0)
    }

    /// Accumulate `x + gate * other` per modulation segment.
    private func gatedResidual(
        _ input: MLXArray,
        gate: MLXArray,
        other: MLXArray,
        segments: [(start: Int, end: Int, row: Int)]
    ) -> MLXArray {
        var pieces: [MLXArray] = []
        pieces.reserveCapacity(segments.count)
        for segment in segments {
            let base = input[segment.start ..< segment.end]
            let delta = other[segment.start ..< segment.end]
            let rowGate = gate[segment.row ..< (segment.row + 1)].asType(base.dtype)
            pieces.append(base + delta * rowGate)
        }
        return MLX.concatenated(pieces, axis: 0)
    }

    private func refinerBlock(_ input: MLXArray, layer: Int) throws -> MLXArray {
        let prefix = "token_refiner.blocks.\(layer)"
        var hidden = input
        hidden = hidden + (try attention(
            try rmsNorm(hidden, weightName: "\(prefix).norm1.weight"),
            prefix: "\(prefix).attn",
            rope: nil
        ))
        hidden = hidden + (try feedForward(
            try rmsNorm(hidden, weightName: "\(prefix).norm2.weight"),
            prefix: "\(prefix).mlp"
        ))
        return hidden
    }

    private func ditBlock(
        _ input: MLXArray,
        layer: Int,
        timeEmbedding: MLXArray,
        segments: [(start: Int, end: Int, row: Int)],
        rope: (cos: MLXArray, sin: MLXArray)
    ) throws -> MLXArray {
        let prefix = "blocks.\(layer)"
        let parameters = try adalnParameters(
            timeEmbedding, prefix: "\(prefix).adaln_proj",
            expand: configuration.adalnExpand,
            modalities: configuration.adalnModalities
        )
        var hidden = input
        var normed = modulate(
            try rmsNorm(hidden, weightName: "\(prefix).norm1.weight"),
            shift: parameters[0], scale: parameters[1], segments: segments
        )
        hidden = gatedResidual(
            hidden, gate: parameters[2],
            other: try attention(normed, prefix: "\(prefix).attn", rope: rope),
            segments: segments
        )
        normed = modulate(
            try rmsNorm(hidden, weightName: "\(prefix).norm2.weight"),
            shift: parameters[3], scale: parameters[4], segments: segments
        )
        hidden = gatedResidual(
            hidden, gate: parameters[5],
            other: try feedForward(normed, prefix: "\(prefix).mlp"),
            segments: segments
        )
        return hidden
    }

    /// `condition_proj` followed by the token refiner stack.
    ///
    /// Exposed on its own so it can be checked against the reference without
    /// materializing the whole 50-block transformer.
    public func refineTextStates(_ textStates: MLXArray) throws -> MLXArray {
        var text = try linear(textStates, "condition_proj", bias: true)
        for layer in 0 ..< configuration.tokenRefinerLayerCount {
            text = try refinerBlock(text, layer: layer)
        }
        return try rmsNorm(text, weightName: "token_refiner.final_norm.weight")
    }

    // MARK: - Forward

    public struct Output {
        /// `[1, 24, T, H, W]`
        public let video: MLXArray
        /// `[1, 32, 2, audioFrames]`
        public let audio: MLXArray
    }

    /// Run the packed-sequence path through the first `layerCount` DiT blocks.
    ///
    /// Used for parity work: the full 50-block stack cannot be held at float32
    /// alongside a reference process, but a couple of blocks can, which isolates
    /// AdaLN modulation, RoPE and the gated residuals from quantization error.
    public func forwardPartial(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        refinedText: MLXArray,
        sigma: Float,
        layerCount: Int,
        conditioning: Conditioning? = nil
    ) throws -> MLXArray {
        let context = try prepare(
            videoLatent: videoLatent,
            audioLatent: audioLatent,
            refinedText: refinedText,
            sigma: sigma,
            conditioning: conditioning
        )
        var hidden = context.hidden
        for layer in 0 ..< layerCount {
            hidden = try ditBlock(
                hidden, layer: layer, timeEmbedding: context.timeEmbedding,
                segments: context.segments, rope: context.rope
            )
        }
        return hidden
    }

    /// Keyframe conditioning for FL2VA.
    ///
    /// Latents are already encoded; the layout places them right after the text
    /// and pins them near sigma 1 so they are never denoised.
    public struct Conditioning {
        public struct Entry {
            public var resolvedFrameIndex: Int
            /// `[1, 24, T, H, W]`
            public var videoLatent: MLXArray?
            /// `[1, 32, 2, T]`
            public var audioLatent: MLXArray?

            public init(
                resolvedFrameIndex: Int,
                videoLatent: MLXArray? = nil,
                audioLatent: MLXArray? = nil
            ) {
                self.resolvedFrameIndex = resolvedFrameIndex
                self.videoLatent = videoLatent
                self.audioLatent = audioLatent
            }
        }

        public var entries: [Entry]
        /// Blend factor for the visual conditioning rows: `aug * rows + (1 - aug) * noise`.
        public var visualNoiseAugmentation: Double
        /// Audio conditioning is unaugmented by default (`aug == 1`).
        public var audioNoiseAugmentation: Double
        /// Noise for the visual blend. The reference restarts the same RNG for
        /// every condition, so one buffer is reused across entries; supply it
        /// explicitly to match a reference run exactly.
        public var visualNoise: MLXArray?

        public init(
            entries: [Entry],
            visualNoiseAugmentation: Double = MiniMaxH3PackedLayout.visualCondTimestep,
            audioNoiseAugmentation: Double = MiniMaxH3PackedLayout.audioCondTimestep,
            visualNoise: MLXArray? = nil
        ) {
            self.entries = entries
            self.visualNoiseAugmentation = visualNoiseAugmentation
            self.audioNoiseAugmentation = audioNoiseAugmentation
            self.visualNoise = visualNoise
        }

        var keyframes: [MiniMaxH3Keyframe] {
            entries.map { entry in
                MiniMaxH3Keyframe(
                    resolvedFrameIndex: entry.resolvedFrameIndex,
                    videoLatentFrames: entry.videoLatent.map { $0.shape[2] },
                    audioLatentFrames: entry.audioLatent.map { $0.shape[3] }
                )
            }
        }
    }

    private struct ForwardContext {
        let hidden: MLXArray
        let timeEmbedding: MLXArray
        let segments: [(start: Int, end: Int, row: Int)]
        let rope: (cos: MLXArray, sin: MLXArray)
        let layout: MiniMaxH3PackedLayout
        let timeRow: [Double: Int]
        let tVideo: Double
        let tAudio: Double
    }

    /// Build the packed sequence, AdaLN row table, time embedding and RoPE.
    private func prepare(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        refinedText: MLXArray,
        sigma: Float,
        conditioning: Conditioning? = nil
    ) throws -> ForwardContext {
        let layout = MiniMaxH3PackedLayout(
            textLength: refinedText.shape[0],
            latentFrames: videoLatent.shape[2],
            latentHeight: videoLatent.shape[3],
            latentWidth: videoLatent.shape[4],
            audioFrames: audioLatent.shape[3],
            keyframes: conditioning?.keyframes ?? []
        )

        let tVideo = 1.0 - Double(sigma)
        let tAudio = 1.0 - Self.timeShiftSigma(
            sigma, from: configuration.videoShift, to: configuration.audioShift
        )
        // Conditioning rows are pinned near sigma 1 so the model treats them as
        // clean anchors rather than something to denoise.
        let tCond = max(tVideo, conditioning?.visualNoiseAugmentation
            ?? MiniMaxH3PackedLayout.visualCondTimestep)
        let tCondAudio = max(tAudio, conditioning?.audioNoiseAugmentation
            ?? MiniMaxH3PackedLayout.audioCondTimestep)

        func time(for kind: MiniMaxH3PackedLayout.Kind) -> Double {
            switch kind {
            case .text, .video: tVideo
            case .audio: tAudio
            case .cond: tCond
            case .condAudio: tCondAudio
            }
        }

        var uniqueTimes = Array(Set(layout.segments.map { time(for: $0.kind) }))
            .sorted()
        if uniqueTimes.isEmpty { uniqueTimes = [tVideo] }
        let timeRow = Dictionary(
            uniqueKeysWithValues: uniqueTimes.enumerated().map { ($1, $0) }
        )

        var segments: [(start: Int, end: Int, row: Int)] = []
        for segment in layout.segments {
            let row = timeRow[time(for: segment.kind)]! * configuration.adalnModalities
                + segment.kind.modalityTag
            segments.append((segment.start, segment.end, row))
        }

        // Conditioning rows precede the target rows in segment order, so the
        // embeds are simply concatenated in that order.
        var videoRowGroups: [MLXArray] = []
        var audioRowGroups: [MLXArray] = []
        if let conditioning {
            let augmentation = conditioning.visualNoiseAugmentation
            for entry in conditioning.entries {
                if let latent = entry.videoLatent {
                    var rows = Self.patchifyVideo(
                        latent, patchSize: configuration.patchSize
                    )
                    if augmentation < 1.0 {
                        let noise = conditioning.visualNoise
                            ?? MLXRandom.normal(rows.shape)
                        rows = rows * Float(augmentation)
                            + noise.asType(rows.dtype) * Float(1.0 - augmentation)
                    }
                    videoRowGroups.append(rows)
                }
                if let latent = entry.audioLatent {
                    audioRowGroups.append(Self.packAudio(latent))
                }
            }
        }
        videoRowGroups.append(
            Self.patchifyVideo(videoLatent, patchSize: configuration.patchSize)
        )
        audioRowGroups.append(Self.packAudio(audioLatent))

        let videoRows = videoRowGroups.count == 1
            ? videoRowGroups[0]
            : MLX.concatenated(videoRowGroups, axis: 0)
        let audioRows = audioRowGroups.count == 1
            ? audioRowGroups[0]
            : MLX.concatenated(audioRowGroups, axis: 0)
        let videoEmbed = (try linear(videoRows, "video_patch_proj", bias: true))
            .asType(computeDType)
        let audioEmbed = (try linear(audioRows, "audio_patch_proj", bias: true))
            .asType(computeDType)

        // Assemble in segment order.
        var pieces: [MLXArray] = []
        var videoOffset = 0
        var audioOffset = 0
        for segment in layout.segments {
            switch segment.kind {
            case .text:
                pieces.append(refinedText.asType(computeDType))
            case .cond, .video:
                pieces.append(
                    videoEmbed[videoOffset ..< (videoOffset + segment.count)]
                )
                videoOffset += segment.count
            case .condAudio, .audio:
                pieces.append(
                    audioEmbed[audioOffset ..< (audioOffset + segment.count)]
                )
                audioOffset += segment.count
            }
        }
        let hidden = MLX.concatenated(pieces, axis: 0)

        let timeValues = uniqueTimes.map { Float($0) }
        let timeEmbedding: MLXArray
        if configuration.usesAdalnCurves {
            timeEmbedding = try adalnCurveEmbedding(timeValues)
        } else {
            timeEmbedding = try self.timeEmbedding(MLXArray(timeValues))
        }
        let embeddedTime = timeEmbedding.asType(computeDType)

        let positionIDs = MLXArray(
            layout.positionIDs.map { Float($0) }, [layout.sequenceLength, 3]
        )
        let angles = try ropeAngles(positionIDs: positionIDs)

        return ForwardContext(
            hidden: hidden,
            timeEmbedding: embeddedTime,
            segments: segments,
            rope: (cos: MLX.cos(angles), sin: MLX.sin(angles)),
            layout: layout,
            timeRow: timeRow,
            tVideo: tVideo,
            tAudio: tAudio
        )
    }

    /// Run one denoising step.
    ///
    /// - Parameters:
    ///   - videoLatent: `[1, 24, T, H, W]`
    ///   - audioLatent: `[1, 32, 2, audioFrames]`
    ///   - textStates: `[textLength, 5120]` Qwen3-VL hidden states, or
    ///     `[textLength, 5376]` if already refined.
    ///   - sigma: the video stream's sigma in [0, 1].
    public func forward(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        textStates: MLXArray,
        sigma: Float,
        conditioning: Conditioning? = nil,
        layerCount: Int? = nil
    ) throws -> Output {
        // Qwen states arrive at text_dim and need refining; already-refined
        // states come in at hidden width and pass through.
        let text = textStates.shape[1] == configuration.hiddenSize
            ? textStates.asType(computeDType)
            : try refineTextStates(textStates)

        let context = try prepare(
            videoLatent: videoLatent,
            audioLatent: audioLatent,
            refinedText: text,
            sigma: sigma,
            conditioning: conditioning
        )

        var hidden = context.hidden
        for layer in 0 ..< (layerCount ?? configuration.layerCount) {
            hidden = try ditBlock(
                hidden, layer: layer, timeEmbedding: context.timeEmbedding,
                segments: context.segments, rope: context.rope
            )
        }

        // Final layer has a single AdaLN modality, so the row is the time index
        // rather than `time * 3 + tag`.
        let finalParameters = try adalnParameters(
            context.timeEmbedding, prefix: "final_layer.adaln_proj",
            expand: configuration.finalAdalnExpand,
            modalities: configuration.finalAdalnModalities
        )
        let normed = try rmsNorm(hidden, weightName: "final_layer.norm.weight")

        func head(_ segment: MiniMaxH3PackedLayout.Segment, time: Double) -> MLXArray {
            let row = context.timeRow[time]!
            let slice = normed[segment.start ..< segment.end]
            let scale = finalParameters[1][row ..< (row + 1)].asType(slice.dtype)
            let shift = finalParameters[0][row ..< (row + 1)].asType(slice.dtype)
            return (slice * (1.0 + scale) + shift).asType(.float32)
        }

        let videoSegment = context.layout.segment(.video)!
        let audioSegment = context.layout.segment(.audio)!
        let videoOut = try linear(
            head(videoSegment, time: context.tVideo),
            "final_layer.video_out", bias: true
        )
        let audioOut = try linear(
            head(audioSegment, time: context.tAudio),
            "final_layer.audio_out", bias: true
        )

        // The reference returns the negated velocity for both streams.
        return Output(
            video: -Self.unpatchifyVideo(
                videoOut,
                frames: videoLatent.shape[2],
                height: videoLatent.shape[3] / configuration.patchSize.height,
                width: videoLatent.shape[4] / configuration.patchSize.width,
                channels: configuration.videoLatentChannels,
                patchSize: configuration.patchSize
            ),
            audio: -Self.unpackAudio(audioOut)
        )
    }

    /// Move a sigma from one shift schedule to another.
    public static func timeShiftSigma(_ sigma: Float, from: Float, to: Float) -> Double {
        let s = Double(sigma)
        let base = s / (Double(from) + s * (1.0 - Double(from)))
        return Double(to) * base / (1.0 + (Double(to) - 1.0) * base)
    }

    // MARK: - Patching

    /// `[B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]`
    static func patchifyVideo(
        _ latent: MLXArray,
        patchSize: (frames: Int, height: Int, width: Int)
    ) -> MLXArray {
        let batch = latent.shape[0]
        let channels = latent.shape[1]
        let t = latent.shape[2] / patchSize.frames
        let h = latent.shape[3] / patchSize.height
        let w = latent.shape[4] / patchSize.width
        return latent
            .reshaped(batch, channels, t, patchSize.frames, h, patchSize.height,
                      w, patchSize.width)
            // nctrhpwq -> nthwcrpq
            .transposed(0, 2, 4, 6, 1, 3, 5, 7)
            .reshaped(
                batch * t * h * w,
                channels * patchSize.frames * patchSize.height * patchSize.width
            )
    }

    static func unpatchifyVideo(
        _ rows: MLXArray,
        frames: Int,
        height: Int,
        width: Int,
        channels: Int,
        patchSize: (frames: Int, height: Int, width: Int)
    ) -> MLXArray {
        rows
            .reshaped(-1, frames, height, width, channels, patchSize.frames,
                      patchSize.height, patchSize.width)
            // nthwcrpq -> nctrhpwq
            .transposed(0, 4, 1, 5, 2, 6, 3, 7)
            .reshaped(
                -1, channels, frames * patchSize.frames,
                height * patchSize.height, width * patchSize.width
            )
    }

    /// `[B, 32, 2, T] -> [2*T, 32]`, channel-major.
    static func packAudio(_ latent: MLXArray) -> MLXArray {
        let channels = latent.shape[1]
        let streams = latent.shape[2]
        let frames = latent.shape[3]
        return latent[0].transposed(1, 2, 0).reshaped(streams * frames, channels)
    }

    /// `[2*T, 32] -> [1, 32, 2, T]`
    static func unpackAudio(_ rows: MLXArray, streams: Int = 2) -> MLXArray {
        let frames = rows.shape[0] / streams
        return rows.reshaped(streams, frames, rows.shape[1])
            .transposed(2, 0, 1)
            .expandedDimensions(axis: 0)
    }
}
