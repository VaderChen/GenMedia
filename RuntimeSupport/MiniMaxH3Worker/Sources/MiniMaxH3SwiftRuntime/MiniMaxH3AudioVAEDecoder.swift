import Foundation
import MLX
import MLXNN

/// Geometry for the MiniMax H3 audio VAE decode path.
///
/// Mirrors `MiniMaxH3AudioVAE.__init__` and `BigVGAN.__init__` in the ComfyUI
/// reference (`audio_vae.py`). Only the decode side is modelled: the encoder,
/// `pre_block`, `mean_proj` and `logs_proj` are encode-only and are not needed
/// to turn latents into audio.
public struct MiniMaxH3AudioVAEConfiguration: Sendable, Equatable {
    public var latentChannels: Int
    public var latentDim: Int
    public var upsampleInitialChannel: Int
    public var upsampleRates: [Int]
    public var upsampleKernelSizes: [Int]
    public var resblockKernelSizes: [Int]
    public var resblockDilations: [[Int]]
    public var sampleRate: Int
    /// Anti-aliasing resampler kernel width, shared by up and down.
    public var filterKernelSize: Int

    /// Samples produced per latent frame: `prod(upsampleRates)` = 800.
    public var hopLength: Int { upsampleRates.reduce(1, *) }
    public var resblockCount: Int { upsampleRates.count * resblockKernelSizes.count }
    /// Channel width after upsample stage `index`.
    public func channels(afterStage index: Int) -> Int {
        upsampleInitialChannel / (1 << (index + 1))
    }
    public var finalChannels: Int { channels(afterStage: upsampleRates.count - 1) }

    public static let `default` = MiniMaxH3AudioVAEConfiguration(
        latentChannels: 32,
        latentDim: 2048,
        upsampleInitialChannel: 1024,
        upsampleRates: [5, 5, 2, 2, 2, 2, 2],
        upsampleKernelSizes: [9, 9, 4, 4, 4, 4, 4],
        resblockKernelSizes: [3, 7, 11],
        resblockDilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        sampleRate: 32000,
        filterKernelSize: 12
    )

    public init(
        latentChannels: Int,
        latentDim: Int,
        upsampleInitialChannel: Int,
        upsampleRates: [Int],
        upsampleKernelSizes: [Int],
        resblockKernelSizes: [Int],
        resblockDilations: [[Int]],
        sampleRate: Int,
        filterKernelSize: Int
    ) {
        self.latentChannels = latentChannels
        self.latentDim = latentDim
        self.upsampleInitialChannel = upsampleInitialChannel
        self.upsampleRates = upsampleRates
        self.upsampleKernelSizes = upsampleKernelSizes
        self.resblockKernelSizes = resblockKernelSizes
        self.resblockDilations = resblockDilations
        self.sampleRate = sampleRate
        self.filterKernelSize = filterKernelSize
    }

    /// Every tensor the decode path requires, with its PyTorch-order shape.
    public var expectedDecoderShapes: [String: [Int]] {
        var expected: [String: [Int]] = [
            "latents_mean": [latentChannels],
            "latents_std": [latentChannels],
            "dec_in_proj.weight": [latentDim, latentChannels, 1],
            "dec_in_proj.bias": [latentDim],
            "decoder.conv_pre.weight": [upsampleInitialChannel, latentDim, 7],
            "decoder.conv_pre.bias": [upsampleInitialChannel],
            // conv_post has use_bias_at_final=False
            "decoder.conv_post.weight": [1, finalChannels, 7],
            "decoder.activation_post.act.alpha": [finalChannels],
            "decoder.activation_post.act.beta": [finalChannels],
            "decoder.activation_post.upsample.filter": [1, 1, filterKernelSize],
            "decoder.activation_post.downsample.lowpass.filter": [1, 1, filterKernelSize]
        ]

        for stage in upsampleRates.indices {
            let inChannels = upsampleInitialChannel / (1 << stage)
            let outChannels = channels(afterStage: stage)
            expected["decoder.ups.\(stage).0.weight"] =
                [inChannels, outChannels, upsampleKernelSizes[stage]]
            expected["decoder.ups.\(stage).0.bias"] = [outChannels]
        }

        for stage in upsampleRates.indices {
            let channelCount = channels(afterStage: stage)
            for (kernelIndex, kernel) in resblockKernelSizes.enumerated() {
                let block = stage * resblockKernelSizes.count + kernelIndex
                let prefix = "decoder.resblocks.\(block)"
                for conv in 0 ..< resblockDilations[kernelIndex].count {
                    expected["\(prefix).convs1.\(conv).weight"] =
                        [channelCount, channelCount, kernel]
                    expected["\(prefix).convs1.\(conv).bias"] = [channelCount]
                    expected["\(prefix).convs2.\(conv).weight"] =
                        [channelCount, channelCount, kernel]
                    expected["\(prefix).convs2.\(conv).bias"] = [channelCount]
                }
                for activation in 0 ..< (resblockDilations[kernelIndex].count * 2) {
                    let actPrefix = "\(prefix).activations.\(activation)"
                    expected["\(actPrefix).act.alpha"] = [channelCount]
                    expected["\(actPrefix).act.beta"] = [channelCount]
                    expected["\(actPrefix).upsample.filter"] = [1, 1, filterKernelSize]
                    expected["\(actPrefix).downsample.lowpass.filter"] =
                        [1, 1, filterKernelSize]
                }
            }
        }
        return expected
    }

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

/// BigVGAN decoder for the MiniMax H3 audio VAE.
///
/// Turns normalized latents `[B, 32, S, T]` into waveforms `[B, S, T*800]` at
/// 32 kHz. All internal tensors are channel-last (`[N, L, C]`) to match MLX's
/// convolution layout; the stored PyTorch weights are `[C_out, C_in, K]` and are
/// transposed on use.
public final class MiniMaxH3AudioVAEDecoder {
    public let configuration: MiniMaxH3AudioVAEConfiguration
    private let weights: [String: MLXArray]

    public init(
        configuration: MiniMaxH3AudioVAEConfiguration = .default,
        weights: [String: MLXArray]
    ) throws {
        var shapes: [String: [Int]] = [:]
        for (name, value) in weights {
            shapes[name] = value.shape
        }
        try configuration.validate(against: shapes)
        self.configuration = configuration
        self.weights = weights
    }

    public static func load(
        fileURL: URL,
        configuration: MiniMaxH3AudioVAEConfiguration = .default
    ) throws -> MiniMaxH3AudioVAEDecoder {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw MiniMaxH3WeightError.missingTensor(fileURL.path)
        }
        let loaded = try MLX.loadArrays(url: fileURL)
        return try MiniMaxH3AudioVAEDecoder(
            configuration: configuration,
            weights: loaded
        )
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let value = weights[name] else {
            throw MiniMaxH3WeightError.missingTensor(name)
        }
        return value
    }

    // MARK: - Primitives

    /// `x + sin^2(alpha * x) / beta`, with alpha and beta stored in log scale.
    private func snakeBeta(
        _ input: MLXArray,
        alpha: MLXArray,
        beta: MLXArray
    ) -> MLXArray {
        let a = MLX.exp(alpha)
        let b = MLX.exp(beta)
        let sine = MLX.sin(input * a)
        return input + (sine * sine) / (b + 1e-9)
    }

    /// PyTorch `[C_out, C_in, K]` -> MLX `[C_out, K, C_in]`.
    private func convWeight(_ stored: MLXArray) -> MLXArray {
        stored.transposed(0, 2, 1)
    }

    /// Channel-last 1-D convolution over `[N, L, C]`.
    private func conv1d(
        _ input: MLXArray,
        _ prefix: String,
        padding: Int,
        dilation: Int = 1,
        bias hasBias: Bool = true
    ) throws -> MLXArray {
        let kernel = convWeight(try weight("\(prefix).weight"))
        var output = MLX.conv1d(
            input, kernel, stride: 1, padding: padding, dilation: dilation, groups: 1
        )
        if hasBias {
            output = output + (try weight("\(prefix).bias"))
        }
        return output
    }

    /// Depthwise resampling filter shared by every channel, `[C, K, 1]`.
    private func depthwiseFilter(_ stored: MLXArray, channels: Int) -> MLXArray {
        // stored: [1, 1, K] -> [1, K, 1] -> broadcast over channels
        let reshaped = stored.transposed(0, 2, 1)
        return MLX.broadcast(reshaped, to: [channels, stored.shape[2], 1])
    }

    /// 2x anti-aliased upsample: replicate-pad, transposed conv with the Kaiser
    /// sinc filter, scale by the ratio, then crop.
    private func upsample(
        _ input: MLXArray,
        filter stored: MLXArray,
        ratio: Int = 2
    ) -> MLXArray {
        let kernelSize = configuration.filterKernelSize
        let channels = input.shape[2]
        let pad = kernelSize / ratio - 1
        let padLeft = pad * ratio + (kernelSize - ratio) / 2
        let padRight = pad * ratio + (kernelSize - ratio + 1) / 2

        let padded = MLX.padded(
            input, widths: [.init((0, 0)), .init((pad, pad)), .init((0, 0))], mode: .edge
        )
        var output = MLX.convTransposed1d(
            padded,
            depthwiseFilter(stored, channels: channels),
            stride: ratio,
            padding: 0,
            groups: channels
        )
        output = output * Float(ratio)
        let length = output.shape[1]
        return output[0..., padLeft ..< (length - padRight), 0...]
    }

    /// 2x anti-aliased downsample: replicate-pad, strided depthwise low-pass.
    private func downsample(
        _ input: MLXArray,
        filter stored: MLXArray,
        ratio: Int = 2
    ) -> MLXArray {
        let kernelSize = configuration.filterKernelSize
        let channels = input.shape[2]
        let padLeft = kernelSize / 2 - (kernelSize % 2 == 0 ? 1 : 0)
        let padRight = kernelSize / 2
        let padded = MLX.padded(
            input,
            widths: [.init((0, 0)), .init((padLeft, padRight)), .init((0, 0))],
            mode: .edge
        )
        return MLX.conv1d(
            padded,
            depthwiseFilter(stored, channels: channels),
            stride: ratio,
            padding: 0,
            groups: channels
        )
    }

    /// `Activation1d`: upsample x2 -> SnakeBeta -> downsample x2.
    private func antiAliasedActivation(
        _ input: MLXArray,
        prefix: String
    ) throws -> MLXArray {
        var hidden = upsample(input, filter: try weight("\(prefix).upsample.filter"))
        hidden = snakeBeta(
            hidden,
            alpha: try weight("\(prefix).act.alpha"),
            beta: try weight("\(prefix).act.beta")
        )
        return downsample(
            hidden,
            filter: try weight("\(prefix).downsample.lowpass.filter")
        )
    }

    // MARK: - Blocks

    static func padding(kernel: Int, dilation: Int) -> Int {
        (kernel * dilation - dilation) / 2
    }

    private func ampBlock(
        _ input: MLXArray,
        block: Int,
        kernel: Int,
        dilations: [Int]
    ) throws -> MLXArray {
        let prefix = "decoder.resblocks.\(block)"
        var hidden = input
        for (index, dilation) in dilations.enumerated() {
            // activations alternate: even indices before convs1, odd before convs2
            var residual = try antiAliasedActivation(
                hidden, prefix: "\(prefix).activations.\(index * 2)"
            )
            residual = try conv1d(
                residual,
                "\(prefix).convs1.\(index)",
                padding: Self.padding(kernel: kernel, dilation: dilation),
                dilation: dilation
            )
            residual = try antiAliasedActivation(
                residual, prefix: "\(prefix).activations.\(index * 2 + 1)"
            )
            residual = try conv1d(
                residual,
                "\(prefix).convs2.\(index)",
                padding: Self.padding(kernel: kernel, dilation: 1)
            )
            hidden = hidden + residual
        }
        return hidden
    }

    // MARK: - Decode

    /// Run the BigVGAN stack over channel-last input `[N, L, latentDim]`.
    private func bigVGAN(_ input: MLXArray) throws -> MLXArray {
        var hidden = try conv1d(input, "decoder.conv_pre", padding: 3)

        for stage in configuration.upsampleRates.indices {
            let kernel = configuration.upsampleKernelSizes[stage]
            let rate = configuration.upsampleRates[stage]
            let stored = try weight("decoder.ups.\(stage).0.weight")
            // PyTorch ConvTranspose1d weight is [C_in, C_out, K];
            // MLX expects [C_out, K, C_in].
            let kernelWeight = stored.transposed(1, 2, 0)
            hidden = MLX.convTransposed1d(
                hidden,
                kernelWeight,
                stride: rate,
                padding: (kernel - rate) / 2
            )
            hidden = hidden + (try weight("decoder.ups.\(stage).0.bias"))

            var accumulated: MLXArray?
            for (kernelIndex, resKernel) in configuration.resblockKernelSizes.enumerated() {
                let block = stage * configuration.resblockKernelSizes.count + kernelIndex
                let out = try ampBlock(
                    hidden,
                    block: block,
                    kernel: resKernel,
                    dilations: configuration.resblockDilations[kernelIndex]
                )
                accumulated = accumulated.map { $0 + out } ?? out
            }
            guard let summed = accumulated else {
                throw MiniMaxH3WeightError.missingTensor("decoder.resblocks")
            }
            hidden = summed / Float(configuration.resblockKernelSizes.count)
        }

        hidden = try antiAliasedActivation(hidden, prefix: "decoder.activation_post")
        hidden = try conv1d(hidden, "decoder.conv_post", padding: 3, bias: false)
        return MLX.clip(hidden, min: -1.0, max: 1.0)
    }

    /// Decode normalized latents into waveforms.
    ///
    /// - Parameter latent: `[B, 32, S, T]`, where `S` is the channel count
    ///   (2 for stereo).
    /// - Returns: `[B, S, T * hopLength]` in [-1, 1] at 32 kHz.
    public func decode(latent: MLXArray) throws -> MLXArray {
        guard latent.ndim == 4, latent.shape[1] == configuration.latentChannels else {
            throw MiniMaxH3WeightError.architectureMismatch(
                expected: [-1, configuration.latentChannels, -1, -1],
                actual: latent.shape,
                detail: "audio VAE decode input"
            )
        }
        let batch = latent.shape[0]
        let streams = latent.shape[2]
        let frames = latent.shape[3]

        // [B, C, S, T] -> [B*S, C, T] -> channel-last [B*S, T, C]
        var hidden = latent.transposed(0, 2, 1, 3)
            .reshaped(batch * streams, configuration.latentChannels, frames)
            .transposed(0, 2, 1)

        let mean = try weight("latents_mean")
        let std = try weight("latents_std")
        hidden = hidden * std + mean

        hidden = try conv1d(hidden, "dec_in_proj", padding: 0)
        let waveform = try bigVGAN(hidden)
        // [B*S, L, 1] -> [B, S, L]
        return waveform.reshaped(batch, streams, waveform.shape[1])
    }
}
