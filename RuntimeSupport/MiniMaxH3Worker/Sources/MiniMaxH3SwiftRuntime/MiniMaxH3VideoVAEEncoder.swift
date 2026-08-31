import Foundation
import MLX

/// Encodes a single keyframe image into an H3 video latent.
///
/// Ported from `EncoderFCN3D` plus `quant_conv` in the ComfyUI reference
/// (`vae.py`). FL2VA anchors are single images, and for one frame the causal
/// temporal convolution degenerates: the front padding is all zeros, so only
/// the last temporal tap of each kernel contributes. That reduces the whole
/// encoder to 2-D convolutions, which is what this implements.
///
/// Multi-frame clips (`x.shape[2] > 1`) take the reference's `encode_temporal`
/// path and are **not** supported here.
public final class MiniMaxH3VideoVAEEncoder {
    /// Channel width per level, `ch * ch_mult`.
    public static let blockChannels = [128, 256, 256, 512, 512, 1024]
    /// Input width per level: the previous level's output.
    public static let blockInputChannels = [128, 128, 256, 256, 512, 512]
    public static let spaceDown = [2, 2, 2, 2, 1, 1]
    public static let timeDown = [1, 2, 2, 1, 1, 1]
    public static let resBlocksPerLevel = 2
    public static let groupCount = 32
    public static let normEps: Float = 1e-6

    public let configuration: MiniMaxH3VideoVAEConfiguration
    private let weights: [String: MLXArray]

    public init(
        configuration: MiniMaxH3VideoVAEConfiguration = .default,
        weights: [String: MLXArray]
    ) throws {
        var shapes: [String: [Int]] = [:]
        for (name, value) in weights {
            shapes[name] = value.shape
        }
        try Self.validate(shapes: shapes, configuration: configuration)
        self.configuration = configuration
        self.weights = weights
    }

    public static func load(
        fileURL: URL,
        configuration: MiniMaxH3VideoVAEConfiguration = .default
    ) throws -> MiniMaxH3VideoVAEEncoder {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw MiniMaxH3WeightError.missingTensor(fileURL.path)
        }
        let loaded = try MLX.loadArrays(url: fileURL)
        let encoderWeights = loaded.filter {
            $0.key.hasPrefix("encoder.") || $0.key.hasPrefix("quant_conv.")
        }
        return try MiniMaxH3VideoVAEEncoder(
            configuration: configuration, weights: encoderWeights
        )
    }

    /// Every tensor the encode path requires, with its PyTorch-order shape.
    public static func expectedShapes(
        _ configuration: MiniMaxH3VideoVAEConfiguration = .default
    ) -> [String: [Int]] {
        let moments = configuration.latentChannels * 2
        var expected: [String: [Int]] = [
            "encoder.conv_in.weight": [blockChannels[0], 3, 3, 3, 3],
            "encoder.conv_in.bias": [blockChannels[0]],
            "encoder.norm_out.weight": [blockChannels.last!],
            "encoder.norm_out.bias": [blockChannels.last!],
            "encoder.conv_out.weight": [moments, blockChannels.last!, 3, 3, 3],
            "encoder.conv_out.bias": [moments],
            "quant_conv.weight": [moments, moments, 1, 1, 1],
            "quant_conv.bias": [moments]
        ]
        for level in blockChannels.indices {
            let out = blockChannels[level]
            for block in 0 ..< resBlocksPerLevel {
                let prefix = "encoder.down.\(level).block.\(block)"
                let input = block == 0 ? blockInputChannels[level] : out
                expected["\(prefix).norm1.weight"] = [input]
                expected["\(prefix).norm1.bias"] = [input]
                expected["\(prefix).norm2.weight"] = [out]
                expected["\(prefix).norm2.bias"] = [out]
                expected["\(prefix).conv1.weight"] = [out, input, 3, 3, 3]
                expected["\(prefix).conv1.bias"] = [out]
                expected["\(prefix).conv2.weight"] = [out, out, 3, 3, 3]
                expected["\(prefix).conv2.bias"] = [out]
                if input != out {
                    expected["\(prefix).nin_shortcut.weight"] = [out, input, 1, 1, 1]
                    expected["\(prefix).nin_shortcut.bias"] = [out]
                }
            }
            if spaceDown[level] * timeDown[level] > 1 {
                let prefix = "encoder.down.\(level).downsample.conv"
                expected["\(prefix).weight"] = [out, out, 3, 3, 3]
                expected["\(prefix).bias"] = [out]
            }
        }
        return expected
    }

    static func validate(
        shapes: [String: [Int]],
        configuration: MiniMaxH3VideoVAEConfiguration
    ) throws {
        for (name, expected) in expectedShapes(configuration).sorted(by: { $0.key < $1.key }) {
            guard let actual = shapes[name] else {
                throw MiniMaxH3WeightError.missingTensor(name)
            }
            guard actual == expected else {
                throw MiniMaxH3WeightError.unexpectedShape(
                    name, expected: expected, actual: actual
                )
            }
        }
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let value = weights[name] else {
            throw MiniMaxH3WeightError.missingTensor(name)
        }
        return value
    }

    // MARK: - Primitives

    /// Reflect padding on H and W of a `[N, H, W, C]` tensor.
    ///
    /// MLX only offers constant and edge padding; reflect excludes the edge
    /// value itself, so it is built from slices.
    /// `F.pad` applies reflect padding one dimension at a time starting from
    /// the last, so width is padded before height and the corners are derived
    /// from the already-width-padded rows. Doing height first gives different
    /// corner values.
    static func reflectPad(_ input: MLXArray, height: Int, width: Int) -> MLXArray {
        var output = input
        if width > 0 {
            let columns = output.shape[2]
            let leftPad = MLX.concatenated(
                (1 ... width).reversed().map { output[0..., 0..., $0 ..< ($0 + 1)] },
                axis: 2
            )
            let rightPad = MLX.concatenated(
                (1 ... width).map { output[0..., 0..., (columns - 1 - $0) ..< (columns - $0)] },
                axis: 2
            )
            output = MLX.concatenated([leftPad, output, rightPad], axis: 2)
        }
        if height > 0 {
            let rows = output.shape[1]
            let topPad = MLX.concatenated(
                (1 ... height).reversed().map { output[0..., $0 ..< ($0 + 1)] }, axis: 1
            )
            let bottomPad = MLX.concatenated(
                (1 ... height).map { output[0..., (rows - 1 - $0) ..< (rows - $0)] },
                axis: 1
            )
            output = MLX.concatenated([topPad, output, bottomPad], axis: 1)
        }
        return output
    }

    /// Constant (zero) padding on the right/bottom, used by `Downsample3D`.
    private static func padBottomRight(_ input: MLXArray) -> MLXArray {
        // `F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")`: width first, then
        // height over the width-padded tensor.
        let columns = input.shape[2]
        let right = input[0..., 0..., (columns - 2) ..< (columns - 1)]
        var output = MLX.concatenated([input, right], axis: 2)
        let rows = output.shape[1]
        let bottom = output[0..., (rows - 2) ..< (rows - 1), 0...]
        output = MLX.concatenated([output, bottom], axis: 1)
        return output
    }

    /// A 3-D conv reduced to its last temporal tap, applied channel-last.
    ///
    /// PyTorch stores `[C_out, C_in, KT, KH, KW]`; MLX wants
    /// `[C_out, KH, KW, C_in]`.
    private func spatialKernel(_ stored: MLXArray) -> MLXArray {
        let outChannels = stored.shape[0]
        let inChannels = stored.shape[1]
        let kernelH = stored.shape[3]
        let kernelW = stored.shape[4]
        let lastTap = stored[0..., 0..., (stored.shape[2] - 1)...]
            .reshaped(outChannels, inChannels, kernelH, kernelW)
        return lastTap.transposed(0, 2, 3, 1)
    }

    /// `CausalConv3d` for a single frame: reflect spatial padding, then a
    /// convolution using only the final temporal tap.
    private func causalConv(
        _ input: MLXArray,
        _ prefix: String,
        padding: Int,
        stride: Int = 1
    ) throws -> MLXArray {
        let kernel = spatialKernel(try weight("\(prefix).weight"))
        let padded = padding > 0
            ? Self.reflectPad(input, height: padding, width: padding)
            : input
        return MLX.conv2d(padded, kernel, stride: .init(stride), padding: .init(0))
            + (try weight("\(prefix).bias"))
    }

    /// GroupNorm over a `[N, H, W, C]` tensor with 32 groups.
    private func groupNorm(_ input: MLXArray, _ prefix: String) throws -> MLXArray {
        let batch = input.shape[0]
        let height = input.shape[1]
        let width = input.shape[2]
        let channels = input.shape[3]
        let perGroup = channels / Self.groupCount
        // Statistics cover all spatial positions within a channel group.
        let grouped = input.reshaped(batch, height * width, Self.groupCount, perGroup)
        let mean = MLX.mean(grouped, axes: [1, 3], keepDims: true)
        let centered = grouped - mean
        let variance = MLX.mean(centered * centered, axes: [1, 3], keepDims: true)
        let normalized = (centered * MLX.rsqrt(variance + Self.normEps))
            .reshaped(batch, height, width, channels)
        return normalized * (try weight("\(prefix).weight"))
            + (try weight("\(prefix).bias"))
    }

    private func silu(_ input: MLXArray) -> MLXArray {
        input * MLX.sigmoid(input)
    }

    // MARK: - Blocks

    private func resnetBlock(
        _ input: MLXArray,
        prefix: String,
        hasShortcut: Bool
    ) throws -> MLXArray {
        var hidden = try causalConv(
            silu(try groupNorm(input, "\(prefix).norm1")), "\(prefix).conv1", padding: 1
        )
        hidden = try causalConv(
            silu(try groupNorm(hidden, "\(prefix).norm2")), "\(prefix).conv2", padding: 1
        )
        let residual = hasShortcut
            ? try causalConv(input, "\(prefix).nin_shortcut", padding: 0)
            : input
        return hidden + residual
    }

    // MARK: - Encode

    /// Encode one frame into a normalized latent.
    ///
    /// - Parameter pixels: `[1, 3, 1, H, W]` in [-1, 1].
    /// - Returns: `[1, 24, 1, H/16, W/16]` normalized latents, ready to use as
    ///   a keyframe anchor.
    public func encode(pixels: MLXArray) throws -> MLXArray {
        guard pixels.ndim == 5, pixels.shape[1] == 3, pixels.shape[2] == 1 else {
            throw MiniMaxH3WeightError.architectureMismatch(
                expected: [1, 3, 1, -1, -1],
                actual: pixels.shape,
                detail: "video VAE encode input (single frame only)"
            )
        }

        // _normalize_pixels: [-1, 1] -> [0, 1] -> ImageNet-normalized.
        let mean = MLXArray(MiniMaxH3VideoVAEConfiguration.pixelMean, [1, 1, 1, 3])
        let std = MLXArray(MiniMaxH3VideoVAEConfiguration.pixelStd, [1, 1, 1, 3])
        // [1, 3, 1, H, W] -> [1, H, W, 3]
        var hidden = pixels[0..., 0..., 0].transposed(0, 2, 3, 1)
        hidden = ((hidden + 1.0) * 0.5 - mean) / std

        hidden = try causalConv(hidden, "encoder.conv_in", padding: 1)

        for level in Self.blockChannels.indices {
            for block in 0 ..< Self.resBlocksPerLevel {
                let prefix = "encoder.down.\(level).block.\(block)"
                let input = block == 0
                    ? Self.blockInputChannels[level] : Self.blockChannels[level]
                hidden = try resnetBlock(
                    hidden, prefix: prefix,
                    hasShortcut: input != Self.blockChannels[level]
                )
            }
            guard Self.spaceDown[level] * Self.timeDown[level] > 1 else { continue }
            // Downsample3D reflect-pads the far edge before the strided conv.
            if Self.spaceDown[level] == 2 {
                hidden = Self.padBottomRight(hidden)
            }
            hidden = try causalConv(
                hidden, "encoder.down.\(level).downsample.conv",
                padding: 0, stride: Self.spaceDown[level]
            )
        }

        hidden = silu(try groupNorm(hidden, "encoder.norm_out"))
        hidden = try causalConv(hidden, "encoder.conv_out", padding: 1)
        hidden = try causalConv(hidden, "quant_conv", padding: 0)

        // The posterior mean is the first half of the moments.
        let latentChannels = configuration.latentChannels
        let posteriorMean = hidden[.ellipsis, 0 ..< latentChannels]

        // [1, H, W, C] -> [1, C, 1, H, W]
        let latent = posteriorMean.transposed(0, 3, 1, 2)
            .expandedDimensions(axis: 2)
        let latentsMean = MLXArray(
            MiniMaxH3VideoVAEConfiguration.latentsMean, [1, latentChannels, 1, 1, 1]
        )
        let latentsStd = MLXArray(
            MiniMaxH3VideoVAEConfiguration.latentsStd, [1, latentChannels, 1, 1, 1]
        )
        return (latent - latentsMean) / latentsStd
    }
}
