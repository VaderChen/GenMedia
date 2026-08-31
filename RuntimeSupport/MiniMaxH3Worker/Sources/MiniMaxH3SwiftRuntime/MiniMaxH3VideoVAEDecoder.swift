import Foundation
import MLX
import MLXNN

/// The MiniMax H3 video VAE decoder: a 36-layer 3-D vision transformer that
/// turns a `[B, 24, T, H, W]` latent into `[B, 3, T*4, H*16, W*16]` pixels.
///
/// Ported from `ViT3DDecoder` in the ComfyUI reference (`vae.py`).
public final class MiniMaxH3VideoVAEDecoder {
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
        try configuration.validate(against: shapes)
        self.configuration = configuration
        self.weights = weights
    }

    /// Load a decoder from a `minimax_h3_video_vae_*.safetensors` file.
    public static func load(
        fileURL: URL,
        configuration: MiniMaxH3VideoVAEConfiguration = .default
    ) throws -> MiniMaxH3VideoVAEDecoder {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw MiniMaxH3WeightError.missingTensor(fileURL.path)
        }
        let loaded = try MLX.loadArrays(url: fileURL)
        // The file holds both halves of the VAE; decoding needs the ViT decoder
        // plus post_quant_conv, which sits between the latent and the decoder.
        let decoderWeights = loaded.filter {
            $0.key.hasPrefix("decoder.") || $0.key.hasPrefix("post_quant_conv.")
        }
        return try MiniMaxH3VideoVAEDecoder(
            configuration: configuration,
            weights: decoderWeights
        )
    }

    private func weight(_ name: String) throws -> MLXArray {
        guard let value = weights[name] else {
            throw MiniMaxH3WeightError.missingTensor(name)
        }
        return value
    }

    // MARK: - Positional encoding

    /// Normalized patch-centre coordinates, `[T*H*W, 3]` in [-1, 1].
    ///
    /// Mirrors `create_token_ids`: each axis uses centres at `arange(0.5, n)/n`
    /// mapped to [-1, 1], then a row-major (`indexing="ij"`) meshgrid.
    /// Patch centres along one axis, mapped to [-1, 1].
    ///
    /// Split out from ``tokenIDs(frames:height:width:)`` so the coordinate
    /// convention can be unit-tested without a Metal device.
    static func patchCentres(_ size: Int) -> [Float] {
        (0 ..< size).map { index in
            2.0 * ((Float(index) + 0.5) / Float(size)) - 1.0
        }
    }

    static func tokenIDs(frames: Int, height: Int, width: Int) -> MLXArray {
        let t = patchCentres(frames)
        let h = patchCentres(height)
        let w = patchCentres(width)
        var flat = [Float]()
        flat.reserveCapacity(frames * height * width * 3)
        for ti in 0 ..< frames {
            for hi in 0 ..< height {
                for wi in 0 ..< width {
                    flat.append(t[ti])
                    flat.append(h[hi])
                    flat.append(w[wi])
                }
            }
        }
        return MLXArray(flat, [frames * height * width, 3])
    }

    /// Cosine/sine tables for the split-half rotary embedding.
    ///
    /// `RotaryEmbeddingND` builds `inv_freq = 1 / theta ** arange(0, 1, 2*n/dim)`
    /// then multiplies by `2*pi*ids`, giving `pairCount` angles per token.
    /// `1 / theta ** arange(0, 1, 2 * axisCount / ropeDim)`.
    ///
    /// Split out for unit testing: the half-open `arange` bound decides how many
    /// frequencies exist, and therefore the rotary pair count.
    static func inverseFrequencies(
        ropeDim: Int,
        theta: Float,
        axisCount: Int = 3
    ) -> [Float] {
        let step = Float(2 * axisCount) / Float(ropeDim)
        var frequencies = [Float]()
        var exponent: Float = 0
        while exponent < 1.0 {
            frequencies.append(pow(theta, -exponent))
            exponent += step
        }
        return frequencies
    }

    func ropeTables(ids: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let axisCount = 3
        let frequencies = Self.inverseFrequencies(
            ropeDim: configuration.ropeDim,
            theta: configuration.ropeTheta,
            axisCount: axisCount
        )
        let inverseFrequency = MLXArray(frequencies, [1, 1, frequencies.count])
        // ids: [S, 3] -> [S, 3, 1] * [1, 1, F] -> [S, 3, F] -> [S, 3*F]
        let sequenceLength = ids.shape[0]
        let angles = (ids.reshaped(sequenceLength, axisCount, 1) * inverseFrequency)
            * (2.0 * Float.pi)
        let flattened = angles.reshaped(sequenceLength, axisCount * frequencies.count)
        return (MLX.cos(flattened), MLX.sin(flattened))
    }

    /// Rotate the leading `ropeDim` channels of each head, split-half style.
    ///
    /// The reference builds an explicit `[c, -s, s, c]` table; the equivalent
    /// closed form is used here. Channels past `ropeDim` pass through.
    func applyRoPE(_ input: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let pairs = configuration.ropePairCount
        let rotated = configuration.ropeDim
        // input: [B, S, heads, headDim]; broadcast tables over batch and heads.
        let cosine = cos.reshaped(1, cos.shape[0], 1, pairs)
        let sine = sin.reshaped(1, sin.shape[0], 1, pairs)
        let first = input[.ellipsis, 0 ..< pairs]
        let second = input[.ellipsis, pairs ..< rotated]
        let outFirst = first * cosine - second * sine
        let outSecond = first * sine + second * cosine
        if rotated == input.shape[input.shape.count - 1] {
            return MLX.concatenated([outFirst, outSecond], axis: -1)
        }
        let passthrough = input[.ellipsis, rotated ..< input.shape[input.shape.count - 1]]
        return MLX.concatenated([outFirst, outSecond, passthrough], axis: -1)
    }

    // MARK: - Primitives

    /// RMS normalization without a learned scale, as used for Q/K.
    private func rmsNorm(_ input: MLXArray) -> MLXArray {
        let meanSquare = MLX.mean(input * input, axis: -1, keepDims: true)
        return input * MLX.rsqrt(meanSquare + configuration.eps)
    }

    /// RMS normalization with a learned scale, as used by `norm1`/`norm2`.
    private func rmsNorm(_ input: MLXArray, weight: MLXArray) -> MLXArray {
        rmsNorm(input) * weight
    }

    private func layerNorm(
        _ input: MLXArray,
        weight: MLXArray,
        bias: MLXArray
    ) -> MLXArray {
        let mean = MLX.mean(input, axis: -1, keepDims: true)
        let centered = input - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        return centered * MLX.rsqrt(variance + configuration.eps) * weight + bias
    }

    private func linear(
        _ input: MLXArray,
        _ prefix: String,
        bias hasBias: Bool = true
    ) throws -> MLXArray {
        let matrix = try weight("\(prefix).weight")
        var output = input.matmul(matrix.transposed())
        if hasBias {
            output = output + (try weight("\(prefix).bias"))
        }
        return output
    }

    // MARK: - Blocks

    private func attention(
        _ input: MLXArray,
        layer: Int,
        cos: MLXArray,
        sin: MLXArray
    ) throws -> MLXArray {
        let prefix = "decoder.transformer_blocks.\(layer).attn"
        let batch = input.shape[0]
        let sequence = input.shape[1]
        let heads = configuration.headCount
        let headDim = configuration.headDim

        let qkv = try linear(input, "\(prefix).to_qkv")
        // The reference views qkv as [B, S, heads, 3*headDim] and *then* chunks
        // along the last axis, so q/k/v are interleaved per head rather than
        // laid out as three contiguous blocks.
        let grouped = qkv.reshaped(batch, sequence, heads, 3 * headDim)
        var query = grouped[.ellipsis, 0 ..< headDim]
        var key = grouped[.ellipsis, headDim ..< (2 * headDim)]
        let value = grouped[.ellipsis, (2 * headDim) ..< (3 * headDim)]

        query = rmsNorm(query)
        key = rmsNorm(key)
        query = applyRoPE(query, cos: cos, sin: sin)
        key = applyRoPE(key, cos: cos, sin: sin)

        let scale = 1.0 / Foundation.sqrt(Float(headDim))
        var output = MLXFast.scaledDotProductAttention(
            queries: query.transposed(0, 2, 1, 3),
            keys: key.transposed(0, 2, 1, 3),
            values: value.transposed(0, 2, 1, 3),
            scale: scale,
            mask: nil
        )
        output = output.transposed(0, 2, 1, 3)
            .reshaped(batch, sequence, heads * headDim)
        // The reference clamps non-finite attention output to zero.
        output = MLX.where(MLX.isNaN(output), MLXArray(Float(0)), output)
        return try linear(output, "\(prefix).to_out")
    }

    private func feedForward(_ input: MLXArray, layer: Int) throws -> MLXArray {
        let prefix = "decoder.transformer_blocks.\(layer).ff"
        let projected = try linear(input, "\(prefix).w1")
        let inner = configuration.hiddenSize * configuration.ffnMultiplier
        let gate = projected[.ellipsis, 0 ..< inner]
        let value = projected[.ellipsis, inner ..< (2 * inner)]
        return try linear(MLXNN.silu(gate) * value, "\(prefix).w2")
    }

    private func transformerBlock(
        _ input: MLXArray,
        layer: Int,
        cos: MLXArray,
        sin: MLXArray
    ) throws -> MLXArray {
        let prefix = "decoder.transformer_blocks.\(layer)"
        var hidden = input
        let normed1 = rmsNorm(hidden, weight: try weight("\(prefix).norm1.weight"))
        hidden = hidden + (try attention(normed1, layer: layer, cos: cos, sin: sin))
            * (try weight("\(prefix).scale1"))
        let normed2 = rmsNorm(hidden, weight: try weight("\(prefix).norm2.weight"))
        hidden = hidden + (try feedForward(normed2, layer: layer))
            * (try weight("\(prefix).scale2"))
        return hidden
    }

    // MARK: - Decode

    /// Decode a latent tensor into pixels.
    ///
    /// - Parameter latent: `[B, latentChannels, T, H, W]`.
    /// - Returns: `[B, 3, T*temporalRatio, H*spatialRatio, W*spatialRatio]`,
    ///   raw decoder output before pixel de-normalization.
    public func decode(latent: MLXArray) throws -> MLXArray {
        guard latent.ndim == 5, latent.shape[1] == configuration.latentChannels else {
            throw MiniMaxH3WeightError.architectureMismatch(
                expected: [-1, configuration.latentChannels, -1, -1, -1],
                actual: latent.shape,
                detail: "video VAE decode input"
            )
        }
        let batch = latent.shape[0]
        let frames = latent.shape[2]
        let height = latent.shape[3]
        let width = latent.shape[4]
        let patchCount = frames * height * width

        // [B, C, T, H, W] -> [B, T*H*W, C]
        var tokens = latent.reshaped(batch, configuration.latentChannels, patchCount)
            .transposed(0, 2, 1)
        // post_quant_conv is a 1x1x1 convolution, i.e. a per-token linear over
        // channels. Its weight is stored 5-D ([out, in, 1, 1, 1]), so collapse
        // the singleton spatial dims before applying it.
        let postQuant = try weight("post_quant_conv.weight")
            .reshaped(configuration.latentChannels, configuration.latentChannels)
        tokens = tokens.matmul(postQuant.transposed())
            + (try weight("post_quant_conv.bias"))
        var hidden = try linear(tokens, "decoder.x_embedder")

        let registers = try weight("decoder.register_tokens")
        let broadcastRegisters = MLX.broadcast(
            registers,
            to: [batch, configuration.registerTokenCount, configuration.hiddenSize]
        )
        let zeroToken = MLX.zeros([batch, 1, configuration.hiddenSize], dtype: hidden.dtype)
        hidden = MLX.concatenated(
            [hidden, broadcastRegisters.asType(hidden.dtype), zeroToken],
            axis: 1
        )

        // Suffix tokens sit at position zero on every axis.
        let patchIDs = Self.tokenIDs(frames: frames, height: height, width: width)
        let suffixIDs = MLX.zeros([configuration.suffixTokenCount, 3], dtype: patchIDs.dtype)
        let ids = MLX.concatenated([patchIDs, suffixIDs], axis: 0)
        let (cos, sin) = ropeTables(ids: ids)

        for layer in 0 ..< configuration.layerCount {
            hidden = try transformerBlock(hidden, layer: layer, cos: cos, sin: sin)
        }

        let normalized = layerNorm(
            hidden,
            weight: try weight("decoder.norm_out.weight"),
            bias: try weight("decoder.norm_out.bias")
        )
        var output = try linear(normalized, "decoder.proj_out")
        output = output[0..., 0 ..< patchCount, 0...]

        // [B, T, H, W, C, pt, ph, pw] -> [B, C, T*pt, H*ph, W*pw]
        let pt = configuration.temporalRatio
        let ps = configuration.spatialRatio
        return output
            .reshaped(batch, frames, height, width, configuration.outputChannels, pt, ps, ps)
            .transposed(0, 4, 1, 5, 2, 6, 3, 7)
            .reshaped(
                batch,
                configuration.outputChannels,
                frames * pt,
                height * ps,
                width * ps
            )
    }

    /// Undo the per-channel latent normalization applied at encode time.
    public static func denormalizeLatent(
        _ latent: MLXArray,
        configuration: MiniMaxH3VideoVAEConfiguration = .default
    ) -> MLXArray {
        let shape = [1, configuration.latentChannels, 1, 1, 1]
        let mean = MLXArray(MiniMaxH3VideoVAEConfiguration.latentsMean, shape)
        let std = MLXArray(MiniMaxH3VideoVAEConfiguration.latentsStd, shape)
        return latent * std + mean
    }

    /// Map raw decoder output to [0, 1] pixels.
    public static func finalizePixels(_ pixels: MLXArray) -> MLXArray {
        let shape = [1, 3, 1, 1, 1]
        let mean = MLXArray(MiniMaxH3VideoVAEConfiguration.pixelMean, shape)
        let std = MLXArray(MiniMaxH3VideoVAEConfiguration.pixelStd, shape)
        return MLX.clip(pixels * std + mean, min: 0.0, max: 1.0)
    }
}
