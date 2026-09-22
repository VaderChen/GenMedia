// Single-frame path of AutoencoderKLQwenImage21, using MLX OHWI checkpoint kernels.
import Foundation
import MLX
import MLXFast
import MLXNN

final class Qwen21VAE {
    let config: Qwen21VAEConfiguration
    let weights: Qwen21Weights

    init(directory: URL) throws {
        config = try readJSON(Qwen21VAEConfiguration.self, at: directory.appendingPathComponent("config.json"))
        guard config.z_dim == 64, config.dim_mult == [1, 2, 4, 8, 8],
              config.latents_mean.count == 64, config.latents_std.count == 64,
              config.latents_std.allSatisfy({ $0 > 0 }) else {
            throw Qwen21Error.invalid("不相容的 RGBA VAE 設定。")
        }
        weights = try Qwen21Weights(directory: directory)
    }

    private func conv(_ x: MLXArray, _ name: String, stride: Int = 1, padding: Int? = nil) throws -> MLXArray {
        let w = try weights.tensor(name + ".weight")
        guard w.ndim == 4, w.dim(3) == x.dim(3) else {
            throw Qwen21Error.invalid("VAE 卷積權重須為 OHWI：\(name)")
        }
        let pad = padding ?? (w.dim(1) / 2)
        return try conv2d(x, w, stride: .init(stride), padding: .init(pad)) + weights.tensor(name + ".bias")
    }

    private func norm(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let f = x.asType(.float32)
        let denominator = maximum(sqrt(sum(f * f, axis: -1, keepDims: true)), 1e-12)
        let gamma = try weights.tensor(name + ".gamma").reshaped([1, 1, 1, x.dim(-1)])
        return (f / denominator * sqrt(Float(x.dim(-1))) * gamma).asType(x.dtype)
    }

    private func resnet(_ input: MLXArray, _ name: String) throws -> MLXArray {
        let shortcut = weights.tensors[name + ".conv_shortcut.weight"] == nil
            ? input : try conv(input, name + ".conv_shortcut")
        var x = try conv(silu(norm(input, name + ".norm1")), name + ".conv1")
        x = try conv(silu(norm(x, name + ".norm2")), name + ".conv2") + shortcut
        eval(x)
        return x
    }

    private func mid(_ input: MLXArray, _ name: String) throws -> MLXArray {
        var x = try resnet(input, name + ".resnets.0")
        let p = name + ".attentions.0", h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let qkv = try conv(norm(x, p + ".norm"), p + ".to_qkv").reshaped([1, h * w, 3, c])
        let q = qkv[0..., 0..., 0, 0...].expandedDimensions(axis: 1)
        let k = qkv[0..., 0..., 1, 0...].expandedDimensions(axis: 1)
        let v = qkv[0..., 0..., 2, 0...].expandedDimensions(axis: 1)
        let attention = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
            scale: 1 / sqrt(Float(c)), mask: .none).reshaped([1, h, w, c])
        x = x + (try conv(attention, p + ".proj"))
        return try resnet(x, name + ".resnets.1")
    }

    /// RGBA in NHWC, scaled to [-1,1]; deterministic posterior mean (no sampling).
    func encode(_ rgba: MLXArray) throws -> MLXArray {
        var x = try conv(rgba.asType(.float32), "encoder.conv_in")
        for block in 0..<5 {
            try Task.checkCancellation()
            let original = x, p = "encoder.down_blocks.\(block)"
            for index in 0..<config.num_res_blocks { x = try resnet(x, p + ".resnets.\(index)") }
            let factor = block < 4 ? 2 : 1, temporal = (1...3).contains(block) ? 2 : 1
            if block < 4 {
                x = padded(x, widths: [.init(0), .init((0, 1)), .init((0, 1)), .init(0)])
                x = try conv(x, p + ".downsampler.resample.1", stride: 2, padding: 0)
            }
            x = x + Self.averageShortcut(original, outChannels: x.dim(-1), spatial: factor, temporal: temporal)
            eval(x)
        }
        x = try mid(x, "encoder.mid_block")
        x = try conv(silu(norm(x, "encoder.norm_out")), "encoder.conv_out")
        x = try conv(x, "quant_conv")[0..., 0..., 0..., ..<64]
        x = (x - MLXArray(config.latents_mean)) / MLXArray(config.latents_std)
        eval(x)
        return x.reshaped([1, x.dim(1) * x.dim(2), 64]).asType(.bfloat16)
    }

    func decode(_ latents: MLXArray, height: Int, width: Int) throws -> MLXArray {
        var x = latents.reshaped([1, height / 16, width / 16, 64]).asType(.float32)
        x = x * MLXArray(config.latents_std) + MLXArray(config.latents_mean)
        x = try conv(x, "post_quant_conv")
        x = try mid(conv(x, "decoder.conv_in"), "decoder.mid_block")
        for block in 0..<5 {
            try Task.checkCancellation()
            let original = x, p = "decoder.up_blocks.\(block)"
            for index in 0...config.num_res_blocks { x = try resnet(x, p + ".resnets.\(index)") }
            if block < 4 {
                x = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
                x = try conv(x, p + ".upsampler.resample.1")
                x = x + Self.duplicateShortcut(original, outChannels: x.dim(-1), temporal: block < 3 ? 2 : 1)
            }
            eval(x)
        }
        x = try conv(silu(norm(x, "decoder.norm_out")), "decoder.conv_out")
        // Preserve alpha; unlike the legacy RGB pipeline this VAE is natively RGBA.
        x = clip((x + 1) / 2, min: 0, max: 1)
        eval(x)
        return x
    }

    static func averageShortcut(_ input: MLXArray, outChannels: Int, spatial s: Int, temporal t: Int) -> MLXArray {
        let h = input.dim(1), w = input.dim(2), c = input.dim(3)
        var x = input.transposed(0, 3, 1, 2).expandedDimensions(axis: 2)
        if t == 2 { x = padded(x, widths: [.init(0), .init(0), .init((1, 0)), .init(0), .init(0)]) }
        x = x.reshaped([1, c, 1, t, h / s, s, w / s, s]).transposed(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshaped([1, outChannels, c * t * s * s / outChannels, h / s, w / s])
        return mean(x, axis: 2).transposed(0, 2, 3, 1)
    }

    static func duplicateShortcut(_ input: MLXArray, outChannels: Int, temporal t: Int) -> MLXArray {
        let h = input.dim(1), w = input.dim(2), c = input.dim(3)
        var x = repeated(input.transposed(0, 3, 1, 2), count: outChannels * t * 4 / c, axis: 1)
        x = x.reshaped([1, outChannels, t, 2, 2, 1, h, w]).transposed(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshaped([1, outChannels, t, h * 2, w * 2])[0..., 0..., t - 1, 0..., 0...]
        return x.transposed(0, 2, 3, 1)
    }
}
