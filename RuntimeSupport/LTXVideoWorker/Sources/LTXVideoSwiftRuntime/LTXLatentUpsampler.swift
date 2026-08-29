import Foundation
import MLX
import MLXNN

public struct LTXLatentUpsamplerConfiguration: Sendable, Equatable {
    public let inChannels: Int
    public let midChannels: Int
    public let numBlocksPerStage: Int
    public let dims: Int
    public let spatialUpsample: Bool
    public let temporalUpsample: Bool
    public let spatialScale: Double
    public let rationalResampler: Bool

    public init(
        inChannels: Int = 128,
        midChannels: Int = 512,
        numBlocksPerStage: Int = 4,
        dims: Int = 3,
        spatialUpsample: Bool = true,
        temporalUpsample: Bool = false,
        spatialScale: Double = 2.0,
        rationalResampler: Bool = false
    ) throws {
        guard inChannels > 0, midChannels > 0, numBlocksPerStage >= 0, dims == 3 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Latent upsampler 的 channels、block 數或 dims 無效。"
            )
        }
        guard spatialUpsample != temporalUpsample else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "Latent upsampler 必須且只能啟用 spatial_upsample 或 temporal_upsample 其中一項。"
            )
        }
        guard spatialScale > 0, spatialScale.isFinite else {
            throw LTXVideoRuntimeError.invalidConfiguration("spatial_scale 必須是正數。")
        }
        if rationalResampler && spatialUpsample {
            let supported = [0.75, 1.5, 2.0, 4.0]
            guard supported.contains(where: { abs($0 - spatialScale) < 1e-9 }) else {
                throw LTXVideoRuntimeError.invalidConfiguration(
                    "rational_resampler 不支援 spatial_scale=\(spatialScale)。"
                )
            }
        }
        guard midChannels.isMultiple(of: 32) else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "mid_channels 必須可被 32 整除，才能建立 GroupNorm。"
            )
        }
        self.inChannels = inChannels
        self.midChannels = midChannels
        self.numBlocksPerStage = numBlocksPerStage
        self.dims = dims
        self.spatialUpsample = spatialUpsample
        self.temporalUpsample = temporalUpsample
        self.spatialScale = spatialScale
        self.rationalResampler = rationalResampler
    }

    public static func load(from url: URL) throws -> Self {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LTXVideoRuntimeError.missingFile(url)
        }
        let root = try JSONDecoder().decode(FileConfiguration.self, from: Data(contentsOf: url))
        let values = root.config
        return try Self(
            inChannels: values?.inChannels ?? root.inChannels ?? 128,
            midChannels: values?.midChannels ?? root.midChannels ?? 512,
            numBlocksPerStage: values?.numBlocksPerStage ?? root.numBlocksPerStage ?? 4,
            dims: values?.dims ?? root.dims ?? 3,
            spatialUpsample: values?.spatialUpsample ?? root.spatialUpsample ?? true,
            temporalUpsample: values?.temporalUpsample ?? root.temporalUpsample ?? false,
            spatialScale: values?.spatialScale ?? root.spatialScale ?? 2.0,
            rationalResampler: values?.rationalResampler ?? root.rationalResampler ?? false
        )
    }

    public func outputShape(for latentShape: [Int]) throws -> [Int] {
        guard latentShape.count == 5,
              latentShape[0] > 0,
              latentShape[1] == inChannels,
              latentShape[2] > 0,
              latentShape[3] > 0,
              latentShape[4] > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "latent 必須是 [B,\(inChannels),F,H,W]，實際為 \(latentShape)。"
            )
        }
        var frameCount = latentShape[2]
        var height = latentShape[3]
        var width = latentShape[4]
        if temporalUpsample {
            frameCount = frameCount * 2 - 1
        } else if spatialUpsample {
            if rationalResampler {
                let (numerator, denominator) = rationalFactors
                height = (height * numerator + 4 - 5) / denominator + 1
                width = (width * numerator + 4 - 5) / denominator + 1
            } else {
                height *= 2
                width *= 2
            }
        }
        return [latentShape[0], inChannels, frameCount, height, width]
    }

    private var rationalFactors: (numerator: Int, denominator: Int) {
        switch spatialScale {
        case 0.75: (3, 4)
        case 1.5: (3, 2)
        case 2.0: (2, 1)
        case 4.0: (4, 1)
        default: (1, 1)
        }
    }

    private struct FileConfiguration: Decodable {
        let config: Values?
        let inChannels: Int?
        let midChannels: Int?
        let numBlocksPerStage: Int?
        let dims: Int?
        let spatialUpsample: Bool?
        let temporalUpsample: Bool?
        let spatialScale: Double?
        let rationalResampler: Bool?

        enum CodingKeys: String, CodingKey {
            case config
            case inChannels = "in_channels"
            case midChannels = "mid_channels"
            case numBlocksPerStage = "num_blocks_per_stage"
            case dims
            case spatialUpsample = "spatial_upsample"
            case temporalUpsample = "temporal_upsample"
            case spatialScale = "spatial_scale"
            case rationalResampler = "rational_resampler"
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            config = try container.decodeIfPresent(Values.self, forKey: .config)
            inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels)
            midChannels = try container.decodeIfPresent(Int.self, forKey: .midChannels)
            numBlocksPerStage = try container.decodeIfPresent(Int.self, forKey: .numBlocksPerStage)
            dims = try container.decodeIfPresent(Int.self, forKey: .dims)
            spatialUpsample = try container.decodeIfPresent(Bool.self, forKey: .spatialUpsample)
            temporalUpsample = try container.decodeIfPresent(Bool.self, forKey: .temporalUpsample)
            spatialScale = try container.decodeIfPresent(Double.self, forKey: .spatialScale)
            rationalResampler = try container.decodeIfPresent(Bool.self, forKey: .rationalResampler)
        }

        struct Values: Decodable {
            let inChannels: Int?
            let midChannels: Int?
            let numBlocksPerStage: Int?
            let dims: Int?
            let spatialUpsample: Bool?
            let temporalUpsample: Bool?
            let spatialScale: Double?
            let rationalResampler: Bool?

            enum CodingKeys: String, CodingKey {
                case inChannels = "in_channels"
                case midChannels = "mid_channels"
                case numBlocksPerStage = "num_blocks_per_stage"
                case dims
                case spatialUpsample = "spatial_upsample"
                case temporalUpsample = "temporal_upsample"
                case spatialScale = "spatial_scale"
                case rationalResampler = "rational_resampler"
            }
        }
    }
}

final class LTXLatentUpsamplerResBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv3d
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv3d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm

    init(channels: Int) {
        _conv1 = ModuleInfo(wrappedValue: Conv3d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: .init(3),
            padding: .init(1),
            bias: true
        ), key: "conv1")
        _norm1 = ModuleInfo(wrappedValue: GroupNorm(
            groupCount: 32,
            dimensions: channels,
            pytorchCompatible: true
        ), key: "norm1")
        _conv2 = ModuleInfo(wrappedValue: Conv3d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: .init(3),
            padding: .init(1),
            bias: true
        ), key: "conv2")
        _norm2 = ModuleInfo(wrappedValue: GroupNorm(
            groupCount: 32,
            dimensions: channels,
            pytorchCompatible: true
        ), key: "norm2")
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let residual = input
        var value = silu(norm1(conv1(input)))
        value = conv2(value)
        value = norm2(value)
        return silu(value + residual)
    }
}

private class LTXLatentUpsamplerOperation: Module {
    func apply(_ input: MLXArray) -> MLXArray {
        fatalError("LTXLatentUpsamplerOperation 必須由具體實作取代。")
    }
}

private final class LTXSpatialUpsamplerOperation: LTXLatentUpsamplerOperation {
    @ModuleInfo(key: "0") var convolution: Conv2d

    init(channels: Int) {
        _convolution = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: channels,
            outputChannels: channels * 4,
            kernelSize: 3,
            padding: 1,
            bias: true
        ), key: "0")
        super.init()
    }

    override func apply(_ input: MLXArray) -> MLXArray {
        let batch = input.shape[0]
        let depth = input.shape[1]
        let value = convolution(input.reshaped(batch * depth, input.shape[2], input.shape[3], input.shape[4]))
        let shuffled = LTXLatentUpsamplerOps.pixelShuffle2D(value, factor: 2)
        return shuffled.reshaped(batch, depth, shuffled.shape[1], shuffled.shape[2], shuffled.shape[3])
    }
}

private final class LTXTemporalUpsamplerOperation: LTXLatentUpsamplerOperation {
    @ModuleInfo(key: "0") var convolution: Conv3d

    init(channels: Int) {
        _convolution = ModuleInfo(wrappedValue: Conv3d(
            inputChannels: channels,
            outputChannels: channels * 2,
            kernelSize: .init(3),
            padding: .init(1),
            bias: true
        ), key: "0")
        super.init()
    }

    override func apply(_ input: MLXArray) -> MLXArray {
        LTXLatentUpsamplerOps.pixelShuffle3D(
            convolution(input),
            spatialFactor: 1,
            temporalFactor: 2
        )
    }
}

private final class LTXLatentBlurDownsample: Module {
    public var kernel: MLXArray
    private let stride: Int

    init(stride: Int, kernelSize: Int = 5) {
        self.stride = stride
        let coefficients = (0..<kernelSize).map {
            Float(Self.binomial(kernelSize - 1, $0))
        }
        let values = coefficients.flatMap { row in coefficients.map { row * $0 } }
        let total = values.reduce(0, +)
        self.kernel = MLXArray(values.map { $0 / total })
            .reshaped(1, kernelSize, kernelSize, 1)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        guard stride != 1 else { return input }
        let batch = input.shape[0]
        let channels = input.shape[3]
        let padding = kernel.shape[1] / 2
        var value = input.transposed(0, 3, 1, 2).reshaped(batch * channels, input.shape[1], input.shape[2], 1)
        value = padded(value, widths: [
            .init((0, 0)), .init((padding, padding)),
            .init((padding, padding)), .init((0, 0))
        ])
        value = conv2d(value, kernel, stride: .init(stride))
        return value.reshaped(batch, channels, value.shape[1], value.shape[2])
            .transposed(0, 2, 3, 1)
    }

    private static func binomial(_ n: Int, _ k: Int) -> Int {
        guard k >= 0, k <= n else { return 0 }
        if k == 0 || k == n { return 1 }
        var result = 1
        for index in 1...min(k, n - k) {
            result = result * (n - index + 1) / index
        }
        return result
    }
}

private final class LTXSpatialRationalResampler: LTXLatentUpsamplerOperation {
    @ModuleInfo(key: "conv") var convolution: Conv2d
    @ModuleInfo(key: "blur_down") var blurDown: LTXLatentBlurDownsample

    private let numerator: Int
    private let denominator: Int

    init(channels: Int, scale: Double) throws {
        let mapping: [(scale: Double, numerator: Int, denominator: Int)] = [
            (0.75, 3, 4), (1.5, 3, 2), (2.0, 2, 1), (4.0, 4, 1)
        ]
        guard let selected = mapping.first(where: { abs($0.scale - scale) < 1e-9 }) else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "rational_resampler 不支援 spatial_scale=\(scale)。"
            )
        }
        numerator = selected.numerator
        denominator = selected.denominator
        _convolution = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: channels,
            outputChannels: channels * selected.numerator * selected.numerator,
            kernelSize: 3,
            padding: 1,
            bias: true
        ), key: "conv")
        _blurDown = ModuleInfo(wrappedValue: LTXLatentBlurDownsample(
            stride: selected.denominator
        ), key: "blur_down")
        super.init()
    }

    override func apply(_ input: MLXArray) -> MLXArray {
        let batch = input.shape[0]
        let depth = input.shape[1]
        let value = convolution(input.reshaped(batch * depth, input.shape[2], input.shape[3], input.shape[4]))
        let shuffled = LTXLatentUpsamplerOps.pixelShuffle2D(value, factor: numerator)
        let blurred = blurDown(shuffled)
        return blurred.reshaped(batch, depth, blurred.shape[1], blurred.shape[2], blurred.shape[3])
    }
}

private enum LTXLatentUpsamplerOps {
    static func pixelShuffle2D(_ input: MLXArray, factor: Int) -> MLXArray {
        let shape = input.shape
        let channels = shape[3] / (factor * factor)
        return input
            .reshaped(shape[0], shape[1], shape[2], channels, factor, factor)
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped(shape[0], shape[1] * factor, shape[2] * factor, channels)
    }

    static func pixelShuffle3D(
        _ input: MLXArray,
        spatialFactor: Int,
        temporalFactor: Int
    ) -> MLXArray {
        let shape = input.shape
        let channels = shape[4] / (spatialFactor * spatialFactor * temporalFactor)
        return input
            .reshaped(
                shape[0], shape[1], shape[2], shape[3], channels,
                temporalFactor, spatialFactor, spatialFactor
            )
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped(
                shape[0], shape[1] * temporalFactor,
                shape[2] * spatialFactor, shape[3] * spatialFactor, channels
            )
    }
}

public final class LTXLatentUpsampler: Module {
    @ModuleInfo(key: "initial_conv") public var initialConv: Conv3d
    @ModuleInfo(key: "initial_norm") public var initialNorm: GroupNorm
    @ModuleInfo(key: "res_blocks") var resBlocks: [LTXLatentUpsamplerResBlock]
    @ModuleInfo(key: "upsampler") private var upsampler: LTXLatentUpsamplerOperation
    @ModuleInfo(key: "post_upsample_res_blocks") var postUpsampleResBlocks: [LTXLatentUpsamplerResBlock]
    @ModuleInfo(key: "final_conv") public var finalConv: Conv3d

    public let configuration: LTXLatentUpsamplerConfiguration

    public init(configuration: LTXLatentUpsamplerConfiguration) throws {
        self.configuration = configuration
        self._initialConv = ModuleInfo(wrappedValue: Conv3d(
            inputChannels: configuration.inChannels,
            outputChannels: configuration.midChannels,
            kernelSize: .init(3),
            padding: .init(1),
            bias: true
        ), key: "initial_conv")
        self._initialNorm = ModuleInfo(wrappedValue: GroupNorm(
            groupCount: 32,
            dimensions: configuration.midChannels,
            pytorchCompatible: true
        ), key: "initial_norm")
        self._resBlocks = ModuleInfo(wrappedValue: (0..<configuration.numBlocksPerStage).map { _ in
            LTXLatentUpsamplerResBlock(channels: configuration.midChannels)
        }, key: "res_blocks")

        if configuration.spatialUpsample {
            if configuration.rationalResampler {
                self._upsampler = ModuleInfo(wrappedValue: try LTXSpatialRationalResampler(
                    channels: configuration.midChannels,
                    scale: configuration.spatialScale
                ), key: "upsampler")
            } else {
                self._upsampler = ModuleInfo(wrappedValue: LTXSpatialUpsamplerOperation(
                    channels: configuration.midChannels
                ), key: "upsampler")
            }
        } else {
            self._upsampler = ModuleInfo(wrappedValue: LTXTemporalUpsamplerOperation(
                channels: configuration.midChannels
            ), key: "upsampler")
        }

        self._postUpsampleResBlocks = ModuleInfo(wrappedValue: (0..<configuration.numBlocksPerStage).map { _ in
            LTXLatentUpsamplerResBlock(channels: configuration.midChannels)
        }, key: "post_upsample_res_blocks")
        self._finalConv = ModuleInfo(wrappedValue: Conv3d(
            inputChannels: configuration.midChannels,
            outputChannels: configuration.inChannels,
            kernelSize: .init(3),
            padding: .init(1),
            bias: true
        ), key: "final_conv")
        super.init()
    }

    public func upsample(_ latent: MLXArray) throws -> MLXArray {
        guard latent.ndim == 5,
              latent.shape[0] > 0,
              latent.shape[1] == configuration.inChannels,
              latent.shape[2] > 0,
              latent.shape[3] > 0,
              latent.shape[4] > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "latent 必須是 [B,\(configuration.inChannels),F,H,W]，實際為 \(latent.shape)。"
            )
        }
        return callAsFunction(latent)
    }

    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        let outputDType = latent.dtype
        var value = latent
        if value.dtype != initialConv.weight.dtype {
            value = value.asType(initialConv.weight.dtype)
        }
        value = value.transposed(0, 2, 3, 4, 1)
        value = silu(initialNorm(initialConv(value)))
        for block in resBlocks {
            value = block(value)
        }
        value = upsampler.apply(value)
        if configuration.temporalUpsample {
            value = value[0..., 1..<value.shape[1], 0..., 0..., 0...]
        }
        for block in postUpsampleResBlocks {
            value = block(value)
        }
        value = finalConv(value)
        return value.transposed(0, 4, 1, 2, 3).asType(outputDType)
    }
}
