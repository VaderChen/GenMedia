import Foundation
import MLX

public struct LTXDistilledGenerationConfiguration: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public let frames: Int
    public let frameRate: Float
    public let seed: UInt64
    public let stage1Steps: Int
    public let stage2Steps: Int
    public let computeDType: LTXVideoComputeDType

    public init(
        width: Int,
        height: Int,
        frames: Int,
        frameRate: Float,
        seed: UInt64,
        stage1Steps: Int = 8,
        stage2Steps: Int = 3,
        computeDType: LTXVideoComputeDType = .bfloat16
    ) throws {
        guard width >= 64, height >= 64 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "LTX 輸出寬高必須至少為 64。"
            )
        }
        guard frames > 0, frames % 8 == 1 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "LTX 輸出幀數必須符合 8n+1。"
            )
        }
        guard frameRate.isFinite, frameRate > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration("frameRate 必須是正數。")
        }
        guard stage1Steps > 0, stage2Steps > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "兩階段的 denoise steps 必須是正整數。"
            )
        }
        self.width = width
        self.height = height
        self.frames = frames
        self.frameRate = frameRate
        self.seed = seed
        self.stage1Steps = stage1Steps
        self.stage2Steps = stage2Steps
        self.computeDType = computeDType
    }

    public var snappedDimensions: (height: Int, width: Int) {
        (
            max(64, (height / 64) * 64),
            max(64, (width / 64) * 64)
        )
    }

    public var latentFrameCount: Int {
        (frames + 7) / 8
    }

    public var audioTokenCount: Int {
        Int((Double(frames) / Double(frameRate) * 25).rounded(.toNearestOrEven))
    }

    public var stage1Sigmas: [Float] {
        Array(LTXDiffusionScheduler.distilledSigmas.prefix(
            min(stage1Steps + 1, LTXDiffusionScheduler.distilledSigmas.count)
        ))
    }

    public var stage2Sigmas: [Float] {
        Array(LTXDiffusionScheduler.stage2Sigmas.prefix(
            min(stage2Steps + 1, LTXDiffusionScheduler.stage2Sigmas.count)
        ))
    }
}

public struct LTXVideoLatentStatistics {
    public let meanOfMeans: MLXArray
    public let stdOfMeans: MLXArray

    public init(meanOfMeans: MLXArray, stdOfMeans: MLXArray) throws {
        guard meanOfMeans.shape == [128], stdOfMeans.shape == [128] else {
            throw LTXVideoRuntimeError.invalidConfiguration(
                "影片 latent 統計值必須是 [128]，實際為 \(meanOfMeans.shape) 與 \(stdOfMeans.shape)。"
            )
        }
        self.meanOfMeans = meanOfMeans
        self.stdOfMeans = stdOfMeans
    }
}

public struct LTXDistilledGenerationProgress: Sendable {
    public enum Stage: String, Sendable {
        case stage1Denoising
        case upscaling
        case stage2Denoising
    }

    public let stage: Stage
    public let completed: Int
    public let total: Int

    public init(stage: Stage, completed: Int, total: Int) {
        self.stage = stage
        self.completed = completed
        self.total = max(1, total)
    }

    public var value: Double {
        Double(completed) / Double(total)
    }
}

public struct LTXDistilledGenerationResult {
    public let videoLatent: MLXArray
    public let audioLatent: MLXArray
    public let width: Int
    public let height: Int
    public let frames: Int

    public init(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        width: Int,
        height: Int,
        frames: Int
    ) {
        self.videoLatent = videoLatent
        self.audioLatent = audioLatent
        self.width = width
        self.height = height
        self.frames = frames
    }
}

public enum LTXDistilledPositionBuilder {
    public static func video(
        frameCount: Int,
        height: Int,
        width: Int,
        frameRate: Float
    ) -> MLXArray {
        let index = arange(frameCount).asType(.float32)
        let starts = maximum(index * 8 + 1 - 8, 0)
        let ends = maximum((index + 1) * 8 + 1 - 8, 0)
        let frameMids = (starts + ends) / (2 * frameRate)
        let heightMids = arange(height).asType(.float32) * 32 + 16
        let widthMids = arange(width).asType(.float32) * 32 + 16

        let frameGrid = repeated(
            repeated(frameMids[0..., .newAxis, .newAxis], count: height, axis: 1),
            count: width,
            axis: 2
        )
        let heightGrid = repeated(
            repeated(heightMids[.newAxis, 0..., .newAxis], count: frameCount, axis: 0),
            count: width,
            axis: 2
        )
        let widthGrid = repeated(
            repeated(widthMids[.newAxis, .newAxis, 0...], count: frameCount, axis: 0),
            count: height,
            axis: 1
        )
        return stacked([frameGrid, heightGrid, widthGrid], axis: -1)
            .reshaped(1, frameCount * height * width, 3)
            .asType(.float32)
    }

    public static func audio(tokenCount: Int) -> MLXArray {
        let index = arange(tokenCount).asType(.float32)
        let starts = maximum(index * 4 + 1 - 4, 0) * 160 / 16_000
        let ends = maximum((index + 1) * 4 + 1 - 4, 0) * 160 / 16_000
        return (((starts + ends) / 2)[.newAxis, 0..., .newAxis]).asType(.float32)
    }
}

public final class LTXDistilledGenerationPipeline {
    public let transformer: LTXTransformer
    public let videoStatistics: LTXVideoLatentStatistics
    public let upsampler: LTXLatentUpsampler

    private let videoPatchifier = LTXVideoLatentPatchifier()
    private let audioPatchifier = LTXAudioLatentPatchifier()

    public init(
        transformer: LTXTransformer,
        videoStatistics: LTXVideoLatentStatistics,
        upsampler: LTXLatentUpsampler
    ) {
        self.transformer = transformer
        self.videoStatistics = videoStatistics
        self.upsampler = upsampler
    }

    public func generate(
        videoTextEmbeds: MLXArray,
        audioTextEmbeds: MLXArray,
        configuration: LTXDistilledGenerationConfiguration,
        progress: (@Sendable (LTXDistilledGenerationProgress) -> Void)? = nil
    ) throws -> LTXDistilledGenerationResult {
        let (height, width) = configuration.snappedDimensions
        let halfHeight = height / 2
        let halfWidth = width / 2
        let latentFrames = configuration.latentFrameCount
        let stage1Dimensions = [latentFrames, halfHeight / 32, halfWidth / 32]
        let stage2Dimensions = [latentFrames, height / 32, width / 32]
        let audioTokens = configuration.audioTokenCount
        guard audioTokens > 0 else {
            throw LTXVideoRuntimeError.invalidConfiguration("影片長度不足以建立 audio latent token。")
        }

        let videoPositions1 = LTXDistilledPositionBuilder.video(
            frameCount: stage1Dimensions[0],
            height: stage1Dimensions[1],
            width: stage1Dimensions[2],
            frameRate: configuration.frameRate
        )
        let videoPositions2 = LTXDistilledPositionBuilder.video(
            frameCount: stage2Dimensions[0],
            height: stage2Dimensions[1],
            width: stage2Dimensions[2],
            frameRate: configuration.frameRate
        )
        let audioPositions = LTXDistilledPositionBuilder.audio(tokenCount: audioTokens)
        let stage1Video = randomLatent(
            shape: [1, stage1Dimensions[0] * stage1Dimensions[1] * stage1Dimensions[2], 128],
            seed: configuration.seed,
            dtype: configuration.computeDType.mlxDType
        )
        let stage1Audio = randomLatent(
            shape: [1, audioTokens, 128],
            seed: configuration.seed &+ 1,
            dtype: configuration.computeDType.mlxDType
        )

        let model = LTXX0Model(transformer: transformer)
        let output1 = try LTXDiffusionScheduler.denoise(
            model: model,
            videoLatent: stage1Video,
            audioLatent: stage1Audio,
            videoTextEmbeds: videoTextEmbeds,
            audioTextEmbeds: audioTextEmbeds,
            videoPositions: videoPositions1,
            audioPositions: audioPositions,
            sigmas: configuration.stage1Sigmas,
            progress: { completed, total in
                progress?(
                    LTXDistilledGenerationProgress(
                        stage: .stage1Denoising,
                        completed: completed,
                        total: total
                    )
                )
            }
        )
        MLX.eval(output1.video, output1.audio)

        let halfLatent = try videoPatchifier.unpatchify(output1.video, dimensions: stage1Dimensions)
        let denormalized = denormalizeVideoLatent(halfLatent)
        let upscaled = try upsampler.upsample(denormalized)
        let normalized = normalizeVideoLatent(upscaled)
        let stage2VideoTokens = try videoPatchifier.patchify(normalized).tokens
        progress?(
            LTXDistilledGenerationProgress(stage: .upscaling, completed: 1, total: 1)
        )

        let startSigma = configuration.stage2Sigmas.first ?? 0.909375
        let stage2VideoNoise = randomLatent(
            shape: stage2VideoTokens.shape,
            seed: configuration.seed &+ 2,
            dtype: stage2VideoTokens.dtype
        )
        let stage2AudioNoise = randomLatent(
            shape: output1.audio.shape,
            seed: configuration.seed &+ 2,
            dtype: output1.audio.dtype
        )
        let stage2Video = stage2VideoNoise * startSigma
            + stage2VideoTokens * (1 - startSigma)
        let stage2Audio = stage2AudioNoise * startSigma
            + output1.audio * (1 - startSigma)

        let output2 = try LTXDiffusionScheduler.denoise(
            model: model,
            videoLatent: stage2Video,
            audioLatent: stage2Audio,
            videoCleanLatent: stage2VideoTokens,
            audioCleanLatent: output1.audio,
            videoTextEmbeds: videoTextEmbeds,
            audioTextEmbeds: audioTextEmbeds,
            videoPositions: videoPositions2,
            audioPositions: audioPositions,
            sigmas: configuration.stage2Sigmas,
            progress: { completed, total in
                progress?(
                    LTXDistilledGenerationProgress(
                        stage: .stage2Denoising,
                        completed: completed,
                        total: total
                    )
                )
            }
        )
        MLX.eval(output2.video, output2.audio)

        let videoLatent = try videoPatchifier.unpatchify(output2.video, dimensions: stage2Dimensions)
        let audioLatent = try audioPatchifier.unpatchify(output2.audio)
        return LTXDistilledGenerationResult(
            videoLatent: videoLatent,
            audioLatent: audioLatent,
            width: width,
            height: height,
            frames: configuration.frames
        )
    }

    private func denormalizeVideoLatent(_ latent: MLXArray) -> MLXArray {
        let channelsLast = latent.transposed(0, 2, 3, 4, 1)
        let mean = videoStatistics.meanOfMeans.reshaped(1, 1, 1, 1, -1)
        let standardDeviation = videoStatistics.stdOfMeans.reshaped(1, 1, 1, 1, -1)
        return (channelsLast * standardDeviation + mean)
            .transposed(0, 4, 1, 2, 3)
    }

    private func normalizeVideoLatent(_ latent: MLXArray) -> MLXArray {
        let channelsLast = latent.transposed(0, 2, 3, 4, 1)
        let mean = videoStatistics.meanOfMeans.reshaped(1, 1, 1, 1, -1)
        let standardDeviation = videoStatistics.stdOfMeans.reshaped(1, 1, 1, 1, -1)
        return ((channelsLast - mean) / standardDeviation)
            .transposed(0, 4, 1, 2, 3)
    }

    private func randomLatent(shape: [Int], seed: UInt64, dtype: DType) -> MLXArray {
        MLXRandom.normal(shape, dtype: dtype, key: MLXRandom.key(seed))
    }
}
