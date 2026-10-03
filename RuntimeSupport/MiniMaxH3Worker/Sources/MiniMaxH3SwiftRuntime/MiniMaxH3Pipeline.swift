import Foundation
import GenImageCore
import MLX
import MLXRandom

/// End-to-end text-to-video-audio sampling for MiniMax H3.
///
/// Stages the two large models rather than holding both: the transformer is
/// released before the VAE loads, since together they exceed what fits
/// comfortably alongside activations.
public struct MiniMaxH3Pipeline {
    public enum Error: LocalizedError, Sendable {
        case missingTextEncoder
        case invalidLatentGeometry(height: Int, width: Int)

        public var errorDescription: String? {
            switch self {
            case .missingTextEncoder:
                "H3 有條件生成需要 --text-encoder 與 --tokenizer；未提供時只能執行空文字的 unconditional smoke test。"
            case let .invalidLatentGeometry(height, width):
                "H3 latent 空間尺寸無效：\(height)x\(width)。Transformer 的 2x2 patch 需要 latent 寬高皆為偶數且至少 4。"
            }
        }
    }

    public struct Request: Sendable {
        public var latentFrames: Int
        public var latentHeight: Int
        public var latentWidth: Int
        public var audioFrames: Int
        public var steps: Int
        public var seed: UInt64
        public var textLength: Int
        public var prompt: String

        public init(
            latentFrames: Int = 3,
            latentHeight: Int = 16,
            latentWidth: Int = 16,
            audioFrames: Int = 8,
            steps: Int = 8,
            seed: UInt64 = 0,
            textLength: Int = 8,
            prompt: String = ""
        ) {
            self.latentFrames = latentFrames
            self.latentHeight = latentHeight
            self.latentWidth = latentWidth
            self.audioFrames = audioFrames
            self.steps = steps
            self.seed = seed
            self.textLength = textLength
            self.prompt = prompt
        }

        /// Pixel dimensions the video VAE will produce.
        public func outputSize(
            _ vae: MiniMaxH3VideoVAEConfiguration = .default
        ) -> (frames: Int, height: Int, width: Int) {
            (latentFrames * vae.temporalRatio,
             latentHeight * vae.spatialRatio,
             latentWidth * vae.spatialRatio)
        }
    }

    public struct Result {
        /// `[1, 3, frames, height, width]` in [0, 1].
        public let pixels: MLXArray
        /// `[1, 2, samples]` in [-1, 1] at 32 kHz, when the audio VAE ran.
        public let audio: MLXArray?
        public let videoLatent: MLXArray
        public let audioLatent: MLXArray
    }

    public var transformerURL: URL
    public var videoVAEURL: URL
    public var audioVAEURL: URL?
    public var textEncoderURL: URL?
    public var tokenizerDirectoryURL: URL?
    public var scheduler: MiniMaxH3FlowScheduler
    public var loras: [MiniMaxH3LoRAConfiguration]
    public var acceleration: MiniMaxH3Acceleration?

    public init(
        transformerURL: URL,
        videoVAEURL: URL,
        audioVAEURL: URL? = nil,
        textEncoderURL: URL? = nil,
        tokenizerDirectoryURL: URL? = nil,
        scheduler: MiniMaxH3FlowScheduler = MiniMaxH3FlowScheduler(),
        loras: [MiniMaxH3LoRAConfiguration] = [],
        acceleration: MiniMaxH3Acceleration? = nil
    ) {
        self.transformerURL = transformerURL
        self.videoVAEURL = videoVAEURL
        self.audioVAEURL = audioVAEURL
        self.textEncoderURL = textEncoderURL
        self.tokenizerDirectoryURL = tokenizerDirectoryURL
        self.scheduler = acceleration.map {
            MiniMaxH3FlowScheduler(videoShift: $0.videoShift, audioShift: $0.audioShift)
        } ?? scheduler
        self.loras = loras
        self.acceleration = acceleration
    }

    /// Run the sampling loop and decode.
    ///
    /// - Parameters:
    ///   - textStates: `[textLength, 5120]` encoder hidden states. Passing nil
    ///     uses zeros, i.e. an unconditional run — useful to exercise the
    ///     pipeline before the text encoder exists, but it will not follow a
    ///     prompt.
    ///   - conditioning: FL2VA keyframe anchors. The worker encodes image
    ///     anchors before calling this pipeline and passes the resulting latents.
    ///   - progress: called with a stage label and a 0...1 fraction.
    public func run(
        request: Request,
        textStates: MLXArray? = nil,
        conditioning: MiniMaxH3Transformer.Conditioning? = nil,
        progress: (String, Double) -> Void = { _, _ in }
    ) throws -> Result {
        guard request.latentHeight >= 4,
              request.latentWidth >= 4,
              request.latentHeight.isMultiple(of: 2),
              request.latentWidth.isMultiple(of: 2) else {
            throw Error.invalidLatentGeometry(
                height: request.latentHeight,
                width: request.latentWidth
            )
        }
        try MiniMaxH3LoRAAdapter.validateRequest(loras: loras, acceleration: acceleration, steps: request.steps)
        if let acceleration, !acceleration.supportsPrunedBase {
            let inventory = try MiniMaxH3GGUFWeightLoader.inspectTransformer(fileURL: transformerURL)
            guard try !MiniMaxH3Configuration.forInventory(inventory).usesAdalnCurves else {
                throw MiniMaxH3LoRAError.invalid("此加速 LoRA 不支援 Pruned 基底；請使用完整 FL2VA。")
            }
        }
        MLXRandom.seed(request.seed)
        let defaultConfiguration = MiniMaxH3Configuration.fl2va

        var videoLatent = MLXRandom.normal([
            1, defaultConfiguration.videoLatentChannels, request.latentFrames,
            request.latentHeight, request.latentWidth
        ])
        var audioLatent = MLXRandom.normal([
            1, defaultConfiguration.audioLatentChannels, 2, request.audioFrames
        ])
        var textEncoder: MiniMaxH3Qwen3VLTextEncoder?
        let text: MLXArray
        if let textStates {
            text = textStates
        } else if !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            guard let textEncoderURL, let tokenizerDirectoryURL else {
                throw Error.missingTextEncoder
            }
            progress("loadingTextEncoder", 0)
            textEncoder = try MiniMaxH3Qwen3VLTextEncoder.load(
                fileURL: textEncoderURL,
                tokenizerDirectory: tokenizerDirectoryURL
            )
            text = try textEncoder!.hiddenStates(for: request.prompt)
            MLX.eval(text)
            progress("loadingTextEncoder", 1)
            textEncoder = nil
            MLX.GPU.clearCache()
        } else {
            text = MLX.zeros([request.textLength, defaultConfiguration.conditionInputDim])
        }

        progress("loadingTransformer", 0)
        var transformer: MiniMaxH3Transformer? = try {
            // Keep the loader's tensor dictionary in this scope so releasing
            // the transformer also releases its base weights before VAE decode.
            let loaded = try MiniMaxH3GGUFQuantizedLoader.load(fileURL: transformerURL)
            var configuration = loaded.configuration
            configuration.videoShift = scheduler.videoShift
            configuration.audioShift = scheduler.audioShift
            return MiniMaxH3Transformer(
                configuration: configuration,
                weights: loaded.tensors,
                quantizedPrefixes: loaded.quantizedPrefixes,
                computeDType: .float32
            )
        }()
        let adaptedLayers = try transformer!.loadLoRAs(loras)
        if adaptedLayers > 0 { progress("loadingLoRA", 1) }
        progress("loadingTransformer", 1)

        do {
            // Text refinement is independent of sigma. Evaluate once after LoRA
            // application and release the prepared context before VAE decoding.
            let refinedText = text.shape[1] == transformer!.configuration.hiddenSize
                ? text : try transformer!.refineTextStates(text)
            MLX.eval(refinedText)
            let sigmas = scheduler.sigmas(steps: request.steps)
            for step in 0 ..< request.steps {
                let sigma = sigmas[step]
                let nextSigma = sigmas[step + 1]

                // Carry the audio latent onto the video schedule so the pack
                // behaves as one flow latent (ModelSamplingAV).
                let carry = scheduler.carry(videoSigma: sigma)
                let carriedAudio = audioLatent * Float(carry)

                let output = try transformer!.forward(
                    videoLatent: videoLatent,
                    audioLatent: carriedAudio,
                    textStates: refinedText,
                    sigma: Float(sigma),
                    conditioning: conditioning
                )
                let audioVelocity = scheduler.uncarryAudioVelocity(
                    output.audio, carriedAudio: carriedAudio, videoSigma: sigma
                )

                videoLatent = MiniMaxH3FlowScheduler.eulerStep(
                    videoLatent, velocity: output.video,
                    sigma: sigma, nextSigma: nextSigma
                )
                audioLatent = MiniMaxH3FlowScheduler.eulerStep(
                    audioLatent, velocity: audioVelocity,
                    sigma: sigma, nextSigma: nextSigma
                )
                MLX.eval(videoLatent, audioLatent)
                progress("denoising", Double(step + 1) / Double(request.steps))
            }
        }

        // Release the transformer before the VAE loads.
        transformer = nil
        MLX.GPU.clearCache()

        progress("videoDecoding", 0)
        let decoder = try MiniMaxH3VideoVAEDecoder.load(fileURL: videoVAEURL)
        let scaled = MiniMaxH3VideoVAEDecoder.denormalizeLatent(videoLatent)
        let pixels: MLXArray
        if decoder.shouldUseTemporalTiling(for: scaled) {
            pixels = try decoder.decodeTemporalTiledPixels(latent: scaled)
        } else {
            let raw = try decoder.decode(latent: scaled)
            pixels = MiniMaxH3VideoVAEDecoder.finalizePixels(raw)
        }
        MLX.eval(pixels)
        progress("videoDecoding", 1)

        var audio: MLXArray?
        if let audioVAEURL {
            progress("audioDecoding", 0)
            let audioDecoder = try MiniMaxH3AudioVAEDecoder.load(fileURL: audioVAEURL)
            let waveform = try audioDecoder.decode(latent: audioLatent)
            MLX.eval(waveform)
            audio = waveform
            progress("audioDecoding", 1)
        }

        return Result(
            pixels: pixels, audio: audio,
            videoLatent: videoLatent, audioLatent: audioLatent
        )
    }
}
