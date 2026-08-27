import Foundation
import MLX

public struct MiniMaxMusic3PipelineProgress: Sendable {
    public enum Stage: String, Sendable {
        case autoregressive
        case denoising
        case vocoder
    }

    public let stage: Stage
    public let value: Double

    public init(stage: Stage, value: Double) {
        self.stage = stage
        self.value = value
    }
}

public struct MiniMaxMusic3GenerationResult {
    public let audio: MLXArray
    public let samplingRate: Int
    public let numFrames: Int
    public let numChunks: Int
    public let iterations: Int
    public let stoppedByEndToken: Bool

    public init(
        audio: MLXArray,
        samplingRate: Int,
        numFrames: Int,
        numChunks: Int,
        iterations: Int,
        stoppedByEndToken: Bool
    ) {
        self.audio = audio
        self.samplingRate = samplingRate
        self.numFrames = numFrames
        self.numChunks = numChunks
        self.iterations = iterations
        self.stoppedByEndToken = stoppedByEndToken
    }
}

public enum MiniMaxMusic3PipelineError: LocalizedError, Sendable {
    case invalidComponents(String)
    case invalidInput(String)
    case emptyLatentChunks

    public var errorDescription: String? {
        switch self {
        case let .invalidComponents(message):
            "MiniMax Music 3 pipeline 元件設定不一致：\(message)"
        case let .invalidInput(message):
            "MiniMax Music 3 pipeline 輸入無效：\(message)"
        case .emptyLatentChunks:
            "MiniMax Music 3 pipeline 沒有 latent chunk。"
        }
    }
}

public final class MiniMaxMusic3Pipeline {
    public let modelConfiguration: MiniMaxMusic3ModelConfiguration
    public let languageModel: MiniMaxMusic3LanguageModel
    public let rvqDepthDecoder: MiniMaxMusic3RVQDepthDecoder
    public let conditionEncoder: MiniMaxMusic3ConditionEncoder
    public let transformer: MiniMaxMusic3FlowTransformer
    public let vocoder: MiniMaxMusic3Vocoder

    private let tokenEncoder: (String) -> [Int]

    public init(
        modelConfiguration: MiniMaxMusic3ModelConfiguration = .music3,
        languageModel: MiniMaxMusic3LanguageModel,
        rvqDepthDecoder: MiniMaxMusic3RVQDepthDecoder,
        conditionEncoder: MiniMaxMusic3ConditionEncoder,
        transformer: MiniMaxMusic3FlowTransformer,
        vocoder: MiniMaxMusic3Vocoder,
        tokenEncoder: @escaping (String) -> [Int]
    ) throws {
        try modelConfiguration.validate()
        guard rvqDepthDecoder.configuration.numberOfCodebooks
                == modelConfiguration.numberOfCodebooks else {
            throw MiniMaxMusic3PipelineError.invalidComponents(
                "RVQ codebook 數量與模型設定不同。"
            )
        }
        guard rvqDepthDecoder.configuration.audioVocabularySize
                == modelConfiguration.audioVocabularySize else {
            throw MiniMaxMusic3PipelineError.invalidComponents(
                "RVQ audio vocabulary 與模型設定不同。"
            )
        }
        guard rvqDepthDecoder.configuration.hiddenSize == languageModel.hiddenSize,
              conditionEncoder.configuration.conditionHiddenDimension == languageModel.hiddenSize else {
            throw MiniMaxMusic3PipelineError.invalidComponents(
                "language、RVQ 與 condition hidden size 不一致。"
            )
        }
        guard conditionEncoder.configuration.numberOfConditionLayers
                == modelConfiguration.numberOfCodebooks else {
            throw MiniMaxMusic3PipelineError.invalidComponents(
                "condition layer 數量與 codebook 數量不同。"
            )
        }
        guard conditionEncoder.configuration.outputDimension
                == transformer.configuration.conditionDim else {
            throw MiniMaxMusic3PipelineError.invalidComponents(
                "condition output 與 transformer condition dimension 不同。"
            )
        }
        guard transformer.configuration.inChannels == vocoder.configuration.latentChannels else {
            throw MiniMaxMusic3PipelineError.invalidComponents(
                "transformer latent channels 與 vocoder 不同。"
            )
        }
        self.modelConfiguration = modelConfiguration
        self.languageModel = languageModel
        self.rvqDepthDecoder = rvqDepthDecoder
        self.conditionEncoder = conditionEncoder
        self.transformer = transformer
        self.vocoder = vocoder
        self.tokenEncoder = tokenEncoder
    }

    public func encodePrompt(prompt: String, lyrics: String) throws -> MLXArray {
        let tokenIDs = try MiniMaxMusic3Prompt.buildCFGTokenIDs(
            prompt: prompt,
            lyrics: lyrics,
            encode: tokenEncoder,
            configuration: MiniMaxMusic3PromptConfiguration(
                audioCfgTokenID: modelConfiguration.audioCFGTokenID,
                maxPromptTokens: modelConfiguration.maxPromptTokens
            )
        )
        guard let length = tokenIDs.first?.count, length > 0,
              tokenIDs.count == 2,
              tokenIDs.allSatisfy({ $0.count == length }) else {
            throw MiniMaxMusic3PipelineError.invalidInput(
                "prompt token 序列必須是兩組等長且非空陣列。"
            )
        }
        return MLXArray(
            tokenIDs.flatMap { $0 }.map(Int32.init),
            [2, length]
        )
    }

    public func generate(
        prompt: String,
        lyrics: String,
        generation: MiniMaxMusic3GenerationConfiguration = .init(),
        progress: (@Sendable (MiniMaxMusic3PipelineProgress) -> Void)? = nil
    ) throws -> MiniMaxMusic3GenerationResult {
        try generation.validate()
        let key = MLXRandom.key(UInt64(bitPattern: Int64(generation.seed)))
        let textIDs = try encodePrompt(prompt: prompt, lyrics: lyrics)
        let frameResult = try MiniMaxMusic3AutoregressiveGenerator
            .generateFrameHiddens(
                textIDs: textIDs,
                languageModel: languageModel,
                decoder: rvqDepthDecoder,
                modelConfiguration: modelConfiguration,
                generation: generation,
                key: key,
                inputDType: generation.inputDType,
                progress: { completed, maximum in
                    let denominator = max(1, maximum)
                    progress?(
                        MiniMaxMusic3PipelineProgress(
                            stage: .autoregressive,
                            value: Double(completed) / Double(denominator)
                        )
                    )
                }
            )
        let frameHiddens = frameResult.frameHiddens
        let autoregressiveKey = frameResult.nextKey
        let (latentChunks, _) = try denoiseChunks(
            frameHiddens: frameHiddens,
            generation: generation,
            key: autoregressiveKey,
            progress: { completed, total in
                progress?(
                    MiniMaxMusic3PipelineProgress(
                        stage: .denoising,
                        value: Double(completed) / Double(max(1, total))
                    )
                )
            }
        )
        let audio = try decodeChunks(
            latentChunks,
            progress: { completed, total in
                progress?(
                    MiniMaxMusic3PipelineProgress(
                        stage: .vocoder,
                        value: Double(completed) / Double(max(1, total))
                    )
                )
            }
        )
        return MiniMaxMusic3GenerationResult(
            audio: audio,
            samplingRate: vocoder.configuration.samplingRate,
            numFrames: frameHiddens.shape[1],
            numChunks: latentChunks.count,
            iterations: frameResult.iterations,
            stoppedByEndToken: frameResult.stoppedByEndToken
        )
    }

    private func denoiseChunks(
        frameHiddens: MLXArray,
        generation: MiniMaxMusic3GenerationConfiguration,
        key: MLXArray,
        progress: (@Sendable (_ completed: Int, _ total: Int) -> Void)? = nil
    ) throws -> ([MLXArray], MLXArray) {
        guard frameHiddens.ndim == 3,
              frameHiddens.shape[0] > 0,
              frameHiddens.shape[1] > 0,
              frameHiddens.shape[2]
                == conditionEncoder.configuration.numberOfConditionLayers
                * conditionEncoder.configuration.conditionHiddenDimension else {
            throw MiniMaxMusic3PipelineError.invalidInput(
                "frame hiddens shape 必須符合 condition encoder 設定。"
            )
        }

        let starts = try MiniMaxMusic3ChunkLayout.starts(for: frameHiddens.shape[1])
        let timestepValues = try MiniMaxMusic3FlowScheduler.flowTimestepValues(
            numInferenceSteps: generation.numInferenceSteps
        )
        var latentChunks: [MLXArray] = []
        var previousLatent: MLXArray?
        var previousCondition: MLXArray?
        var nextKey = key

        let totalSteps = max(1, starts.count * timestepValues.count)
        var completedSteps = 0
        for start in starts {
            let end = min(start + MiniMaxMusic3ChunkLayout.chunkFrames, frameHiddens.shape[1])
            var condition = try conditionEncoder(
                frameHiddens[0..., start..<end, 0...]
            )
            var overlap = 0
            if let previousLatent, let previousCondition {
                overlap = min(previousLatent.shape[1], condition.shape[1])
                condition = replacePrefix(
                    sequence: condition,
                    prefix: previousCondition,
                    length: overlap
                )
            }

            let randomKeys = MLXRandom.split(key: nextKey, into: 2)
            nextKey = randomKeys[0]
            var latents = MLXRandom.normal(
                [1, condition.shape[1], transformer.configuration.inChannels],
                dtype: condition.dtype,
                key: randomKeys[1]
            )
            let noisePrompt = latents[0..., 0..<overlap, 0...]

            for timestepValue in timestepValues {
                let timestep = MLXArray(
                    Array(repeating: timestepValue, count: latents.shape[0])
                )
                if overlap > 0 {
                    latents = try MiniMaxMusic3FlowScheduler.blendOverlap(
                        latents: latents,
                        noisePrompt: noisePrompt,
                        previousLatent: previousLatent!,
                        overlap: overlap,
                        timestep: timestep
                    )
                }
                let conditional = try transformer(
                    latents,
                    timestep: timestep,
                    encoderHiddenStates: condition
                )
                let unconditional = try transformer(
                    latents,
                    timestep: timestep,
                    encoderHiddenStates: MLXArray.zeros(like: condition)
                )
                let velocity = unconditional + generation.flowCFGScale * (
                    conditional - unconditional
                )
                latents = try MiniMaxMusic3FlowScheduler.eulerStep(
                    sample: latents,
                    velocity: velocity,
                    numInferenceSteps: generation.numInferenceSteps
                )
                MLX.eval(latents)
                completedSteps += 1
                progress?(completedSteps, totalSteps)
            }

            if overlap > 0 {
                latents = try MiniMaxMusic3FlowScheduler.restoreOverlap(
                    latents: latents,
                    previousLatent: previousLatent!,
                    overlap: overlap
                )
            }
            previousLatent = try MiniMaxMusic3FlowScheduler.carryWindow(latents)
            previousCondition = try MiniMaxMusic3FlowScheduler.carryWindow(condition)
            latentChunks.append(latents)
            MLX.eval(previousLatent!, previousCondition!, latents)
        }

        return (latentChunks, nextKey)
    }

    private func decodeChunks(
        _ latentChunks: [MLXArray],
        progress: (@Sendable (_ completed: Int, _ total: Int) -> Void)? = nil
    ) throws -> MLXArray {
        guard !latentChunks.isEmpty else {
            throw MiniMaxMusic3PipelineError.emptyLatentChunks
        }
        var waveforms: [MLXArray] = []
        for (index, latents) in latentChunks.enumerated() {
            guard latents.ndim == 3,
                  latents.shape[2] == transformer.configuration.inChannels else {
                throw MiniMaxMusic3PipelineError.invalidInput(
                    "latent chunk 必須是 [batch, frames, channels]。"
                )
            }
            let waveform = MLX.clip(
                vocoder(latents).asType(.float32),
                min: Float(-1),
                max: Float(1)
            )
            let crop = try MiniMaxMusic3ChunkLayout.waveformCrop(
                chunkIndex: index,
                chunkCount: latentChunks.count,
                hopLength: vocoder.configuration.hopLength
            )
            let length = waveform.shape[2]
            let end = crop.right == 0 ? length : length - crop.right
            let normalizedEnd = end >= 0 ? end : length + end
            guard crop.left <= normalizedEnd else {
                throw MiniMaxMusic3PipelineError.invalidInput(
                    "vocoder 輸出短於 chunk crop 範圍。"
                )
            }
            waveforms.append(waveform[0..., 0..., crop.left..<normalizedEnd])
            progress?(index + 1, latentChunks.count)
        }
        let audio = MLX.concatenated(waveforms, axis: 2)
        let clipped = MLX.clip(audio.asType(.float32), min: Float(-1), max: Float(1))
        MLX.eval(clipped)
        return clipped
    }

    private func replacePrefix(
        sequence: MLXArray,
        prefix: MLXArray,
        length: Int
    ) -> MLXArray {
        guard length > 0 else { return sequence }
        return MLX.concatenated(
            [
                prefix[0..., 0..<length, 0...],
                sequence[0..., length..<sequence.shape[1], 0...]
            ],
            axis: 1
        )
    }
}
