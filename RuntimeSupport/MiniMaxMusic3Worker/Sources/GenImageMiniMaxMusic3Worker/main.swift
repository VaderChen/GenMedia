import Darwin
import Foundation
import MLX
import MiniMaxMusic3SwiftRuntime
import Tokenizers

private struct WorkerRequest: Decodable {
    let modelDirectory: String
    let outputPath: String
    let prompt: String
    let lyrics: String
    let seed: Int
    let audioDuration: Float
    let steps: Int
    let arCfgScale: Float
    let flowCfgScale: Float
    let topK: Int
}

private struct WorkerEvent: Encodable {
    let type: String
    let stage: String?
    let value: Double?
    let durationSeconds: Double?
    let sampleRate: Int?
    let numFrames: Int?
    let numChunks: Int?
    let message: String?

    static func progress(stage: String, value: Double) -> Self {
        Self(
            type: "progress",
            stage: stage,
            value: min(1, max(0, value)),
            durationSeconds: nil,
            sampleRate: nil,
            numFrames: nil,
            numChunks: nil,
            message: nil
        )
    }

    static func completed(
        durationSeconds: Double,
        sampleRate: Int,
        numFrames: Int,
        numChunks: Int
    ) -> Self {
        Self(
            type: "completed",
            stage: nil,
            value: 1,
            durationSeconds: durationSeconds,
            sampleRate: sampleRate,
            numFrames: numFrames,
            numChunks: numChunks,
            message: nil
        )
    }

    static func error(_ message: String) -> Self {
        Self(
            type: "error",
            stage: nil,
            value: nil,
            durationSeconds: nil,
            sampleRate: nil,
            numFrames: nil,
            numChunks: nil,
            message: message
        )
    }
}

private enum WorkerError: LocalizedError {
    case usage
    case missingOutput

    var errorDescription: String? {
        switch self {
        case .usage:
            "用法：GenImageMiniMaxMusic3Worker --request <request.json>"
        case .missingOutput:
            "MiniMax Music 3 Worker 完成但沒有產生 WAV。"
        }
    }
}

@main
private enum GenImageMiniMaxMusic3Worker {
    static func main() async {
        do {
            let requestURL = try requestURL(from: CommandLine.arguments)
            let request = try JSONDecoder().decode(
                WorkerRequest.self,
                from: Data(contentsOf: requestURL)
            )
            try await run(request)
        } catch {
            emit(.error(error.localizedDescription))
            Darwin.exit(EXIT_FAILURE)
        }
    }

    private static func run(_ request: WorkerRequest) async throws {
        let modelDirectory = URL(fileURLWithPath: request.modelDirectory, isDirectory: true)
        let outputURL = URL(fileURLWithPath: request.outputPath)
        let tokenizerDirectory = modelDirectory.appendingPathComponent(
            "tokenizer",
            isDirectory: true
        )

        if MiniMaxMusic3GGUFModel.isAvailable(at: modelDirectory) {
            try await runGGUF(request, modelDirectory: modelDirectory)
            return
        }

        emit(.progress(stage: "loadingModel", value: 0.01))
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        let modelConfiguration = try MiniMaxMusic3ModelConfiguration.load(from: modelDirectory)

        let language = try MiniMaxMusic3LanguageModel(modelDirectory: modelDirectory)
        emit(.progress(stage: "loadingModel", value: 0.02))
        let loadedRVQ = try loadRVQDecoder(modelDirectory: modelDirectory)
        emit(.progress(stage: "loadingModel", value: 0.035))
        let loadedCondition = try loadConditionEncoder(modelDirectory: modelDirectory)
        emit(.progress(stage: "loadingModel", value: 0.05))
        let loadedTransformer = try loadFlowTransformer(modelDirectory: modelDirectory)
        emit(.progress(stage: "loadingModel", value: 0.07))
        let decoder = try MiniMaxMusic3Decoder(modelDirectory: modelDirectory)
        emit(.progress(stage: "loadingModel", value: 0.08))

        let generation = MiniMaxMusic3GenerationConfiguration(
            audioDuration: request.audioDuration,
            seed: request.seed,
            numInferenceSteps: request.steps,
            arCFGScale: request.arCfgScale,
            flowCFGScale: request.flowCfgScale,
            topK: request.topK,
            inputDType: .bfloat16
        )
        let pipeline = try MiniMaxMusic3Pipeline(
            modelConfiguration: modelConfiguration,
            languageModel: language,
            rvqDepthDecoder: loadedRVQ.model,
            conditionEncoder: loadedCondition.model,
            transformer: loadedTransformer.model,
            vocoder: decoder.vocoder,
            tokenEncoder: { tokenizer.encode(text: $0) }
        )

        let result = try pipeline.generate(
            prompt: request.prompt,
            lyrics: request.lyrics,
            generation: generation,
            progress: { event in
                let value: Double
                let stage: String
                switch event.stage {
                case .autoregressive:
                    stage = "autoregressive"
                    value = 0.08 + event.value * 0.72
                case .denoising:
                    stage = "denoising"
                    value = 0.80 + event.value * 0.17
                case .vocoder:
                    stage = "vocoder"
                    value = 0.97 + event.value * 0.02
                }
                emit(.progress(stage: stage, value: value))
            }
        )

        let audioReport = try MiniMaxMusic3Audio.writeWAV(
            result.audio,
            sampleRate: result.samplingRate,
            to: outputURL
        )
        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            throw WorkerError.missingOutput
        }
        emit(
            .completed(
                durationSeconds: audioReport.durationSeconds,
                sampleRate: audioReport.sampleRate,
                numFrames: result.numFrames,
                numChunks: result.numChunks
            )
        )
    }

    private static func runGGUF(
        _ request: WorkerRequest,
        modelDirectory: URL
    ) async throws {
        emit(.progress(stage: "loadingModel", value: 0.01))
        let tokenizerDirectory = modelDirectory.appendingPathComponent(
            "tokenizer",
            isDirectory: true
        )
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        let language = try MiniMaxMusic3LanguageModel(
            ggufModelDirectory: modelDirectory
        )
        emit(.progress(stage: "loadingModel", value: 0.18))

        let rvqURL = try MiniMaxMusic3GGUFModel.componentURL(
            for: .rvqDepthDecoder,
            at: modelDirectory
        )
        let rvqBits = try MiniMaxMusic3GGUFModel.quantizationBits(
            for: rvqURL,
            component: .rvqDepthDecoder
        )
        let rvq = MiniMaxMusic3RVQDepthDecoder(
            configuration: .music3,
            quantizationBits: rvqBits,
            quantizeAudioHeads: true,
            quantizeProjection: true
        )
        _ = try MiniMaxMusic3RVQDecoderWeightLoader.loadGGUF(
            model: rvq,
            from: rvqURL
        )
        emit(.progress(stage: "loadingModel", value: 0.27))

        let conditionURL = try MiniMaxMusic3GGUFModel.componentURL(
            for: .conditionEncoder,
            at: modelDirectory
        )
        let condition = MiniMaxMusic3ConditionEncoder(configuration: .music3)
        _ = try MiniMaxMusic3ConditionEncoderWeightLoader.loadGGUF(
            model: condition,
            from: conditionURL
        )
        emit(.progress(stage: "loadingModel", value: 0.31))

        let transformerURL = try MiniMaxMusic3GGUFModel.componentURL(
            for: .transformer,
            at: modelDirectory
        )
        let transformerBits = try MiniMaxMusic3GGUFModel.quantizationBits(
            for: transformerURL,
            component: .transformer
        )
        let transformer = MiniMaxMusic3FlowTransformer(
            configuration: .music3,
            quantizationBits: transformerBits
        )
        _ = try MiniMaxMusic3FlowTransformerWeightLoader.loadGGUF(
            model: transformer,
            from: transformerURL
        )
        emit(.progress(stage: "loadingModel", value: 0.46))

        let vocoderURL = try MiniMaxMusic3GGUFModel.componentURL(
            for: .vocoder,
            at: modelDirectory
        )
        let vocoder = MiniMaxMusic3Vocoder(configuration: .music3)
        _ = try MiniMaxMusic3VocoderWeightLoader.loadGGUF(
            model: vocoder,
            from: vocoderURL
        )
        emit(.progress(stage: "loadingModel", value: 0.5))

        let generation = MiniMaxMusic3GenerationConfiguration(
            audioDuration: request.audioDuration,
            seed: request.seed,
            numInferenceSteps: request.steps,
            arCFGScale: request.arCfgScale,
            flowCFGScale: request.flowCfgScale,
            topK: request.topK,
            inputDType: .bfloat16
        )
        let pipeline = try MiniMaxMusic3Pipeline(
            modelConfiguration: .music3,
            languageModel: language,
            rvqDepthDecoder: rvq,
            conditionEncoder: condition,
            transformer: transformer,
            vocoder: vocoder,
            tokenEncoder: { tokenizer.encode(text: $0) }
        )
        let result = try pipeline.generate(
            prompt: request.prompt,
            lyrics: request.lyrics,
            generation: generation,
            progress: { event in
                let value: Double
                let stage: String
                switch event.stage {
                case .autoregressive:
                    stage = "autoregressive"
                    value = 0.5 + event.value * 0.38
                case .denoising:
                    stage = "denoising"
                    value = 0.88 + event.value * 0.08
                case .vocoder:
                    stage = "vocoder"
                    value = 0.96 + event.value * 0.03
                }
                emit(.progress(stage: stage, value: value))
            }
        )
        let outputURL = URL(fileURLWithPath: request.outputPath)
        let audioReport = try MiniMaxMusic3Audio.writeWAV(
            result.audio,
            sampleRate: result.samplingRate,
            to: outputURL
        )
        guard FileManager.default.fileExists(atPath: outputURL.path) else {
            throw WorkerError.missingOutput
        }
        emit(
            .completed(
                durationSeconds: audioReport.durationSeconds,
                sampleRate: audioReport.sampleRate,
                numFrames: result.numFrames,
                numChunks: result.numChunks
            )
        )
    }

    private static func loadRVQDecoder(
        modelDirectory: URL
    ) throws -> (
        model: MiniMaxMusic3RVQDepthDecoder,
        report: MiniMaxMusic3RVQDecoderWeightLoadReport
    ) {
        let configuration = try MiniMaxMusic3RVQDecoderConfiguration.load(from: modelDirectory)
        let model = MiniMaxMusic3RVQDepthDecoder(configuration: configuration)
        let report = try MiniMaxMusic3RVQDecoderWeightLoader.load(
            model: model,
            from: modelDirectory
        )
        return (model, report)
    }

    private static func loadConditionEncoder(
        modelDirectory: URL
    ) throws -> (
        model: MiniMaxMusic3ConditionEncoder,
        report: MiniMaxMusic3ConditionEncoderWeightLoadReport
    ) {
        let configuration = try MiniMaxMusic3ConditionEncoderConfiguration.load(from: modelDirectory)
        let model = MiniMaxMusic3ConditionEncoder(configuration: configuration)
        let report = try MiniMaxMusic3ConditionEncoderWeightLoader.load(
            model: model,
            from: modelDirectory
        )
        return (model, report)
    }

    private static func loadFlowTransformer(
        modelDirectory: URL
    ) throws -> (
        model: MiniMaxMusic3FlowTransformer,
        report: MiniMaxMusic3FlowTransformerWeightLoadReport
    ) {
        let configuration = try MiniMaxMusic3FlowTransformerConfiguration.load(from: modelDirectory)
        let model = MiniMaxMusic3FlowTransformer(configuration: configuration)
        let report = try MiniMaxMusic3FlowTransformerWeightLoader.load(
            model: model,
            from: modelDirectory
        )
        return (model, report)
    }

    private static func requestURL(from arguments: [String]) throws -> URL {
        guard arguments.count == 3, arguments[1] == "--request" else {
            throw WorkerError.usage
        }
        return URL(fileURLWithPath: arguments[2])
    }

    private static func emit(_ event: WorkerEvent) {
        guard let data = try? JSONEncoder().encode(event) else { return }
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0A]))
        try? FileHandle.standardOutput.synchronize()
    }
}
