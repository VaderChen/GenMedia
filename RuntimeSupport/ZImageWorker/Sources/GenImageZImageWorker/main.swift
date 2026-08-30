import Darwin
import Foundation
import Logging
import ZImage

private struct WorkerRequest: Decodable {
    var modelDirectory: String
    var outputPaths: [String]
    var prompt: String
    var negativePrompt: String
    var width: Int
    var height: Int
    var steps: Int
    var seed: UInt64
    var loraPath: String?
    var loraScale: Double?
}

private struct WorkerEvent: Encodable {
    var type: String
    var stage: String?
    var value: Double?
    var message: String?
}

private enum WorkerError: LocalizedError {
    case usage
    case invalidRequest(String)
    case missingFile(URL)
    case outputMissing(URL)
    case pipelineFailure(String)

    var errorDescription: String? {
        switch self {
        case .usage:
            "用法：GenImageZImageWorker --request <request.json>"
        case let .invalidRequest(message):
            "Z-Image Worker 請求無效：\(message)"
        case let .missingFile(url):
            "找不到必要檔案：\(url.path)"
        case let .outputMissing(url):
            "Z-Image 沒有產生輸出檔案：\(url.path)"
        case let .pipelineFailure(message):
            message
        }
    }
}

@main
private enum GenImageZImageWorker {
    static func main() async {
        do {
            LoggingSystem.bootstrap { label in
                var handler = StreamLogHandler.standardError(label: label)
                handler.logLevel = .warning
                return handler
            }
            let requestURL = try requestURL(from: CommandLine.arguments)
            let request = try JSONDecoder().decode(
                WorkerRequest.self,
                from: Data(contentsOf: requestURL)
            )
            try await run(request)
        } catch {
            emit(
                WorkerEvent(type: "error", stage: nil, value: nil, message: error.localizedDescription),
                to: .standardError
            )
            Darwin.exit(EXIT_FAILURE)
        }
    }

    private static func run(_ request: WorkerRequest) async throws {
        let modelURL = URL(fileURLWithPath: request.modelDirectory, isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw WorkerError.missingFile(modelURL)
        }
        guard !request.outputPaths.isEmpty else {
            throw WorkerError.invalidRequest("outputPaths 不可為空。")
        }
        guard request.outputPaths.allSatisfy({ !$0.isEmpty }) else {
            throw WorkerError.invalidRequest("outputPaths 含有空白路徑。")
        }

        let lora: LoRAConfiguration?
        if let loraPath = request.loraPath {
            let loraURL = URL(fileURLWithPath: loraPath)
            guard FileManager.default.fileExists(atPath: loraURL.path) else {
                throw WorkerError.missingFile(loraURL)
            }
            lora = .local(loraURL, scale: Float(request.loraScale ?? 1))
        } else {
            lora = nil
        }

        let outputURLs = request.outputPaths.map { URL(fileURLWithPath: $0) }
        for outputURL in outputURLs {
            try FileManager.default.createDirectory(
                at: outputURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        }

        let pipeline = ZImagePipeline(logger: Logger(label: "genimage.zimage.worker"))
        let outputCount = outputURLs.count
        for (index, outputURL) in outputURLs.enumerated() {
            try Task.checkCancellation()
            let generationRequest = ZImageGenerationRequest(
                prompt: request.prompt,
                negativePrompt: request.negativePrompt.isEmpty ? nil : request.negativePrompt,
                width: request.width,
                height: request.height,
                steps: request.steps,
                guidanceScale: 0,
                seed: request.seed &+ UInt64(index),
                outputPath: outputURL,
                model: modelURL.path,
                lora: lora,
                runtimeOptions: ZImageRuntimeOptions(residencyPolicy: .warm)
            )
            do {
                _ = try await pipeline.generate(generationRequest) { update in
                    let localProgress: Double
                    switch update.stage {
                    case .loadingModel: localProgress = 0.02
                    case .encodingText: localProgress = 0.12
                    case .loadingTransformer: localProgress = 0.18
                    case .loadingLoRA: localProgress = 0.22
                    case .loadingVAE: localProgress = 0.24
                    case .denoising: localProgress = 0.25 + update.fractionCompleted * 0.60
                    case .decoding: localProgress = 0.90
                    case .saving: localProgress = 0.98
                    }
                    let aggregate = (Double(index) + localProgress) / Double(outputCount)
                    emit(
                        WorkerEvent(
                            type: "progress",
                            stage: String(describing: update.stage),
                            value: min(0.99, max(0, aggregate)),
                            message: nil
                        )
                    )
                }
            } catch let error as ZImagePipeline.PipelineError {
                throw WorkerError.pipelineFailure(Self.pipelineErrorMessage(error))
            }
            try Task.checkCancellation()
            guard FileManager.default.fileExists(atPath: outputURL.path) else {
                throw WorkerError.outputMissing(outputURL)
            }
            emit(
                WorkerEvent(
                    type: "progress",
                    stage: "completedImage",
                    value: Double(index + 1) / Double(outputCount),
                    message: nil
                )
            )
        }

        emit(WorkerEvent(type: "completed", stage: nil, value: 1, message: nil))
    }

    private static func requestURL(from arguments: [String]) throws -> URL {
        guard arguments.count == 3, arguments[1] == "--request" else {
            throw WorkerError.usage
        }
        return URL(fileURLWithPath: arguments[2])
    }

    private static func pipelineErrorMessage(_ error: ZImagePipeline.PipelineError) -> String {
        switch error {
        case .notImplemented:
            return "Z-Image Runtime 尚未實作此功能。"
        case .tokenizerNotLoaded:
            return "Z-Image tokenizer 尚未載入。"
        case let .invalidDimensions(message), let .invalidModelPath(message), let .weightsMissing(message):
            return message
        case .textEncoderNotLoaded:
            return "Z-Image 文字編碼器尚未載入。"
        case .transformerNotLoaded:
            return "Z-Image Transformer 尚未載入。"
        case .vaeNotLoaded:
            return "Z-Image VAE 尚未載入。"
        case .modelNotLoaded:
            return "Z-Image 模型尚未載入。"
        case let .loraError(error):
            return "LoRA 載入失敗：\(error.localizedDescription)"
        }
    }

    private static func emit(_ event: WorkerEvent, to handle: FileHandle = .standardOutput) {
        guard let data = try? JSONEncoder().encode(event) else { return }
        handle.write(data)
        handle.write(Data([0x0a]))
    }
}
