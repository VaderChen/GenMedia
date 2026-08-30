import Foundation
import GenImageCore
import MLXLLM
import MLXLMCommon
import MLXVLM
import MLXHuggingFace
import Tokenizers

public actor QwenTextGenerationService: TextGenerating {
    private var container: ModelContainer?
    private var loadedModelURL: URL?

    public init() {}

    public func generateText(
        request: TextGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> String {
        guard request.profile.capability == .textToText else {
            throw QwenTextGenerationError.incompatibleProfile
        }
        guard request.profile.architecture == .mlxSwift else {
            throw QwenTextGenerationError.unsupportedArchitecture(request.profile.architecture)
        }
        let prompt = request.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            throw QwenTextGenerationError.emptyPrompt
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: request.modelURL.path,
            isDirectory: &isDirectory
        ), isDirectory.boolValue else {
            throw QwenTextGenerationError.modelNotFound(request.modelURL)
        }

        progress(0.01)
        let container = try await loadContainer(at: request.modelURL, progress: progress)
        try Task.checkCancellation()
        progress(0.48)

        let input = UserInput(
            chat: [.user(prompt)],
            additionalContext: [
                "enable_thinking": false,
                "reasoning_effort": "low"
            ]
        )
        let prepared = try await container.prepare(input: input)
        progress(0.55)

        let profileLimit = min(max(request.profile.defaults.maxTokens ?? 2_048, 128), 8_192)
        let maximumOutputTokens = request.maximumOutputTokens.map {
            min(max($0, 32), profileLimit)
        } ?? profileLimit
        let parameters = GenerateParameters(
            maxTokens: maximumOutputTokens,
            temperature: 0.2,
            topP: 0.9,
            repetitionPenalty: 1.08,
            repetitionContextSize: 128
        )
        let stream = try await container.generate(input: prepared, parameters: parameters)
        let expectedChunks = min(maximumOutputTokens, 512)
        var result = ""
        var chunks = 0
        var lastRepetitionCheck = 0
        var stopReason: GenerateStopReason?

        generationLoop: for await event in stream {
            try Task.checkCancellation()
            switch event {
            case let .chunk(text):
                result += text
                chunks += 1
                progress(min(0.99, 0.55 + Double(chunks) / Double(expectedChunks) * 0.44))
                if !request.expectsStructuredOutput,
                   chunks - lastRepetitionCheck >= 32,
                   result.count > 400 {
                    lastRepetitionCheck = chunks
                    if Self.hasRepeatingTail(result) { break generationLoop }
                }
            case let .info(info):
                stopReason = info.stopReason
            case .toolCall:
                continue
            }
        }

        if case .some(.length) = stopReason {
            throw QwenTextGenerationError.outputTruncated(maximumOutputTokens: maximumOutputTokens)
        }
        let output = TextGenerationOutputNormalizer.finalAnswer(result)
        guard !output.isEmpty else {
            throw QwenTextGenerationError.emptyResponse
        }
        progress(1)
        return output
    }

    public func unload() async {
        container = nil
        loadedModelURL = nil
    }

    private func loadContainer(
        at modelURL: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ModelContainer {
        let normalizedURL = modelURL.resolvingSymlinksInPath().standardizedFileURL
        if let container, loadedModelURL == normalizedURL {
            progress(0.45)
            return container
        }
        container = nil
        loadedModelURL = nil
        let loaded: ModelContainer
        switch Self.factoryKind(at: normalizedURL) {
        case .language:
            loaded = try await LLMModelFactory.shared.loadContainer(
                from: normalizedURL,
                using: #huggingFaceTokenizerLoader()
            )
        case .visionLanguage:
            loaded = try await VLMModelFactory.shared.loadContainer(
                from: normalizedURL,
                using: #huggingFaceTokenizerLoader()
            )
        }
        progress(0.45)
        container = loaded
        loadedModelURL = normalizedURL
        return loaded
    }

    private enum FactoryKind {
        case language
        case visionLanguage
    }

    private static func factoryKind(at modelURL: URL) -> FactoryKind {
        guard let data = try? Data(contentsOf: modelURL.appendingPathComponent("config.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .language
        }

        let multimodalKeys = [
            "vision_config", "vision_tower", "image_token_id", "video_token_id",
            "mm_projector_type", "image_processor_type"
        ]
        if multimodalKeys.contains(where: { object[$0] != nil }) {
            return .visionLanguage
        }
        if let languageModelOnly = object["language_model_only"] as? Bool,
           !languageModelOnly {
            return .visionLanguage
        }

        let modelType = (object["model_type"] as? String ?? "").lowercased()
        let architectures = (object["architectures"] as? [String] ?? [])
            .joined(separator: " ")
            .lowercased()
        let signature = "\(modelType) \(architectures)"
        let visionMarkers = ["vision", "_vl", "vl_", "llava", "paligemma", "pixtral", "idefics"]
        return visionMarkers.contains(where: signature.contains) ? .visionLanguage : .language
    }

    private static func hasRepeatingTail(_ output: String) -> Bool {
        let sampleLength = 90
        guard output.count > sampleLength * 3 else { return false }
        let needle = String(output.suffix(sampleLength))
        guard !needle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        var occurrences = 0
        var searchRange = output.startIndex..<output.endIndex
        while let found = output.range(of: needle, range: searchRange) {
            occurrences += 1
            if occurrences >= 3 { return true }
            guard found.upperBound < output.endIndex else { break }
            searchRange = found.upperBound..<output.endIndex
        }
        return false
    }
}

public enum QwenTextGenerationError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case emptyPrompt
    case modelNotFound(URL)
    case emptyResponse
    case outputTruncated(maximumOutputTokens: Int)

    public var errorDescription: String? {
        switch self {
        case .incompatibleProfile:
            "Profile 不是文生文類型。"
        case let .unsupportedArchitecture(architecture):
            "文生文 Runtime 不支援架構：\(architecture.title)。"
        case .emptyPrompt:
            "文生文 Prompt 不可為空。"
        case let .modelNotFound(url):
            "找不到文生文模型：\(url.path)"
        case .emptyResponse:
            "文生文模型沒有回傳內容。"
        case let .outputTruncated(maximumOutputTokens):
            "文生文模型輸出達到 \(maximumOutputTokens) token 上限，內容可能被截斷；請縮小字幕批次或改用 token 上限較大的模型。"
        }
    }
}
