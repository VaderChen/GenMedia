import Foundation
import GenImageCore
import ImageIO
import MLXLMCommon
import MLXVLM
import MLXHuggingFace
import Tokenizers
import UniformTypeIdentifiers

public actor QwenVLImageDescriptionService: ImageDescribing {
    private var container: ModelContainer?
    private var loadedModelPath: String?

    public init() {}

    /// 釋放目前常駐的 Qwen-VL 容器，供記憶體壓力保護使用。
    public func unload() {
        container = nil
        loadedModelPath = nil
    }

    public func describe(
        request: ImageDescriptionRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> String {
        guard request.profile.capability == .imageToText else {
            throw QwenVLRuntimeError.incompatibleProfile
        }
        guard request.profile.architecture == .mlxSwift else {
            throw QwenVLRuntimeError.unsupportedArchitecture(request.profile.architecture)
        }
        guard let imageURL = request.asset.fileURL,
              FileManager.default.fileExists(atPath: imageURL.path) else {
            throw QwenVLRuntimeError.missingInputFile
        }

        let modelURL = URL(fileURLWithPath: request.profile.modelID, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw QwenVLRuntimeError.modelNotFound(modelURL)
        }

        let preparedImage = try Self.preparedImageURL(from: imageURL)
        defer {
            if preparedImage.isTemporary {
                try? FileManager.default.removeItem(at: preparedImage.url)
            }
        }

        progress(0.02)
        let modelContainer = try await loadContainer(at: modelURL, progress: progress)
        try Task.checkCancellation()
        progress(0.48)

        let prompt = Self.prompt(for: request.languageCode)
        let maxTokens = max(32, min(request.profile.defaults.maxTokens ?? 512, 2_048))
        let attempts = [
            GenerateParameters(
                maxTokens: maxTokens,
                temperature: 0.2,
                topP: 0.9,
                repetitionPenalty: 1.1,
                repetitionContextSize: 64
            ),
            GenerateParameters(
                maxTokens: maxTokens,
                temperature: 0.6,
                topP: 0.85,
                repetitionPenalty: 1.18,
                repetitionContextSize: 128
            )
        ]

        var receivedText = false
        var generationProgressBase = 0.55
        let expectedGenerationChunks = min(maxTokens, 192)
        for (attemptIndex, parameters) in attempts.enumerated() {
            let input = UserInput(chat: [
                .user(prompt, images: [.url(preparedImage.url)])
            ])
            let prepared = try await modelContainer.prepare(input: input)
            progress(generationProgressBase)
            let progressEnd = attemptIndex == 0 ? 0.95 : 0.99
            let generation = try await generateDescription(
                modelContainer: modelContainer,
                input: prepared,
                parameters: parameters,
                expectedChunks: expectedGenerationChunks,
                progressBase: generationProgressBase,
                progressSpan: progressEnd - generationProgressBase,
                progress: progress
            )
            let description = generation.text
            let completedFraction = min(
                1,
                Double(generation.chunks) / Double(expectedGenerationChunks)
            )
            generationProgressBase += (progressEnd - generationProgressBase) * completedFraction
            receivedText = receivedText || !description.isEmpty
            if Self.isUsableDescription(description) {
                progress(1)
                return description
            }
        }

        throw receivedText ? QwenVLRuntimeError.degenerateResponse : QwenVLRuntimeError.emptyResponse
    }

    private static func preparedImageURL(
        from sourceURL: URL,
        maximumPixelSize: Int = 768
    ) throws -> (url: URL, isTemporary: Bool) {
        guard let imageSource = CGImageSourceCreateWithURL(sourceURL as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil)
                as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let height = properties[kCGImagePropertyPixelHeight] as? NSNumber else {
            throw QwenVLRuntimeError.imageResizeFailed(sourceURL)
        }

        guard max(width.intValue, height.intValue) > maximumPixelSize else {
            return (sourceURL, false)
        }

        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumPixelSize,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(
            imageSource,
            0,
            thumbnailOptions as CFDictionary
        ) else {
            throw QwenVLRuntimeError.imageResizeFailed(sourceURL)
        }

        let temporaryURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("GenImage-caption-\(UUID().uuidString)")
            .appendingPathExtension("png")
        guard let destination = CGImageDestinationCreateWithURL(
            temporaryURL as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw QwenVLRuntimeError.imageResizeFailed(sourceURL)
        }

        CGImageDestinationAddImage(destination, thumbnail, nil)
        guard CGImageDestinationFinalize(destination) else {
            try? FileManager.default.removeItem(at: temporaryURL)
            throw QwenVLRuntimeError.imageResizeFailed(sourceURL)
        }

        return (temporaryURL, true)
    }

    private struct DescriptionGenerationResult {
        var text: String
        var chunks: Int
    }

    private func generateDescription(
        modelContainer: ModelContainer,
        input: sending LMInput,
        parameters: GenerateParameters,
        expectedChunks: Int,
        progressBase: Double,
        progressSpan: Double,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> DescriptionGenerationResult {
        let stream = try await modelContainer.generate(input: input, parameters: parameters)
        var result = ""
        var chunks = 0

        for await event in stream {
            try Task.checkCancellation()
            if case let .chunk(text) = event {
                result += text
                chunks += 1
                let generationProgress = min(
                    progressBase + progressSpan,
                    progressBase + Double(chunks) / Double(expectedChunks) * progressSpan
                )
                progress(generationProgress)
                if Self.isClearlyDegenerate(result) { break }
            }
        }

        return DescriptionGenerationResult(
            text: result.trimmingCharacters(in: .whitespacesAndNewlines),
            chunks: chunks
        )
    }

    private static func isUsableDescription(_ text: String) -> Bool {
        let scalars = Array(text.unicodeScalars.filter {
            !CharacterSet.whitespacesAndNewlines.contains($0)
        })
        guard scalars.count >= 16, !isClearlyDegenerate(text) else { return false }

        let meaningfulCount = scalars.filter {
            CharacterSet.letters.contains($0) || CharacterSet.decimalDigits.contains($0)
        }.count
        return meaningfulCount >= 8 && Double(meaningfulCount) / Double(scalars.count) >= 0.3
    }

    private static func isClearlyDegenerate(_ text: String) -> Bool {
        let scalars = Array(text.unicodeScalars.filter {
            !CharacterSet.whitespacesAndNewlines.contains($0)
        })
        guard scalars.count >= 16 else { return false }

        var longestRun = 1
        var currentRun = 1
        for index in 1..<scalars.count {
            if scalars[index] == scalars[index - 1] {
                currentRun += 1
                longestRun = max(longestRun, currentRun)
            } else {
                currentRun = 1
            }
        }
        if longestRun >= 10 { return true }

        let punctuationAndSymbols = CharacterSet.punctuationCharacters
            .union(.symbols)
        let punctuationCount = scalars.filter { punctuationAndSymbols.contains($0) }.count
        return scalars.count >= 40
            && Double(punctuationCount) / Double(scalars.count) > 0.5
    }

    private func loadContainer(
        at modelURL: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ModelContainer {
        if let container, loadedModelPath == modelURL.standardizedFileURL.path {
            return container
        }

        container = nil
        loadedModelPath = nil
        let loaded = try await VLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: #huggingFaceTokenizerLoader()
        )
        progress(0.45)
        container = loaded
        loadedModelPath = modelURL.standardizedFileURL.path
        return loaded
    }

    private static func prompt(for languageCode: String) -> String {
        switch languageCode.lowercased() {
        case let code where code.hasPrefix("en"):
            "Describe this image accurately and in detail. Mention the main subject, environment, composition, lighting, colors, and style. Include scene text only when it is clearly legible and naturally belongs to the photographed environment; omit uncertain text. Ignore watermarks, creator signatures, stock-photo marks, and website or brand overlays. Do not repeat characters, punctuation, phrases, or sentences. Return only one coherent, reusable image-generation prompt."
        case let code where code.hasPrefix("ja"):
            "この画像を正確かつ詳しく説明してください。主題、環境、構図、照明、色、スタイルを含めてください。シーン内の文字は明確に読めて実際の環境に属する場合だけ記述し、不明瞭な文字は省略してください。透かし、作者署名、ストックフォト表示、Web サイトやブランドのオーバーレイは無視してください。文字、句読点、語句、文を繰り返さず、画像生成に再利用できる一つの自然な日本語プロンプトだけを返してください。"
        case let code where code.hasPrefix("ko"):
            "이 이미지를 정확하고 자세하게 설명하세요. 주요 대상, 환경, 구도, 조명, 색상과 스타일을 포함하세요. 장면의 문자는 명확히 읽히고 실제 환경에 속할 때만 설명하며 불확실한 문자는 생략하세요. 워터마크, 작가 서명, 스톡 사진 표시, 웹사이트 또는 브랜드 오버레이는 무시하세요. 문자, 문장부호, 구절 또는 문장을 반복하지 말고 이미지 생성에 재사용할 수 있는 하나의 자연스러운 한국어 프롬프트만 반환하세요."
        default:
            "請以繁體中文準確且詳細描述這張圖片，包含主體、環境、構圖、光線、色彩與風格。場景文字只有在清晰可辨且確實屬於實景時才描述，無法確定的文字請省略。忽略浮水印、作者簽名、圖庫標記、網站或品牌疊加標誌。不得連續重複字元、標點、詞語或句子，只輸出一段通順且可直接用於文生圖的 Prompt。"
        }
    }
}

public enum QwenVLRuntimeError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case missingInputFile
    case modelNotFound(URL)
    case imageResizeFailed(URL)
    case emptyResponse
    case degenerateResponse

    public var errorDescription: String? {
        switch self {
        case .incompatibleProfile:
            "Profile 不是圖生文類型。"
        case let .unsupportedArchitecture(architecture):
            "Qwen3-VL Runtime 不支援此架構：\(architecture.title)。"
        case .missingInputFile:
            "圖片資產沒有可讀取的本機檔案。"
        case let .modelNotFound(url):
            "找不到 Qwen3-VL 模型：\(url.path)"
        case let .imageResizeFailed(url):
            "無法將圖生文來源圖片縮放至 768 px：\(url.lastPathComponent)"
        case .emptyResponse:
            "Qwen3-VL 沒有回傳圖片描述。"
        case .degenerateResponse:
            "Qwen3-VL 連續兩次產生重複或無意義內容，已保留原本的 Prompt，請重新執行圖生文。"
        }
    }
}
