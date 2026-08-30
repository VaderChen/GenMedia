import Foundation
import GenImageCore
import Testing

@testable import GenImageRuntime

struct SubtitleGenerationRouterTests {
    @Test func selectsTheFirstAdapterThatSupportsTheProfile() async throws {
        let unsupported = StubTranscriber(supportedModelIDs: ["other-model"])
        let supported = StubTranscriber(supportedModelIDs: ["test-model"])
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let router = SubtitleGenerationRouter(
            outputDirectory: directory,
            adapters: [unsupported, supported],
            translator: StubTextGenerator()
        )

        let result = try await router.generate(request: request()) { _ in }

        #expect(await unsupported.transcriptionCount == 0)
        #expect(await supported.transcriptionCount == 1)
        #expect(result.asset.kind == .generatedSubtitle)
        #expect(result.asset.parentAssetID == request().sourceAsset.id)
        #expect(result.asset.fileURL?.pathExtension == "srt")
        #expect(result.asset.fileURL.map { FileManager.default.fileExists(atPath: $0.path) } == true)
    }

    @Test func throwsWhenNoAdapterSupportsTheProfile() async {
        let router = SubtitleGenerationRouter(
            outputDirectory: temporaryDirectory(),
            adapters: [StubTranscriber(supportedModelIDs: ["other-model"])],
            translator: StubTextGenerator()
        )

        do {
            _ = try await router.generate(request: request()) { _ in }
            Issue.record("預期 unsupportedProfile 錯誤")
        } catch let error as SubtitleTranscriptionError {
            guard case let .unsupportedProfile(name) = error else {
                Issue.record("收到非預期的字幕錯誤：\(error)")
                return
            }
            #expect(name == "測試字幕 Profile")
        } catch {
            Issue.record("收到非預期的錯誤型別：\(error)")
        }
    }

    @Test func multipleMatchingAdaptersUseOnlyTheFirstOne() async throws {
        let first = StubTranscriber(supportedModelIDs: ["test-model"])
        let second = StubTranscriber(supportedModelIDs: ["test-model"])
        let directory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let router = SubtitleGenerationRouter(
            outputDirectory: directory,
            adapters: [first, second],
            translator: StubTextGenerator()
        )

        _ = try await router.generate(request: request()) { _ in }

        #expect(await first.transcriptionCount == 1)
        #expect(await second.transcriptionCount == 0)
    }

    @Test func writesSubtitleBesideSourceWithMatchingBaseName() async throws {
        let outputDirectory = temporaryDirectory()
        let sourceDirectory = temporaryDirectory()
        defer {
            try? FileManager.default.removeItem(at: outputDirectory)
            try? FileManager.default.removeItem(at: sourceDirectory)
        }
        try FileManager.default.createDirectory(
            at: sourceDirectory,
            withIntermediateDirectories: true
        )
        let sourceURL = sourceDirectory.appendingPathComponent("source.mp4")
        try Data().write(to: sourceURL)

        let router = SubtitleGenerationRouter(
            outputDirectory: outputDirectory,
            adapters: [StubTranscriber(supportedModelIDs: ["test-model"])],
            translator: StubTextGenerator()
        )
        var subtitleRequest = request()
        subtitleRequest.sourceAsset.fileURL = sourceURL

        let result = try await router.generate(request: subtitleRequest) { _ in }
        let outputURL = try #require(result.asset.fileURL)

        #expect(outputURL == sourceDirectory.appendingPathComponent("source.srt"))
        #expect(FileManager.default.fileExists(atPath: outputURL.path))
        #expect(!FileManager.default.fileExists(atPath: outputDirectory.path))
    }

    @Test func translationBatchesUseTheReducedStructuredOutputLimits() {
        let segments = (0..<13).map { index in
            TimedTranscriptSegment(start: Double(index), end: Double(index + 1), text: "字幕 \(index)")
        }

        let batches = SubtitleGenerationRouter.translationBatches(segments)

        #expect(batches.count == 2)
        #expect(batches[0].count == 12)
        #expect(batches[1].count == 1)
        #expect(batches.allSatisfy { $0.count <= SubtitleGenerationRouter.maximumTranslationBatchItems })
        #expect(batches.allSatisfy {
            $0.reduce(0) { $0 + $1.text.count }
                <= SubtitleGenerationRouter.maximumTranslationBatchCharacters
        })
    }

    @Test func translationKeepsOriginalRecognitionAboveTranslatedText() async throws {
        let outputDirectory = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: outputDirectory) }
        let router = SubtitleGenerationRouter(
            outputDirectory: outputDirectory,
            adapters: [StubTranscriber(supportedModelIDs: ["test-model"])],
            translator: StubTextGenerator(
                output: "[{\"index\":0,\"text\":\"Translated subtitle\"}]"
            )
        )
        var translationRequest = request()
        translationRequest.translation = SubtitleTranslationConfiguration(
            targetLanguage: .english,
            profile: InferenceProfile(
                name: "測試翻譯 Profile",
                capability: .textToText,
                modelID: "translation-model",
                architecture: .coreML
            ),
            modelURL: URL(fileURLWithPath: "/tmp/translation-model", isDirectory: true)
        )

        let result = try await router.generate(request: translationRequest) { _ in }
        let segment = try #require(result.transcript.segments.first)
        let subtitleURL = try #require(result.asset.fileURL)
        let document = try String(contentsOf: subtitleURL, encoding: .utf8)
        let originalSubtitleURL = subtitleURL
            .deletingPathExtension()
            .appendingPathExtension("org")
            .appendingPathExtension("srt")
        let originalDocument = try String(
            contentsOf: originalSubtitleURL,
            encoding: .utf8
        )

        #expect(segment.text == "測試字幕\nTranslated subtitle")
        #expect(result.transcript.languageCode == "en")
        #expect(document.contains("00:00:00,000 --> 00:00:01,250\n測試字幕\nTranslated subtitle"))
        #expect(originalDocument.contains("00:00:00,000 --> 00:00:01,250\n測試字幕"))
        #expect(!originalDocument.contains("Translated subtitle"))
    }

    private func temporaryDirectory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent(
            "genimage-subtitle-router-test-\(UUID().uuidString)",
            isDirectory: true
        )
    }

    private func request() -> SubtitleGenerationRequest {
        let projectID = UUID(uuidString: "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")!
        let asset = MediaAsset(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            projectID: projectID,
            kind: .importedVideo,
            title: "source",
            fileURL: URL(fileURLWithPath: "/tmp/source.mp4"),
            pixelWidth: 1_920,
            pixelHeight: 1_080
        )
        let profile = InferenceProfile(
            name: "測試字幕 Profile",
            capability: .videoToText,
            modelID: "test-model",
            architecture: .coreML
        )
        return SubtitleGenerationRequest(
            projectID: projectID,
            sourceAsset: asset,
            profile: profile,
            modelURL: URL(fileURLWithPath: "/tmp/test-model", isDirectory: true),
            format: .srt
        )
    }
}

private actor StubTranscriber: MediaTranscribing {
    private nonisolated let supportedModelIDs: Set<String>
    private(set) var transcriptionCount = 0

    init(supportedModelIDs: Set<String>) {
        self.supportedModelIDs = supportedModelIDs
    }

    nonisolated func supports(profile: InferenceProfile) -> Bool {
        supportedModelIDs.contains(profile.modelID)
    }

    func transcribe(
        request: SubtitleGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> TranscriptResult {
        transcriptionCount += 1
        progress(1)
        let segments = [
            TimedTranscriptSegment(start: 0, end: 1.25, text: "測試字幕")
        ]
        return TranscriptResult(
            text: "測試字幕",
            languageCode: "zh",
            durationSeconds: 1.25,
            segments: segments
        )
    }

    func unload() async {}
}

private actor StubTextGenerator: TextGenerating {
    private let output: String?

    init(output: String? = nil) {
        self.output = output
    }

    func generateText(
        request: TextGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> String {
        guard let output else {
            throw StubTextGeneratorError.unexpectedCall
        }
        progress(1)
        return output
    }

    func unload() async {}
}

private enum StubTextGeneratorError: Error {
    case unexpectedCall
}
