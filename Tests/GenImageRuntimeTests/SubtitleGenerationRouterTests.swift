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
    func generateText(
        request: TextGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> String {
        throw StubTextGeneratorError.unexpectedCall
    }

    func unload() async {}
}

private enum StubTextGeneratorError: Error {
    case unexpectedCall
}
