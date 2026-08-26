import Foundation
import GenImageCore
import WhisperKit

public actor WhisperSubtitleTranscriber: MediaTranscribing {
    public static let modelID = "argmaxinc/whisperkit-coreml@large-v3-turbo"

    private var loadedModelURL: URL?
    private var runtime: WhisperKit?

    public init() {}

    public nonisolated func supports(profile: InferenceProfile) -> Bool {
        profile.modelID == Self.modelID
    }

    public func transcribe(
        request: SubtitleGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> TranscriptResult {
        guard let sourceURL = request.sourceAsset.fileURL else {
            throw SubtitleTranscriptionError.sourceFileMissing
        }
        progress(0.02)
        let prepared = try await MediaAudioPreparer.prepare(sourceURL: sourceURL)
        defer { try? FileManager.default.removeItem(at: prepared.temporaryDirectory) }
        try Task.checkCancellation()
        progress(0.08)

        let whisperKit = try await loadRuntime(modelURL: request.modelURL)
        try Task.checkCancellation()
        progress(0.18)

        let requestedLanguage = request.profile.defaults.languageCode.flatMap { language in
            language.lowercased() == "auto" ? nil : language
        }
        var decodeOptions = DecodingOptions(
            task: .transcribe,
            language: requestedLanguage,
            usePrefillPrompt: requestedLanguage != nil,
            detectLanguage: requestedLanguage == nil,
            skipSpecialTokens: true,
            withoutTimestamps: false,
            wordTimestamps: false,
            chunkingStrategy: .vad
        )
        decodeOptions.verbose = false
        let results = try await whisperKit.transcribe(
            audioPath: prepared.audioURL.path,
            audioInputOptions: AudioInputOptions(
                channelMode: .sumChannels(nil),
                audioLoadingMode: .incremental(
                    chunkDurationSeconds: 120,
                    maxBufferedChunks: 2
                )
            ),
            decodeOptions: decodeOptions
        )
        try Task.checkCancellation()
        progress(0.94)

        var previousEnd = 0.0
        let segments = results
            .flatMap(\.segments)
            .filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .sorted { lhs, rhs in
                lhs.start == rhs.start ? lhs.end < rhs.end : lhs.start < rhs.start
            }
            .map { segment -> TimedTranscriptSegment in
                let start = max(previousEnd, Double(max(0, segment.start)))
                let end = max(start + 0.05, Double(segment.end))
                previousEnd = end
                return TimedTranscriptSegment(
                    start: start,
                    end: end,
                    text: segment.text.trimmingCharacters(in: .whitespacesAndNewlines),
                    confidence: exp(Double(segment.avgLogprob))
                )
            }
        guard !segments.isEmpty else {
            throw SubtitleTranscriptionError.emptyTranscript
        }
        let languageCode = results.first?.language ?? requestedLanguage ?? "und"
        return TranscriptResult(
            text: segments.map(\.text).joined(separator: "\n"),
            languageCode: languageCode,
            durationSeconds: prepared.durationSeconds,
            segments: segments
        )
    }

    public func unload() async {
        runtime = nil
        loadedModelURL = nil
    }

    private func loadRuntime(modelURL: URL) async throws -> WhisperKit {
        if let runtime, loadedModelURL == modelURL {
            return runtime
        }
        let config = WhisperKitConfig(
            model: modelURL.lastPathComponent,
            modelFolder: modelURL.path,
            verbose: false,
            prewarm: false,
            load: true,
            download: false
        )
        let loaded = try await WhisperKit(config)
        runtime = loaded
        loadedModelURL = modelURL
        return loaded
    }
}
