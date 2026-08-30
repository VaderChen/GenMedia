import Foundation
import GenImageCore
import WhisperKit

public actor WhisperSubtitleTranscriber: MediaTranscribing {
    public static let modelID = "argmaxinc/whisperkit-coreml@large-v3-turbo"
    public static let smallModelID = "argmaxinc/whisperkit-coreml@small"
    public static let defaultModelID = smallModelID
    public static let supportedModelIDs: Set<String> = [modelID, smallModelID]

    private var loadedModelURL: URL?
    private var runtime: WhisperKit?

    public init() {}

    public nonisolated func supports(profile: InferenceProfile) -> Bool {
        Self.supportedModelIDs.contains(profile.modelID)
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
        let progressReporter = WhisperTimelineProgressReporter(
            durationSeconds: prepared.durationSeconds,
            progress: progress
        )
        whisperKit.segmentDiscoveryCallback = { segments in
            progressReporter.report(segments)
        }
        defer { whisperKit.segmentDiscoveryCallback = nil }
        let results = try await withTaskCancellationHandler {
            try await whisperKit.transcribe(
                audioPath: prepared.audioURL.path,
                audioInputOptions: AudioInputOptions(
                    channelMode: .sumChannels(nil),
                    audioLoadingMode: .incremental(
                        chunkDurationSeconds: 120,
                        maxBufferedChunks: 2
                    )
                ),
                decodeOptions: decodeOptions,
                callback: { _ in progressReporter.shouldContinue }
            )
        } onCancel: {
            progressReporter.cancel()
        }
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

private final class WhisperTimelineProgressReporter: @unchecked Sendable {
    private let durationSeconds: Double
    private let progress: @Sendable (Double) -> Void
    private let lock = NSLock()
    private var furthestRecognizedTime = 0.0
    private var isCancelled = false

    init(
        durationSeconds: Double,
        progress: @escaping @Sendable (Double) -> Void
    ) {
        self.durationSeconds = durationSeconds
        self.progress = progress
    }

    var shouldContinue: Bool {
        lock.lock()
        defer { lock.unlock() }
        return !isCancelled
    }

    func cancel() {
        lock.lock()
        isCancelled = true
        lock.unlock()
    }

    func report(_ segments: [TranscriptionSegment]) {
        guard durationSeconds.isFinite, durationSeconds > 0,
              let recognizedTime = segments.map({ Double($0.end) }).max(),
              recognizedTime.isFinite else {
            return
        }

        let reportedProgress: Double?
        lock.lock()
        if isCancelled || recognizedTime <= furthestRecognizedTime {
            reportedProgress = nil
        } else {
            furthestRecognizedTime = recognizedTime
            let fraction = min(1, max(0, recognizedTime / durationSeconds))
            reportedProgress = min(0.92, 0.18 + fraction * 0.74)
        }
        lock.unlock()

        if let reportedProgress {
            progress(reportedProgress)
        }
    }
}
