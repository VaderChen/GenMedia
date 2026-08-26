import FluidAudio
import Foundation
import GenImageCore

public actor ParakeetJapaneseSubtitleTranscriber: MediaTranscribing {
    public static let modelID = "FluidInference/parakeet-0.6b-ja-coreml"

    private var loadedModelURL: URL?
    private var manager: AsrManager?

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

        let manager = try await loadManager(modelURL: request.modelURL)
        let decoderLayers = await manager.decoderLayerCount
        var decoderState = try TdtDecoderState(decoderLayers: decoderLayers)
        let progressStream = await manager.transcriptionProgressStream
        let progressTask = Task {
            do {
                for try await value in progressStream {
                    progress(0.18 + min(1, max(0, value)) * 0.76)
                }
            } catch is CancellationError {
                return
            } catch {
                return
            }
        }
        defer { progressTask.cancel() }

        progress(0.18)
        let result = try await manager.transcribe(
            prepared.audioURL,
            decoderState: &decoderState
        )
        try Task.checkCancellation()
        progress(0.94)

        let segments = JapaneseSubtitleSegmenter.segments(
            timings: result.tokenTimings ?? [],
            fallbackText: result.text,
            durationSeconds: prepared.durationSeconds
        )
        guard !segments.isEmpty else {
            throw SubtitleTranscriptionError.emptyTranscript
        }
        return TranscriptResult(
            text: result.text.trimmingCharacters(in: .whitespacesAndNewlines),
            languageCode: "ja",
            durationSeconds: prepared.durationSeconds,
            segments: segments
        )
    }

    public func unload() async {
        manager = nil
        loadedModelURL = nil
    }

    private func loadManager(modelURL: URL) async throws -> AsrManager {
        if let manager, loadedModelURL == modelURL {
            return manager
        }
        let models = try await AsrModels.load(
            from: modelURL,
            version: .tdtJa
        )
        let loaded = AsrManager(models: models)
        manager = loaded
        loadedModelURL = modelURL
        return loaded
    }
}

private enum JapaneseSubtitleSegmenter {
    private static let maximumDuration = 6.0
    private static let maximumCharacters = 42
    private static let pauseThreshold = 0.75
    private static let sentenceEndings = CharacterSet(charactersIn: "。！？!?…")

    static func segments(
        timings: [TokenTiming],
        fallbackText: String,
        durationSeconds: Double
    ) -> [TimedTranscriptSegment] {
        let usable = timings
            .filter { timing in
                !timing.token.isEmpty
                    && timing.token != "<blank>"
                    && timing.token != "<pad>"
                    && timing.endTime > timing.startTime
            }
            .sorted { lhs, rhs in
                lhs.startTime == rhs.startTime
                    ? lhs.endTime < rhs.endTime
                    : lhs.startTime < rhs.startTime
            }
        guard !usable.isEmpty else {
            let text = fallbackText.trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? [] : [TimedTranscriptSegment(
                start: 0,
                end: max(0.05, durationSeconds),
                text: text
            )]
        }

        var result: [TimedTranscriptSegment] = []
        var currentText = ""
        var currentStart = usable[0].startTime
        var currentEnd = usable[0].endTime
        var confidences: [Double] = []

        func flush() {
            let text = normalized(currentText)
            guard !text.isEmpty else { return }
            result.append(
                TimedTranscriptSegment(
                    start: currentStart,
                    end: max(currentStart + 0.05, currentEnd),
                    text: text,
                    confidence: confidences.isEmpty
                        ? nil
                        : confidences.reduce(0, +) / Double(confidences.count)
                )
            )
            currentText = ""
            confidences.removeAll(keepingCapacity: true)
        }

        for timing in usable {
            let token = timing.token.replacingOccurrences(of: "▁", with: " ")
            let gap = timing.startTime - currentEnd
            let projectedText = normalized(currentText + token)
            let shouldStartNewSegment = !currentText.isEmpty && (
                gap >= pauseThreshold
                    || timing.endTime - currentStart >= maximumDuration
                    || projectedText.count > maximumCharacters
            )
            if shouldStartNewSegment {
                flush()
                currentStart = timing.startTime
            }
            currentText += token
            currentEnd = timing.endTime
            confidences.append(Double(timing.confidence))

            if token.unicodeScalars.last.map(sentenceEndings.contains) == true {
                flush()
                currentStart = timing.endTime
            }
        }
        flush()
        return result
    }

    private static func normalized(_ text: String) -> String {
        text
            .replacingOccurrences(of: "  ", with: " ")
            .replacingOccurrences(of: " 。", with: "。")
            .replacingOccurrences(of: " 、", with: "、")
            .replacingOccurrences(of: " ！", with: "！")
            .replacingOccurrences(of: " ？", with: "？")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
