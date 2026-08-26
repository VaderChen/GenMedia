import FluidAudio
import Foundation
import GenImageCore

public actor ParaformerChineseSubtitleTranscriber: MediaTranscribing {
    public static let modelID = "FluidInference/paraformer-large-zh-coreml"

    private static let audioFormat = MediaAudioPreparer.speechRecognitionFormat
    private static let chunkDurationSeconds = 28.0
    private static let overlapDurationSeconds = 1.0

    private var loadedModelURL: URL?
    private var manager: ParaformerManager?

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

        let converter = AudioConverter(sampleRate: Self.audioFormat.sampleRate)
        let audio = try converter.resampleAudioFile(prepared.audioURL)
        try Task.checkCancellation()
        progress(0.14)

        let manager = try await loadManager(modelURL: request.modelURL)
        try Task.checkCancellation()
        progress(0.18)

        let chunks = Self.audioChunks(audio)
        var tokens: [TimestampedSegment] = []
        tokens.reserveCapacity(max(32, Int(prepared.durationSeconds * 4)))

        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            let recognized = try await manager.transcribeWithTimestamps(audio: chunk.samples)
            let isFirst = index == chunks.startIndex
            let isLast = index == chunks.index(before: chunks.endIndex)
            let lowerBound = isFirst
                ? 0
                : chunk.startTime + Self.overlapDurationSeconds / 2
            let upperBound = isLast
                ? .greatestFiniteMagnitude
                : chunk.endTime - Self.overlapDurationSeconds / 2

            tokens.append(contentsOf: recognized.compactMap { segment in
                let start = segment.startTime + chunk.startTime
                let end = segment.endTime + chunk.startTime
                let midpoint = (start + end) / 2
                guard midpoint >= lowerBound, midpoint < upperBound else { return nil }
                return TimestampedSegment(
                    startTime: start,
                    endTime: end,
                    text: segment.text
                )
            })
            let completed = Double(index + 1) / Double(max(1, chunks.count))
            progress(0.18 + completed * 0.76)
        }

        let segments = ChineseSubtitleSegmenter.segments(tokens: tokens)
        guard !segments.isEmpty else {
            throw SubtitleTranscriptionError.emptyTranscript
        }
        return TranscriptResult(
            text: segments.map(\.text).joined(separator: "\n"),
            languageCode: "zh",
            durationSeconds: prepared.durationSeconds,
            segments: segments
        )
    }

    public func unload() async {
        manager = nil
        loadedModelURL = nil
    }

    private func loadManager(modelURL: URL) async throws -> ParaformerManager {
        if let manager, loadedModelURL == modelURL {
            return manager
        }
        let models = try ParaformerModels.load(from: modelURL, precision: .int8)
        let loaded = ParaformerManager(models: models)
        manager = loaded
        loadedModelURL = modelURL
        return loaded
    }

    private struct AudioChunk: Sendable {
        let samples: [Float]
        let startTime: Double
        let endTime: Double
    }

    private static func audioChunks(_ audio: [Float]) -> [AudioChunk] {
        guard !audio.isEmpty else { return [] }
        let chunkSamples = Int(chunkDurationSeconds * audioFormat.sampleRate)
        let overlapSamples = Int(overlapDurationSeconds * audioFormat.sampleRate)
        let stepSamples = max(1, chunkSamples - overlapSamples)
        var result: [AudioChunk] = []
        var start = 0

        while start < audio.count {
            let end = min(audio.count, start + chunkSamples)
            result.append(
                AudioChunk(
                    samples: Array(audio[start..<end]),
                    startTime: Double(start) / audioFormat.sampleRate,
                    endTime: Double(end) / audioFormat.sampleRate
                )
            )
            if end == audio.count { break }
            start += stepSamples
        }
        return result
    }
}

private enum ChineseSubtitleSegmenter {
    private static let maximumDuration = 6.0
    private static let maximumCharacters = 30
    private static let pauseThreshold = 0.7
    private static let sentenceEndings = CharacterSet(charactersIn: "。！？!?…；;")
    private static let punctuation = CharacterSet(charactersIn: "，。！？、：；,.!?;:%％)]}》〉」』】")

    static func segments(tokens: [TimestampedSegment]) -> [TimedTranscriptSegment] {
        let usable = tokens
            .filter { !$0.text.isEmpty && $0.endTime > $0.startTime }
            .sorted { lhs, rhs in
                lhs.startTime == rhs.startTime
                    ? lhs.endTime < rhs.endTime
                    : lhs.startTime < rhs.startTime
            }
        guard !usable.isEmpty else { return [] }

        var result: [TimedTranscriptSegment] = []
        var currentText = ""
        var currentStart = usable[0].startTime
        var currentEnd = usable[0].endTime

        func flush() {
            let text = currentText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return }
            result.append(
                TimedTranscriptSegment(
                    start: currentStart,
                    end: max(currentStart + 0.05, currentEnd),
                    text: text
                )
            )
            currentText = ""
        }

        for token in usable {
            let text = normalizedToken(token.text)
            guard !text.isEmpty else { continue }
            let gap = token.startTime - currentEnd
            let projectedLength = currentText.count + text.count
            let shouldStartNewSegment = !currentText.isEmpty && (
                gap >= pauseThreshold
                    || token.endTime - currentStart >= maximumDuration
                    || projectedLength > maximumCharacters
            )
            if shouldStartNewSegment {
                flush()
                currentStart = token.startTime
            }

            if shouldInsertSpace(before: text, existingText: currentText) {
                currentText += " "
            }
            currentText += text
            currentEnd = token.endTime

            if text.unicodeScalars.last.map(sentenceEndings.contains) == true {
                flush()
                currentStart = token.endTime
            }
        }
        flush()
        return result
    }

    private static func normalizedToken(_ text: String) -> String {
        text
            .replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func shouldInsertSpace(before token: String, existingText: String) -> Bool {
        guard let previous = existingText.unicodeScalars.last,
              let next = token.unicodeScalars.first else { return false }
        if punctuation.contains(next) { return false }
        return CharacterSet.alphanumerics.contains(previous)
            && CharacterSet.alphanumerics.contains(next)
            && previous.value < 128
            && next.value < 128
    }
}
