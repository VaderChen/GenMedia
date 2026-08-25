import ACEStepSwiftRuntime
import Foundation
import GenImageCore

public final class ACEStepMusicGenerationService: MusicRuntimeAdapter, Sendable {
    private static let minimumDurationSeconds = 10
    private let outputDirectory: URL

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public func supports(_ request: MusicGenerationRequest) -> Bool {
        request.profile.capability == .textToMusic
            && request.profile.architecture == .mlxSwift
            && request.profile.modelID.caseInsensitiveCompare(
                HuggingFaceModelInstaller.aceStep15TurboModelID
            ) == .orderedSame
    }

    public func generate(
        request: MusicGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> MediaAsset {
        progress(0.001)
        try request.options.validate()
        guard supports(request) else {
            throw ACEStepRuntimeError.incompatibleProfile(
                modelID: request.profile.modelID,
                architecture: request.profile.architecture
            )
        }
        guard request.options.durationSeconds >= Self.minimumDurationSeconds else {
            throw ACEStepRuntimeError.invalidDuration(request.options.durationSeconds)
        }
        guard (1...20).contains(request.options.steps) else {
            throw ACEStepRuntimeError.invalidSteps(request.options.steps)
        }
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let identifier = UUID().uuidString
        let waveURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("ace-step-\(identifier).wav")
        let outputURL = OutputFileNaming.musicURL(
            in: outputDirectory,
            pathExtension: request.options.format.fileExtension
        )
        var completed = false
        defer {
            try? FileManager.default.removeItem(at: waveURL)
            if !completed {
                try? FileManager.default.removeItem(at: outputURL)
            }
        }

        _ = try ACEStepNativeGenerator.generate(
            modelRoot: request.modelURL,
            prompt: request.options.prompt,
            lyrics: request.options.lyrics,
            durationSeconds: request.options.durationSeconds,
            inferenceSteps: request.options.steps,
            seed: request.options.seed,
            outputURL: waveURL,
            progress: { progress(0.01 + 0.89 * $0) }
        )
        guard RuntimeLog.fileSize(at: waveURL) > 0 else {
            throw ACEStepRuntimeError.waveOutputMissing(waveURL)
        }
        try Task.checkCancellation()
        progress(0.92)
        let metadata = try await AudioOutputEncoder.encodeWave(
            inputURL: waveURL,
            outputURL: outputURL,
            format: request.options.format
        )

        progress(1)
        completed = true
        return MediaAsset(
            projectID: request.projectID,
            kind: .generatedAudio,
            title: "生成音樂",
            fileURL: outputURL,
            pixelWidth: 0,
            pixelHeight: 0,
            mediaDurationSeconds: metadata.durationSeconds,
            sampleRate: metadata.sampleRate,
            channelCount: metadata.channelCount,
            audioFormat: request.options.format,
            recipeID: request.recipeID
        )
    }
}

public enum ACEStepRuntimeError: LocalizedError, Sendable {
    case incompatibleProfile(modelID: String, architecture: InferenceArchitecture)
    case invalidDuration(Int)
    case invalidSteps(Int)
    case waveOutputMissing(URL)

    public var errorDescription: String? {
        switch self {
        case let .incompatibleProfile(modelID, architecture):
            "ACE-Step 原生 Runtime 不支援模型「\(modelID)」與架構「\(architecture.title)」。"
        case let .invalidDuration(seconds):
            "ACE-Step 音樂長度必須介於 10 到 300 秒，目前為 \(seconds) 秒。"
        case let .invalidSteps(steps):
            "ACE-Step Turbo 推論步數必須介於 1 到 20，目前為 \(steps)。"
        case let .waveOutputMissing(url):
            "ACE-Step 完成但未產生暫存 WAV 音訊：\(url.path)"
        }
    }
}
