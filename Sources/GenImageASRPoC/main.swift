@preconcurrency import AVFoundation
import Foundation
import GenImageCore
import WhisperKit

private enum ChineseScript: String {
    case asr
    case traditional
    case simplified

    func transform(_ text: String) -> String {
        guard self != .asr else { return text }
        let mutableText = NSMutableString(string: text)
        let transformName = self == .traditional ? "Hans-Hant" : "Hant-Hans"
        _ = CFStringTransform(mutableText, nil, transformName as CFString, false)
        return mutableText as String
    }
}

private struct Arguments {
    var inputs: [URL] = []
    var model = "large-v3-v20240930_turbo"
    var modelRepo = "argmaxinc/whisperkit-coreml"
    var modelFolder: URL?
    var modelCacheDirectory = ApplicationSupport.directory(.models)
        .appendingPathComponent("WhisperKit", isDirectory: true)
    var outputDirectory = ApplicationSupport.directory(.generated)
        .appendingPathComponent("ASR", isDirectory: true)
    var language: String?
    var chineseScript: ChineseScript = .asr
    var wordTimestamps = false
    var downloadModels = true
    var keepPreparedAudio = false
    var showHelp = false

    static func parse(_ values: [String]) throws -> Arguments {
        var result = Arguments()
        var index = 0

        while index < values.count {
            let value = values[index]
            switch value {
            case "--input":
                index += 1
                result.inputs.append(try requiredURL(values, index: index, option: value))
            case "--model":
                index += 1
                result.model = try requiredValue(values, index: index, option: value)
            case "--model-repo":
                index += 1
                result.modelRepo = try requiredValue(values, index: index, option: value)
            case "--model-folder":
                index += 1
                result.modelFolder = try requiredURL(values, index: index, option: value)
                result.downloadModels = false
            case "--model-cache":
                index += 1
                result.modelCacheDirectory = try requiredURL(values, index: index, option: value)
            case "--output-dir":
                index += 1
                result.outputDirectory = try requiredURL(values, index: index, option: value)
            case "--language":
                index += 1
                let language = try requiredValue(values, index: index, option: value)
                result.language = language.lowercased() == "auto" ? nil : language
            case "--chinese-script":
                index += 1
                let script = try requiredValue(values, index: index, option: value)
                guard let parsedScript = ChineseScript(rawValue: script.lowercased()) else {
                    throw PoCError.invalidArguments("--chinese-script 可用值：asr、traditional、simplified。")
                }
                result.chineseScript = parsedScript
            case "--word-timestamps":
                result.wordTimestamps = true
            case "--no-download":
                result.downloadModels = false
            case "--keep-prepared-audio":
                result.keepPreparedAudio = true
            case "-h", "--help":
                result.showHelp = true
            default:
                if value.hasPrefix("-") {
                    throw PoCError.invalidArguments("不支援的參數：\(value)")
                }
                result.inputs.append(Self.expandedURL(value))
            }
            index += 1
        }

        if !result.showHelp, result.inputs.isEmpty {
            throw PoCError.invalidArguments("至少需要一個 --input 媒體檔案或位置參數。")
        }
        return result
    }

    private static func requiredValue(
        _ values: [String],
        index: Int,
        option: String
    ) throws -> String {
        guard values.indices.contains(index), !values[index].isEmpty else {
            throw PoCError.invalidArguments("\(option) 後方需要值。")
        }
        return values[index]
    }

    private static func requiredURL(
        _ values: [String],
        index: Int,
        option: String
    ) throws -> URL {
        Self.expandedURL(try requiredValue(values, index: index, option: option))
    }

    private static func expandedURL(_ value: String) -> URL {
        URL(fileURLWithPath: (value as NSString).expandingTildeInPath)
            .standardizedFileURL
    }
}

private enum PoCError: LocalizedError {
    case invalidArguments(String)
    case inputMissing(URL)
    case inputUnreadable(URL, String)
    case noAudioTrack(URL)
    case audioExportFailed(URL, String)
    case modelFolderMissing(URL)
    case outputFailed(URL, String)

    var errorDescription: String? {
        switch self {
        case let .invalidArguments(message):
            message
        case let .inputMissing(url):
            "找不到輸入檔案：\(url.path)"
        case let .inputUnreadable(url, reason):
            "無法讀取媒體「\(url.lastPathComponent)」：\(reason)"
        case let .noAudioTrack(url):
            "媒體沒有可辨識的音訊軌：\(url.lastPathComponent)"
        case let .audioExportFailed(url, reason):
            "無法從「\(url.lastPathComponent)」準備 Whisper 音訊：\(reason)"
        case let .modelFolderMissing(url):
            "找不到指定的 Whisper 模型目錄：\(url.path)"
        case let .outputFailed(url, reason):
            "無法寫入輸出檔案「\(url.path)」：\(reason)"
        }
    }
}

private struct PreparedMedia {
    let sourceURL: URL
    let audioURL: URL
    let durationSeconds: Double
}

private final class ExportSessionHandle: @unchecked Sendable {
    let session: AVAssetExportSession

    init(_ session: AVAssetExportSession) {
        self.session = session
    }
}

private enum MediaAudioPreparer {
    static func prepare(
        sourceURL: URL,
        temporaryDirectory: URL
    ) async throws -> PreparedMedia {
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw PoCError.inputMissing(sourceURL)
        }

        let asset = AVURLAsset(url: sourceURL)
        let audioTracks: [AVAssetTrack]
        let duration: CMTime
        do {
            audioTracks = try await asset.loadTracks(withMediaType: .audio)
            duration = try await asset.load(.duration)
        } catch {
            throw PoCError.inputUnreadable(sourceURL, error.localizedDescription)
        }
        guard !audioTracks.isEmpty else {
            throw PoCError.noAudioTrack(sourceURL)
        }

        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
        let outputURL = temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: false)
            .appendingPathExtension("m4a")

        guard let exporter = AVAssetExportSession(
            asset: asset,
            presetName: AVAssetExportPresetAppleM4A
        ) else {
            throw PoCError.audioExportFailed(sourceURL, "系統不支援 Apple M4A 音訊輸出。")
        }
        let exportSession = ExportSessionHandle(exporter)
        exportSession.session.outputURL = outputURL
        exportSession.session.outputFileType = .m4a
        exportSession.session.shouldOptimizeForNetworkUse = false

        do {
            try await withCheckedThrowingContinuation { continuation in
                exportSession.session.exportAsynchronously {
                    switch exportSession.session.status {
                    case .completed:
                        continuation.resume()
                    case .cancelled:
                        continuation.resume(throwing: CancellationError())
                    case .failed, .waiting, .exporting, .unknown:
                        continuation.resume(
                            throwing: exportSession.session.error
                                ?? PoCError.audioExportFailed(
                                    sourceURL,
                                    "ExportSession 狀態：\(exportSession.session.status.rawValue)"
                                )
                        )
                    @unknown default:
                        continuation.resume(
                            throwing: PoCError.audioExportFailed(
                                sourceURL,
                                "ExportSession 回傳未知狀態。"
                            )
                        )
                    }
                }
            }
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw PoCError.audioExportFailed(sourceURL, error.localizedDescription)
        }

        let durationSeconds = duration.isNumeric
            ? max(0, duration.seconds)
            : 0
        return PreparedMedia(
            sourceURL: sourceURL,
            audioURL: outputURL,
            durationSeconds: durationSeconds
        )
    }
}

private struct ASRWord: Codable, Sendable {
    let text: String
    let start: Double
    let end: Double
    let probability: Double
}

private struct ASRSegment: Codable, Sendable {
    let index: Int
    let start: Double
    let end: Double
    let text: String
    let averageLogProbability: Double
    let noSpeechProbability: Double
    let words: [ASRWord]?
}

private struct ASRReport: Codable, Sendable {
    let sourceFile: String
    let durationSeconds: Double
    let model: String
    let language: String
    let text: String
    let segments: [ASRSegment]
    let generatedAt: Date
}

private enum SubtitleWriter {
    static func srt(_ segments: [ASRSegment]) -> String {
        segments.enumerated().map { index, segment in
            "\(index + 1)\n"
                + "\(timestamp(segment.start, separator: ",")) --> \(timestamp(segment.end, separator: ","))\n"
                + "\(segment.text)\n"
        }.joined(separator: "\n")
    }

    static func vtt(_ segments: [ASRSegment]) -> String {
        "WEBVTT\n\n" + segments.map { segment in
            "\(timestamp(segment.start, separator: ".")) --> "
                + "\(timestamp(segment.end, separator: "."))\n"
                + "\(segment.text)\n"
        }.joined(separator: "\n")
    }

    private static func timestamp(
        _ value: Double,
        separator: String,
        includeHours: Bool = true
    ) -> String {
        let milliseconds = max(0, Int((value * 1_000).rounded()))
        let hours = milliseconds / 3_600_000
        let minutes = (milliseconds / 60_000) % 60
        let seconds = (milliseconds / 1_000) % 60
        let remainder = milliseconds % 1_000
        if includeHours {
            return String(format: "%02d:%02d:%02d%@%03d", hours, minutes, seconds, separator, remainder)
        }
        return String(format: "%02d:%02d%@%03d", minutes + hours * 60, seconds, separator, remainder)
    }
}

private enum OutputFiles {
    static func write(
        report: ASRReport,
        segments: [ASRSegment],
        baseName: String,
        directory: URL
    ) throws -> [URL] {
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        let stem = uniqueStem(baseName, in: directory)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let jsonURL = directory.appendingPathComponent("\(stem).json")
        let jsonData = try encoder.encode(report)
        try writeAtomically(jsonData, to: jsonURL)

        let srtURL = directory.appendingPathComponent("\(stem).srt")
        try writeAtomically(Data(SubtitleWriter.srt(segments).utf8), to: srtURL)

        let vttURL = directory.appendingPathComponent("\(stem).vtt")
        try writeAtomically(Data(SubtitleWriter.vtt(segments).utf8), to: vttURL)
        return [jsonURL, srtURL, vttURL]
    }

    private static func uniqueStem(_ baseName: String, in directory: URL) -> String {
        let sanitized = baseName
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
        var candidate = "\(sanitized)-asr"
        var suffix = 2
        while FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("\(candidate).json").path
        ) {
            candidate = "\(sanitized)-asr-\(suffix)"
            suffix += 1
        }
        return candidate
    }

    private static func writeAtomically(_ data: Data, to url: URL) throws {
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw PoCError.outputFailed(url, error.localizedDescription)
        }
    }
}

private enum Help {
    static let text = """
    GenImageASRPoC：使用 WhisperKit 對影片或音訊檔產生帶時間軸的字幕。

    用法：
      swift run GenImageASRPoC --input /path/to/video.mp4
      swift run GenImageASRPoC --input /path/to/video.mp4 --input /path/to/audio.m4a
      swift run GenImageASRPoC /path/to/audio.wav --language zh

    選項：
      --input PATH              可重複指定影片或音訊檔案。
      --model NAME              WhisperKit 模型名稱；預設 large-v3-v20240930_turbo。
      --model-repo REPOSITORY   模型來源；預設 argmaxinc/whisperkit-coreml。
      --model-folder PATH       使用已下載的本機模型目錄，不下載模型。
      --model-cache PATH        模型快取目錄；預設 Application Support/GenImage/Models/WhisperKit。
      --output-dir PATH         輸出目錄；預設 Application Support/GenImage/Generated/ASR。
      --language CODE|auto      指定語言，例如 zh、ja、ko、en；預設自動偵測。
      --chinese-script NAME     中文字形：asr、traditional 或 simplified；預設保留 ASR 原文。
      --word-timestamps         在 JSON 額外輸出單字級時間軸。
      --no-download             禁止自動下載模型。
      --keep-prepared-audio     保留轉換後供 Whisper 使用的暫存 M4A。
      -h, --help                顯示說明。

    每個輸入會輸出：
      <檔名>-asr.json  原始文字、片段時間軸與辨識資訊
      <檔名>-asr.srt   SRT 字幕
      <檔名>-asr.vtt   WebVTT 字幕
    """
}

@main
struct GenImageASRPoC {
    static func main() async {
        do {
            let arguments = try Arguments.parse(Array(CommandLine.arguments.dropFirst()))
            if arguments.showHelp {
                print(Help.text)
                return
            }
            try await run(arguments)
        } catch is CancellationError {
            fputs("已取消 ASR PoC。\n", stderr)
            Foundation.exit(EXIT_FAILURE)
        } catch {
            fputs("錯誤：\(error.localizedDescription)\n\n", stderr)
            fputs(Help.text, stderr)
            Foundation.exit(EXIT_FAILURE)
        }
    }

    private static func run(_ arguments: Arguments) async throws {
        if let modelFolder = arguments.modelFolder,
           !FileManager.default.fileExists(atPath: modelFolder.path) {
            throw PoCError.modelFolderMissing(modelFolder)
        }
        try FileManager.default.createDirectory(
            at: arguments.outputDirectory,
            withIntermediateDirectories: true
        )
        if arguments.downloadModels {
            try FileManager.default.createDirectory(
                at: arguments.modelCacheDirectory,
                withIntermediateDirectories: true
            )
        }

        print("正在初始化 WhisperKit：\(arguments.model)")
        let config = WhisperKitConfig(
            model: arguments.model,
            downloadBase: arguments.modelCacheDirectory,
            modelRepo: arguments.modelRepo,
            modelFolder: arguments.modelFolder?.path,
            verbose: false,
            prewarm: false,
            load: true,
            download: arguments.downloadModels
        )
        let whisperKit = try await WhisperKit(config)

        for input in arguments.inputs {
            try Task.checkCancellation()
            try await transcribe(
                input: input,
                arguments: arguments,
                whisperKit: whisperKit
            )
        }
    }

    private static func transcribe(
        input: URL,
        arguments: Arguments,
        whisperKit: WhisperKit
    ) async throws {
        let temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("genimage-asr-poc-\(UUID().uuidString)", isDirectory: true)
        defer {
            if !arguments.keepPreparedAudio {
                try? FileManager.default.removeItem(at: temporaryDirectory)
            }
        }

        print("\n準備媒體：\(input.path)")
        let prepared = try await MediaAudioPreparer.prepare(
            sourceURL: input,
            temporaryDirectory: temporaryDirectory
        )
        print("開始辨識，音訊長度：\(String(format: "%.2f", prepared.durationSeconds)) 秒")

        var decodeOptions = DecodingOptions(
            task: .transcribe,
            language: arguments.language,
            usePrefillPrompt: arguments.language != nil,
            detectLanguage: arguments.language == nil,
            skipSpecialTokens: true,
            withoutTimestamps: false,
            wordTimestamps: arguments.wordTimestamps,
            chunkingStrategy: .vad
        )
        decodeOptions.verbose = false
        let audioOptions = AudioInputOptions(
            channelMode: .sumChannels(nil),
            audioLoadingMode: .incremental(
                chunkDurationSeconds: 120,
                maxBufferedChunks: 2
            )
        )
        let results = try await whisperKit.transcribe(
            audioPath: prepared.audioURL.path,
            audioInputOptions: audioOptions,
            decodeOptions: decodeOptions
        )
        let segments = results
            .flatMap(\.segments)
            .filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .sorted { lhs, rhs in
                lhs.start == rhs.start ? lhs.end < rhs.end : lhs.start < rhs.start
            }
            .reduce(into: (segments: [ASRSegment](), previousEnd: 0.0)) { result, segment in
                let start = max(result.previousEnd, Double(max(0, segment.start)))
                let end = max(start + 0.05, Double(segment.end))
                let words = segment.words?.compactMap { word -> ASRWord? in
                    let wordStart = max(start, Double(max(0, word.start)))
                    let wordEnd = max(wordStart, Double(word.end))
                    guard wordEnd > wordStart else { return nil }
                    return ASRWord(
                        text: arguments.chineseScript.transform(word.word),
                        start: wordStart,
                        end: wordEnd,
                        probability: Double(word.probability)
                    )
                }
                result.segments.append(
                    ASRSegment(
                        index: result.segments.count + 1,
                        start: start,
                        end: end,
                        text: arguments.chineseScript.transform(
                            segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
                        ),
                        averageLogProbability: Double(segment.avgLogprob),
                        noSpeechProbability: Double(segment.noSpeechProb),
                        words: words
                    )
                )
                result.previousEnd = end
            }
            .segments
        let language = results.first?.language ?? arguments.language ?? "unknown"
        let report = ASRReport(
            sourceFile: prepared.sourceURL.path,
            durationSeconds: prepared.durationSeconds,
            model: arguments.model,
            language: language,
            text: segments.map(\.text).joined(separator: " "),
            segments: segments,
            generatedAt: .now
        )
        let outputs = try OutputFiles.write(
            report: report,
            segments: segments,
            baseName: input.deletingPathExtension().lastPathComponent,
            directory: arguments.outputDirectory
        )
        print("辨識完成：\(segments.count) 個字幕片段，語言：\(language)")
        outputs.forEach { print("輸出：\($0.path)") }
        if arguments.keepPreparedAudio {
            print("保留 Whisper 音訊：\(prepared.audioURL.path)")
        }
    }
}
