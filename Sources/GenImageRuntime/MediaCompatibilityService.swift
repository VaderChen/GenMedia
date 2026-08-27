import Foundation

enum MediaCompatibilityTool: String, CaseIterable, Sendable {
    case ffmpeg
    case ffprobe
}

struct MediaProbeResult: Equatable, Sendable {
    let durationSeconds: Double
    let hasAudio: Bool
    let hasVideo: Bool
    let videoCodec: String?
    let videoCodecTag: String?
    let audioCodec: String?
    let pixelWidth: Int
    let pixelHeight: Int
}

enum MediaCompatibilityService {
    static func executable(
        for tool: MediaCompatibilityTool,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        bundleURL: URL? = Bundle.main.bundleURL,
        executableURL: URL? = Bundle.main.executableURL
    ) throws -> URL {
        let candidates = executableCandidates(
            for: tool,
            environment: environment,
            bundleURL: bundleURL,
            executableURL: executableURL
        )
        guard let executable = RuntimeExecutable.locate(candidates) else {
            throw MediaCompatibilityError.toolNotFound(
                tool: tool,
                checkedPaths: candidates.map(\.path)
            )
        }
        return executable
    }

    static func executableCandidates(
        for tool: MediaCompatibilityTool,
        environment: [String: String],
        bundleURL: URL?,
        executableURL: URL?
    ) -> [URL] {
        var candidates: [URL] = []

        for key in environmentKeys(for: tool) {
            if let configured = environment[key], !configured.isEmpty {
                candidates.append(URL(fileURLWithPath: configured))
            }
        }

        if tool == .ffprobe {
            for key in environmentKeys(for: .ffmpeg) {
                guard let configured = environment[key], !configured.isEmpty else { continue }
                candidates.append(
                    URL(fileURLWithPath: configured)
                        .deletingLastPathComponent()
                        .appendingPathComponent(tool.rawValue)
                )
            }
        }

        if let bundleURL, bundleURL.pathExtension.lowercased() == "app" {
            candidates.append(
                bundleURL
                    .appendingPathComponent("Contents", isDirectory: true)
                    .appendingPathComponent("Resources", isDirectory: true)
                    .appendingPathComponent("bin", isDirectory: true)
                    .appendingPathComponent(tool.rawValue, isDirectory: false)
            )
        }
        if let executableURL {
            let contentsDirectory = executableURL
                .deletingLastPathComponent()
                .deletingLastPathComponent()
            if contentsDirectory.lastPathComponent == "Contents" {
                candidates.append(
                    contentsDirectory
                        .appendingPathComponent("Resources", isDirectory: true)
                        .appendingPathComponent("bin", isDirectory: true)
                        .appendingPathComponent(tool.rawValue, isDirectory: false)
                )
            }
        }

        candidates.append(URL(fileURLWithPath: "/opt/homebrew/bin/\(tool.rawValue)"))
        candidates.append(URL(fileURLWithPath: "/usr/local/bin/\(tool.rawValue)"))
        candidates.append(URL(fileURLWithPath: "/usr/bin/\(tool.rawValue)"))
        candidates.append(
            contentsOf: RuntimeExecutable.pathCandidates(
                for: tool.rawValue,
                environment: environment
            )
        )

        return candidates.reduce(into: [URL]()) { result, candidate in
            if !result.contains(candidate) {
                result.append(candidate)
            }
        }
    }

    static func videoBitrateArguments(pixelWidth: Int, pixelHeight: Int) -> [String] {
        let referencePixelCount = 1_280.0 * 720.0
        let pixelCount = Double(max(2, pixelWidth)) * Double(max(2, pixelHeight))
        let targetBitsPerSecond = min(
            40_000_000.0,
            max(1_500_000.0, (pixelCount / referencePixelCount) * 8_000_000.0)
        )
        let bitrate = roundedBitrate(targetBitsPerSecond)
        let maximumBitrate = roundedBitrate(targetBitsPerSecond * 1.5)
        let bufferSize = roundedBitrate(targetBitsPerSecond * 2)
        return [
            "-b:v", formatBitrate(bitrate),
            "-maxrate", formatBitrate(maximumBitrate),
            "-bufsize", formatBitrate(bufferSize)
        ]
    }

    private static func roundedBitrate(_ bitsPerSecond: Double) -> Int {
        Int((bitsPerSecond / 100_000).rounded()) * 100_000
    }

    private static func formatBitrate(_ bitsPerSecond: Int) -> String {
        if bitsPerSecond.isMultiple(of: 1_000_000) {
            return "\(bitsPerSecond / 1_000_000)M"
        }
        return "\(bitsPerSecond / 1_000)k"
    }

    static func runFFmpeg(
        arguments: [String],
        log: RuntimeLog,
        pollInterval: Duration = .milliseconds(250),
        onPoll: () throws -> Void = {}
    ) async throws -> Int32 {
        try await RuntimeProcess.run(
            executable: executable(for: .ffmpeg),
            arguments: arguments,
            environment: RuntimeExecutable.environment(),
            log: log,
            pollInterval: pollInterval,
            onPoll: onPoll
        )
    }

    static func probe(sourceURL: URL) async throws -> MediaProbeResult {
        let logURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("genmedia-ffprobe-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: logURL) }
        let log = try RuntimeLog(at: logURL)
        defer { log.close() }

        let startedAt = Date()
        let status = try await RuntimeProcess.run(
            executable: executable(for: .ffprobe),
            arguments: [
                "-v", "error",
                "-print_format", "json",
                "-show_entries",
                "format=duration:stream=codec_type,codec_name,codec_tag_string,width,height:stream_tags=rotate:stream_side_data=rotation:stream_disposition=attached_pic",
                sourceURL.path
            ],
            environment: RuntimeExecutable.environment(),
            log: log,
            pollInterval: .milliseconds(100)
        ) {
            if Date().timeIntervalSince(startedAt) >= 60 {
                throw MediaCompatibilityError.probeTimedOut(sourceURL)
            }
        }
        guard status == 0 else {
            throw MediaCompatibilityError.probeFailed(
                sourceURL,
                status: status,
                message: log.message(fallback: "ffprobe 未提供錯誤訊息。")
            )
        }
        guard let data = log.data(),
              let document = try? JSONDecoder().decode(FFprobeDocument.self, from: data)
        else {
            throw MediaCompatibilityError.invalidProbeOutput(sourceURL)
        }

        let duration = document.format?.duration
            .flatMap { $0.isFinite ? max(0, $0) : nil }
            ?? 0
        let videoStream = document.streams.first {
            $0.codecType == "video" && !$0.isAttachedPicture
        }
        let audioStream = document.streams.first { $0.codecType == "audio" }
        let sourceWidth = max(0, videoStream?.width ?? 0)
        let sourceHeight = max(0, videoStream?.height ?? 0)
        let swapsDimensions = videoStream?.rotationDegrees.isRightAngleRotation == true
        return MediaProbeResult(
            durationSeconds: duration,
            hasAudio: audioStream != nil,
            hasVideo: videoStream != nil,
            videoCodec: videoStream?.codecName?.lowercased(),
            videoCodecTag: videoStream?.codecTagString?.lowercased(),
            audioCodec: audioStream?.codecName?.lowercased(),
            pixelWidth: swapsDimensions ? sourceHeight : sourceWidth,
            pixelHeight: swapsDimensions ? sourceWidth : sourceHeight
        )
    }

    private static func environmentKeys(for tool: MediaCompatibilityTool) -> [String] {
        switch tool {
        case .ffmpeg:
            ["GENMEDIA_FFMPEG", "GENIMAGE_FFMPEG"]
        case .ffprobe:
            ["GENMEDIA_FFPROBE", "GENIMAGE_FFPROBE"]
        }
    }
}

private struct FFprobeDocument: Decodable {
    struct Stream: Decodable {
        let codecType: String?
        let codecName: String?
        let codecTagString: String?
        let width: Int?
        let height: Int?
        let tags: Tags?
        let sideDataList: [SideData]
        let disposition: Disposition?

        struct Tags: Decodable {
            let rotate: String?
        }

        struct SideData: Decodable {
            let rotation: Double?
        }

        struct Disposition: Decodable {
            let attachedPicture: Int?

            enum CodingKeys: String, CodingKey {
                case attachedPicture = "attached_pic"
            }
        }

        var isAttachedPicture: Bool {
            disposition?.attachedPicture == 1
        }

        var rotationDegrees: Double {
            sideDataList.compactMap(\.rotation).first
                ?? tags?.rotate.flatMap(Double.init)
                ?? 0
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            codecType = try container.decodeIfPresent(String.self, forKey: .codecType)
            codecName = try container.decodeIfPresent(String.self, forKey: .codecName)
            codecTagString = try container.decodeIfPresent(String.self, forKey: .codecTagString)
            width = try container.decodeIfPresent(Int.self, forKey: .width)
            height = try container.decodeIfPresent(Int.self, forKey: .height)
            tags = try container.decodeIfPresent(Tags.self, forKey: .tags)
            sideDataList = try container.decodeIfPresent([SideData].self, forKey: .sideDataList) ?? []
            disposition = try container.decodeIfPresent(Disposition.self, forKey: .disposition)
        }

        enum CodingKeys: String, CodingKey {
            case codecType = "codec_type"
            case codecName = "codec_name"
            case codecTagString = "codec_tag_string"
            case width
            case height
            case tags
            case sideDataList = "side_data_list"
            case disposition
        }
    }

    struct Format: Decodable {
        let duration: Double?

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            if let value = try? container.decode(Double.self, forKey: .duration) {
                duration = value
            } else if let value = try? container.decode(String.self, forKey: .duration) {
                duration = Double(value)
            } else {
                duration = nil
            }
        }

        enum CodingKeys: String, CodingKey {
            case duration
        }
    }

    let streams: [Stream]
    let format: Format?

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        streams = try container.decodeIfPresent([Stream].self, forKey: .streams) ?? []
        format = try container.decodeIfPresent(Format.self, forKey: .format)
    }

    enum CodingKeys: String, CodingKey {
        case streams
        case format
    }
}

private extension Double {
    var isRightAngleRotation: Bool {
        Int(abs(self).rounded()) % 180 == 90
    }
}

enum MediaCompatibilityError: LocalizedError, Sendable {
    case toolNotFound(tool: MediaCompatibilityTool, checkedPaths: [String])
    case probeTimedOut(URL)
    case probeFailed(URL, status: Int32, message: String)
    case invalidProbeOutput(URL)

    var errorDescription: String? {
        switch self {
        case let .toolNotFound(tool, checkedPaths):
            "找不到內建或本機 \(tool.rawValue)；已檢查：\(checkedPaths.joined(separator: "、"))"
        case let .probeTimedOut(url):
            "媒體探測超過 60 秒：\(url.lastPathComponent)"
        case let .probeFailed(url, status, message):
            "無法探測媒體「\(url.lastPathComponent)」（\(status)）：\(message)"
        case let .invalidProbeOutput(url):
            "ffprobe 未回傳有效的媒體資訊：\(url.lastPathComponent)"
        }
    }
}
