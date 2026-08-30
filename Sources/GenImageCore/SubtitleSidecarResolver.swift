import Foundation

public struct SubtitleSidecar: Hashable, Sendable {
    public let fileURL: URL
    public let format: SubtitleFormat
    public let assetID: UUID?

    public init(fileURL: URL, format: SubtitleFormat, assetID: UUID? = nil) {
        self.fileURL = fileURL
        self.format = format
        self.assetID = assetID
    }
}

public enum SubtitleSidecarResolver {
    private static let preferredExtensions = ["vtt", "srt"]

    public static func locate(
        for mediaURL: URL,
        fileManager: FileManager = .default
    ) -> SubtitleSidecar? {
        guard mediaURL.isFileURL else { return nil }
        let directoryURL = mediaURL.deletingLastPathComponent()
        let mediaStem = mediaURL.deletingPathExtension().lastPathComponent
        guard !mediaStem.isEmpty,
              let entries = try? fileManager.contentsOfDirectory(
                  at: directoryURL,
                  includingPropertiesForKeys: [.isRegularFileKey],
                  options: []
              ) else {
            return nil
        }

        for fileExtension in preferredExtensions {
            guard let candidate = entries.first(where: { entry in
                entry.pathExtension.caseInsensitiveCompare(fileExtension) == .orderedSame
                    && entry.deletingPathExtension().lastPathComponent
                        .caseInsensitiveCompare(mediaStem) == .orderedSame
                    && (try? entry.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
            }) else {
                continue
            }
            return SubtitleSidecar(
                fileURL: candidate,
                format: SubtitleFormat(rawValue: fileExtension)!
            )
        }
        return nil
    }

    public static func locate(
        for videoAsset: MediaAsset,
        among assets: [MediaAsset],
        fileManager: FileManager = .default
    ) -> SubtitleSidecar? {
        guard videoAsset.kind == .importedVideo || videoAsset.kind == .generatedVideo else {
            return nil
        }

        let mediaURLs = [videoAsset.fileURL, videoAsset.playbackURL].compactMap { $0 }
        for mediaURL in mediaURLs {
            if let sidecar = locate(for: mediaURL, fileManager: fileManager) {
                return SubtitleSidecar(
                    fileURL: sidecar.fileURL,
                    format: sidecar.format,
                    assetID: videoAsset.id
                )
            }
        }

        guard let subtitleAsset = assets.first(where: {
            $0.kind == .generatedSubtitle && $0.parentAssetID == videoAsset.id
        }) else {
            return nil
        }
        return subtitle(for: subtitleAsset)
    }

    public static func subtitle(for asset: MediaAsset) -> SubtitleSidecar? {
        guard asset.kind == .generatedSubtitle,
              let fileURL = asset.fileURL,
              let format = asset.subtitleFormat,
              let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey]),
              values.isRegularFile == true else {
            return nil
        }
        return SubtitleSidecar(fileURL: fileURL, format: format, assetID: asset.id)
    }

    public static func webVTTData(
        from data: Data,
        format: SubtitleFormat
    ) -> Data? {
        guard let decoded = decode(data) else { return nil }
        let normalized = decoded
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let content = normalized.first == "\u{feff}"
            ? String(normalized.dropFirst())
            : normalized

        let webVTT: String
        switch format {
        case .srt:
            let convertedLines = content
                .split(separator: "\n", omittingEmptySubsequences: false)
                .map { line in
                    line.contains("-->")
                        ? line.replacingOccurrences(of: ",", with: ".")
                        : String(line)
                }
            webVTT = "WEBVTT\n\n" + convertedLines.joined(separator: "\n")
        case .vtt:
            webVTT = content.hasPrefix("WEBVTT")
                ? content
                : "WEBVTT\n\n" + content
        }
        return webVTT.data(using: .utf8)
    }

    private static func decode(_ data: Data) -> String? {
        String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .utf16)
            ?? String(data: data, encoding: .utf16LittleEndian)
            ?? String(data: data, encoding: .utf16BigEndian)
    }
}
