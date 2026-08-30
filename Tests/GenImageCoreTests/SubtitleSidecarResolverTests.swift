import Foundation
import Testing

@testable import GenImageCore

struct SubtitleSidecarResolverTests {
    @Test func locatesSameStemAndPrefersWebVTT() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let mediaURL = directory.appendingPathComponent("Movie.MP4")
        FileManager.default.createFile(atPath: mediaURL.path, contents: Data())
        FileManager.default.createFile(
            atPath: directory.appendingPathComponent("movie.srt").path,
            contents: Data()
        )
        let vttURL = directory.appendingPathComponent("MOVIE.VTT")
        FileManager.default.createFile(atPath: vttURL.path, contents: Data())

        let sidecar = SubtitleSidecarResolver.locate(for: mediaURL)

        #expect(sidecar?.fileURL.standardizedFileURL == vttURL.standardizedFileURL)
        #expect(sidecar?.format == .vtt)
    }

    @Test func doesNotMatchAFileWithAnotherStem() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let mediaURL = directory.appendingPathComponent("Movie.MP4")
        FileManager.default.createFile(atPath: mediaURL.path, contents: Data())
        FileManager.default.createFile(
            atPath: directory.appendingPathComponent("Movie.en.srt").path,
            contents: Data()
        )

        #expect(SubtitleSidecarResolver.locate(for: mediaURL) == nil)
    }

    @Test func locatesGeneratedSubtitleLinkedToVideoAsset() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let videoID = UUID()
        let subtitleID = UUID()
        let videoURL = directory.appendingPathComponent("Movie.mp4")
        let subtitleURL = directory.appendingPathComponent("Subtitle-1.srt")
        FileManager.default.createFile(atPath: videoURL.path, contents: Data())
        FileManager.default.createFile(atPath: subtitleURL.path, contents: Data())
        let video = MediaAsset(
            id: videoID,
            projectID: UUID(),
            kind: .importedVideo,
            title: "Movie",
            fileURL: videoURL,
            pixelWidth: 1920,
            pixelHeight: 1080,
            mediaDurationSeconds: 5
        )
        let subtitle = MediaAsset(
            id: subtitleID,
            projectID: video.projectID,
            parentAssetID: videoID,
            kind: .generatedSubtitle,
            title: "字幕",
            fileURL: subtitleURL,
            pixelWidth: 0,
            pixelHeight: 0,
            subtitleFormat: .srt
        )

        let sidecar = SubtitleSidecarResolver.locate(for: video, among: [video, subtitle])

        #expect(sidecar?.fileURL.standardizedFileURL == subtitleURL.standardizedFileURL)
        #expect(sidecar?.format == .srt)
        #expect(sidecar?.assetID == subtitleID)
    }

    @Test func convertsSRTTimingToWebVTT() throws {
        let srt = "1\n00:00:01,250 --> 00:00:02,500\n字幕\n"
        let data = try #require(
            SubtitleSidecarResolver.webVTTData(
                from: Data(srt.utf8),
                format: .srt
            )
        )

        #expect(
            String(data: data, encoding: .utf8)
                == "WEBVTT\n\n1\n00:00:01.250 --> 00:00:02.500\n字幕\n"
        )
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("GenImage-SubtitleSidecar-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }
}
