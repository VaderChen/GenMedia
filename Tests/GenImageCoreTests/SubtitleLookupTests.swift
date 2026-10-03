import Foundation
import Testing
@testable import GenImageCore

struct SubtitleLookupTests {
    private final class CountingFileManager: FileManager, @unchecked Sendable {
        var listings = 0
        override func contentsOfDirectory(at url: URL, includingPropertiesForKeys keys: [URLResourceKey]?,
            options mask: FileManager.DirectoryEnumerationOptions = []) throws -> [URL] {
            listings += 1
            return try super.contentsOfDirectory(at: url, includingPropertiesForKeys: keys, options: mask)
        }
    }

    @Test func sharesOneListingAndRefreshesWithANewLookup() throws {
        let directory = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = video(directory.appendingPathComponent("影片.MP4"))
        let second = video(directory.appendingPathComponent("Second.mp4"))
        let srt = directory.appendingPathComponent("影片.srt")
        let vtt = directory.appendingPathComponent("SECOND.VTT")
        try Data().write(to: srt)
        try Data().write(to: vtt)
        let manager = CountingFileManager()
        var lookup = SubtitleSidecarResolver.Lookup(assets: [first, second], fileManager: manager)
        #expect(lookup.locate(for: first)?.fileURL.standardizedFileURL == srt.standardizedFileURL)
        #expect(lookup.locate(for: second)?.fileURL.standardizedFileURL == vtt.standardizedFileURL)
        #expect(manager.listings == 1)
        let newVTT = directory.appendingPathComponent("影片.vtt")
        try Data().write(to: newVTT)
        var refreshed = SubtitleSidecarResolver.Lookup(assets: [first], fileManager: manager)
        #expect(refreshed.locate(for: first)?.fileURL.standardizedFileURL == newVTT.standardizedFileURL)
        #expect(manager.listings == 2)
    }

    @Test func preservesOriginalPlaybackAndFirstLinkedSubtitlePriority() throws {
        let directory = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var asset = video(directory.appendingPathComponent("original.mp4"))
        asset.playbackURL = directory.appendingPathComponent("proxy.mp4")
        let original = directory.appendingPathComponent("original.srt")
        let proxy = directory.appendingPathComponent("proxy.vtt")
        let linked = directory.appendingPathComponent("linked.srt")
        for url in [original, proxy, linked] { try Data().write(to: url) }
        var subtitle = MediaAsset(projectID: asset.projectID, parentAssetID: asset.id,
            kind: .generatedSubtitle, title: "caption", fileURL: linked,
            pixelWidth: 0, pixelHeight: 0, subtitleFormat: .srt)
        var lookup = SubtitleSidecarResolver.Lookup(assets: [subtitle])
        #expect(lookup.locate(for: asset)?.fileURL.standardizedFileURL == original.standardizedFileURL)
        try FileManager.default.removeItem(at: original)
        lookup = SubtitleSidecarResolver.Lookup(assets: [subtitle])
        #expect(lookup.locate(for: asset)?.fileURL.standardizedFileURL == proxy.standardizedFileURL)
        try FileManager.default.removeItem(at: proxy)
        lookup = SubtitleSidecarResolver.Lookup(assets: [subtitle])
        #expect(lookup.locate(for: asset)?.assetID == subtitle.id)
        let valid = subtitle
        subtitle.fileURL = nil
        lookup = SubtitleSidecarResolver.Lookup(assets: [subtitle, valid])
        #expect(lookup.locate(for: asset) == nil)
        var image = asset
        image.kind = .imported
        #expect(lookup.locate(for: image) == nil)
    }

    private func video(_ url: URL) -> MediaAsset {
        MediaAsset(projectID: UUID(), kind: .importedVideo, title: "video", fileURL: url,
            pixelWidth: 256, pixelHeight: 256)
    }
}
