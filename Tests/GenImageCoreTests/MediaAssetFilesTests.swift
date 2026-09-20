import Foundation
import Testing
@testable import GenImageCore

struct MediaAssetFilesTests {
    private func sandbox() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("asset-files-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func asset(_ source: URL?, playback: URL? = nil) -> MediaAsset {
        MediaAsset(projectID: UUID(), kind: .importedVideo, title: "original",
            fileURL: source, playbackURL: playback, pixelWidth: 1, pixelHeight: 1)
    }

    @Test func sharedSourcesAndPlaybackFilesSurviveUntilTheirLastReferenceIsRemoved() throws {
        let root = try sandbox()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("shared.mp4")
        try Data("keep".utf8).write(to: file)
        for remaining in [[asset(file)], [asset(nil, playback: file)]] {
            #expect(try MediaAssetFiles.remove(at: file,
                preserving: MediaAssetFiles.references(in: remaining)) == .retained)
            #expect(try Data(contentsOf: file) == Data("keep".utf8))
        }
        #expect(try MediaAssetFiles.remove(at: file, preserving: []) == .removed)
        #expect(!FileManager.default.fileExists(atPath: file.path))
        #expect(try MediaAssetFiles.remove(at: file, preserving: []) == .missing)
    }

    @Test func automaticCleanupKeepsSourcesAndRejectsPathsOutsideTheCache() throws {
        let root = try sandbox()
        defer { try? FileManager.default.removeItem(at: root) }
        let cache = root.appendingPathComponent("MediaCache", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("generated.mp4")
        let proxy = cache.appendingPathComponent("proxy.mp4")
        let alias = cache.appendingPathComponent("external.mp4")
        for url in [source, proxy] { try Data("keep".utf8).write(to: url) }
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: source)
        #expect(try MediaAssetFiles.remove(at: source, preserving: [], within: cache) == .outsideManagedDirectory)
        #expect(try MediaAssetFiles.remove(at: alias, preserving: [], within: cache) == .outsideManagedDirectory)
        #expect(try MediaAssetFiles.remove(at: proxy, preserving: [proxy], within: cache) == .retained)
        #expect(try MediaAssetFiles.remove(at: cache, preserving: [], within: cache) == .outsideManagedDirectory)
        #expect(try Data(contentsOf: source) == Data("keep".utf8))
        #expect(try Data(contentsOf: proxy) == Data("keep".utf8))
        #expect(try MediaAssetFiles.remove(at: proxy, preserving: [], within: cache) == .removed)
    }

    @Test func symlinkReferencesProtectTheirTargetFromDeletion() throws {
        let root = try sandbox()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("source.mp4")
        let alias = root.appendingPathComponent("alias.mp4")
        try Data("keep".utf8).write(to: file)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: file)
        #expect(try MediaAssetFiles.remove(at: file, preserving: [alias]) == .retained)
        #expect(try Data(contentsOf: alias) == Data("keep".utf8))
    }

    @Test func renameUpdatesEveryWorkspaceAndPlaybackReference() throws {
        let root = try sandbox()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("source.mp4")
        let destination = root.appendingPathComponent("renamed.mp4")
        let directoryAlias = root.appendingPathComponent("alias")
        try Data("keep".utf8).write(to: file)
        try FileManager.default.createSymbolicLink(at: directoryAlias, withDestinationURL: root)
        let first = asset(file, playback: file)
        let duplicate = asset(directoryAlias.appendingPathComponent("source.mp4"))
        let playbackOnly = asset(nil, playback: file)
        let result = try MediaAssetFiles.rename(assetID: first.id, to: "renamed",
            in: [first, duplicate, playbackOnly])
        #expect(result[0].fileURL == destination && result[1].fileURL == destination)
        #expect(result[0].playbackURL == destination && result[2].playbackURL == destination)
        #expect(result[0].title == "renamed" && result[1].title == "renamed")
        #expect(result[2].title == playbackOnly.title)
        #expect(result.map(\.id) == [first.id, duplicate.id, playbackOnly.id])
        #expect(!FileManager.default.fileExists(atPath: file.path))
        #expect(try Data(contentsOf: destination) == Data("keep".utf8))
    }

    @Test func renamingASymlinkDoesNotRedirectAssetsUsingItsTarget() throws {
        let root = try sandbox()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("source.mp4")
        let alias = root.appendingPathComponent("alias.mp4")
        try Data("keep".utf8).write(to: file)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: file)
        let direct = asset(file)
        let link = asset(alias)
        let result = try MediaAssetFiles.rename(assetID: link.id, to: "new-alias", in: [direct, link])
        #expect(result[0] == direct)
        #expect(result[1].fileURL == root.appendingPathComponent("new-alias.mp4"))
        #expect(try Data(contentsOf: file) == Data("keep".utf8))
    }

    @Test func directoriesCannotBeRenamedOrRemovedAsMedia() throws {
        let root = try sandbox()
        defer { try? FileManager.default.removeItem(at: root) }
        let directory = root.appendingPathComponent("folder")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let sentinel = directory.appendingPathComponent("keep")
        try Data("keep".utf8).write(to: sentinel)
        let input = asset(directory)
        #expect(throws: CocoaError.self) { try MediaAssetFiles.remove(at: directory, preserving: []) }
        #expect(throws: MediaAssetFiles.RenameError.self) {
            try MediaAssetFiles.rename(assetID: input.id, to: "moved", in: [input])
        }
        #expect(try Data(contentsOf: sentinel) == Data("keep".utf8))
    }

    @Test(arguments: ["../escape", "", ".", "existing.mp4", "bad\0name"])
    func rejectedRenamesPreserveBothFiles(name: String) throws {
        let root = try sandbox()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.mp4")
        let existing = root.appendingPathComponent("existing.mp4")
        try Data("source".utf8).write(to: source)
        try Data("existing".utf8).write(to: existing)
        let input = asset(source)
        #expect(throws: MediaAssetFiles.RenameError.self) {
            try MediaAssetFiles.rename(assetID: input.id, to: name, in: [input])
        }
        #expect(try Data(contentsOf: source) == Data("source".utf8))
        #expect(try Data(contentsOf: existing) == Data("existing".utf8))
    }
}
