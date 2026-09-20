import Foundation
import Testing
@testable import GenImageCore

struct PersistenceSafetyTests {
    @Test func unwrittenOutputsHaveUniqueNamesEvenInConcurrentBatches() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let date = Date(timeIntervalSince1970: 1_000)
        let urls = await withTaskGroup(of: URL.self, returning: [URL].self) { group in
            for _ in 0..<100 {
                group.addTask { OutputFileNaming.imageURL(in: directory, date: date) }
            }
            var result: [URL] = []
            for await url in group { result.append(url) }
            return result
        }
        #expect(Set(urls).count == 100)
        #expect(urls.allSatisfy { $0.pathExtension == "png" && $0.lastPathComponent.hasPrefix("Image-") })
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    @Test func failedWorkspaceRestorePreservesOriginalDataAndDisablesDestructiveWrites() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("workspace.json")
        let missing = ProjectWorkspacePersistence.restore(from: url)
        #expect(missing.snapshot == nil)
        #expect(missing.allowsPersistence && missing.allowsCacheCleanup)

        let malformed = Data("{broken".utf8)
        try malformed.write(to: url)
        let corrupt = ProjectWorkspacePersistence.restore(from: url)
        #expect(corrupt.errorMessage != nil)
        #expect(!corrupt.allowsPersistence && !corrupt.allowsCacheCleanup)
        #expect(try Data(contentsOf: url) == malformed)

        let newer = ProjectWorkspaceSnapshot(schemaVersion: 999, projects: [], selectedProjectID: UUID(),
            assets: [], operations: [], selectedAssetID: nil, comparisonAssetID: nil)
        try ProjectWorkspacePersistence.save(newer, to: url)
        let original = try Data(contentsOf: url)
        let unsupported = ProjectWorkspacePersistence.restore(from: url)
        #expect(unsupported.snapshot == nil)
        #expect(!unsupported.allowsPersistence && !unsupported.allowsCacheCleanup)
        #expect(try Data(contentsOf: url) == original)

        let unreadable = ProjectWorkspacePersistence.restore(from: directory)
        #expect(!unreadable.allowsPersistence && !unreadable.allowsCacheCleanup)
    }

    @Test func customProfilesSurviveReloadAndDuplicationWithMusicSettings() throws {
        let suite = "GenImage.tests.\(UUID())"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let builtIn = InferenceProfile(name: "Music", capability: .textToMusic, modelID: "local/model",
            architecture: .mlxSwift, defaults: ProfileDefaults(durationSeconds: 45),
            music: ProfileMusicConfiguration(minimumDurationSeconds: 10, maximumDurationSeconds: 120,
                durationSemantics: .maximum), notes: "preserve", isBuiltIn: true)
        let copy = builtIn.duplicated()
        #expect(copy.id != builtIn.id)
        #expect(!copy.isBuiltIn)
        #expect(copy.music == builtIn.music)
        #expect(copy.defaults == builtIn.defaults)
        #expect(copy.loras == builtIn.loras)
        try CustomProfilePersistence.save([builtIn, copy], to: defaults)
        let reloaded = try CustomProfilePersistence.load(from: defaults)
        #expect(reloaded == [copy])

        let invalid = Data("broken".utf8)
        defaults.set(invalid, forKey: CustomProfilePersistence.key)
        #expect(throws: (any Error).self) { try CustomProfilePersistence.load(from: defaults) }
        #expect(defaults.data(forKey: CustomProfilePersistence.key) == invalid)
    }
}
