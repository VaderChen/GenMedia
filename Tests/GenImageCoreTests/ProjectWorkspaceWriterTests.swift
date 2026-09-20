import Foundation
import Testing
@testable import GenImageCore

struct ProjectWorkspaceWriterTests {
    @Test func flushSavesLatestQueuedSelectionAndAssets() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("workspace.json")
        let writer = ProjectWorkspaceWriter(url: url, delay: 60)
        let project = Project(name: "test")
        var snapshot = ProjectWorkspaceSnapshot(projects: [project], selectedProjectID: project.id,
            assets: [], operations: [], selectedAssetID: nil, comparisonAssetID: nil)
        writer.schedule(snapshot)
        let asset = MediaAsset(projectID: project.id, kind: .imported, title: "latest", pixelWidth: 1, pixelHeight: 1, createdAt: Date(timeIntervalSince1970: 1_700_000_000))
        snapshot.assets = [asset]
        snapshot.selectedAssetID = asset.id
        writer.schedule(snapshot)
        try writer.flush()
        let restored = try #require(try ProjectWorkspacePersistence.load(from: url))
        #expect(restored.assets == [asset])
        #expect(restored.selectedAssetID == asset.id)
    }
}
