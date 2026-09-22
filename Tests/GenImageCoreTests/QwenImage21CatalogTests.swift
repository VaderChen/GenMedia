import Foundation
import Testing
@testable import GenImageCore

struct QwenImage21CatalogTests {
    @Test func sharesOneInstallationAcrossBothCapabilities() {
        let model = ModelCatalog.builtIn.filter { $0.id == QwenImage21Model.id }
        #expect(model.count == 1)
        #expect(model[0].capabilities == [.textToImage, .imageToImage])
        let profiles = ModelCatalog.builtInProfiles.filter { $0.modelID == QwenImage21Model.id }
        #expect(profiles.count == 2)
        #expect(profiles.allSatisfy { $0.architecture == .mlxSwift && $0.modelRevision == QwenImage21Model.revision })
        #expect(model[0].licenseName == "Qwen Research License")
    }

    @Test func discoveryRejectsIncompleteAndEmptyWeights() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let model = root.appendingPathComponent(QwenImage21Model.directoryName)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = try JSONSerialization.data(withJSONObject: ["modelID": QwenImage21Model.id])
        try manifest.write(to: model.appendingPathComponent("genimage-model.json"))
        for path in QwenImage21Model.requiredFiles {
            let url = model.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([1]).write(to: url)
        }
        #expect(LocalModelDiscovery.discover(at: root).models.contains { $0.id == QwenImage21Model.id })
        let weight = model.appendingPathComponent("vae/model.safetensors")
        try Data().write(to: weight)
        #expect(!LocalModelDiscovery.discover(at: root).models.contains { $0.id == QwenImage21Model.id })
        try FileManager.default.removeItem(at: weight)
        #expect(!LocalModelDiscovery.discover(at: root).models.contains { $0.id == QwenImage21Model.id })
    }
}
