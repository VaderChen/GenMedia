import Foundation
import Testing
@testable import GenImageCore

struct LoRACatalogTests {
    @Test func newAdaptersHavePinnedDownloadsAndResolvableVideoProfiles() {
        let entries = LoRACatalog.entries
        #expect(entries.count == 13)
        #expect(Set(entries.map(\.id)).count == entries.count)
        #expect(Set(entries.map(\.directoryName)).count == entries.count)
        for entry in entries {
            #expect(entry.revision.count == 40)
            #expect(entry.filename.hasSuffix(".safetensors"))
            #expect(entry.sizeBytes > 0)
            #expect(ModelCatalog.builtIn.contains { $0.id == entry.id })
        }
        for profile in LoRACatalog.videoProfiles {
            #expect(profile.capability == .textToVideo)
            #expect(ModelCatalog.builtIn.contains { $0.id == profile.modelID })
            #expect(profile.loras.count == 1)
            #expect(profile.loras.allSatisfy { LoRACatalog.entry(for: $0.modelID)?.capability == .textToVideo })
        }
    }

    @Test func discoveryKeepsVideoAdaptersOutOfImageSelection() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        for entry in LoRACatalog.entries {
            let directory = root.appendingPathComponent(entry.directoryName)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data([1]).write(to: directory.appendingPathComponent(entry.filename))
            try JSONSerialization.data(withJSONObject: ["modelID": entry.id])
                .write(to: directory.appendingPathComponent("genimage-model.json"))
        }
        let catalog = LocalModelDiscovery.discover(at: root)
        #expect(catalog.loras.count == LoRACatalog.entries.count)
        for entry in LoRACatalog.entries {
            let lora = try #require(catalog.loras.first { $0.displayName == entry.displayName })
            #expect(lora.compatibleCapabilities == entry.compatibleCapabilities)
            #expect(catalog.models.contains { $0.id == entry.id && $0.localURL == lora.localURL })
        }
    }
}
