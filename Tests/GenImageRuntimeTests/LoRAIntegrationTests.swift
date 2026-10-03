import Foundation
import GenImageCore
import Testing
@testable import GenImageRuntime

struct LoRAIntegrationTests {
    @Test func allNewDownloadsResolveToVerifiedAdapterFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        for entry in LoRACatalog.entries {
            let directory = root.appendingPathComponent(entry.directoryName)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let file = directory.appendingPathComponent(entry.filename)
            try Data([1, 2, 3]).write(to: file)
            let manifest: [String: Any] = ["schemaVersion": 1, "modelID": entry.id,
                "installedAt": "2026-10-03T00:00:00Z",
                "files": [["relativePath": entry.filename, "size": 3, "repository": entry.repository, "revision": entry.revision]]]
            try JSONSerialization.data(withJSONObject: manifest).write(to: directory.appendingPathComponent("genimage-model.json"))
            let verified = try HuggingFaceModelInstaller.verify(modelID: entry.id, rootURL: root)
            #expect(verified == file)
            try Data([0]).write(to: file)
            #expect(throws: (any Error).self) { try HuggingFaceModelInstaller.verify(modelID: entry.id, rootURL: root) }
        }
    }

    @Test func videoLoRAValidationAcceptsLTX23AndRejectsIncompatibleRequests() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data([1]).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let entry = try #require(LoRACatalog.entries.first { $0.capability == .textToVideo })
        let lora = VideoGenerationLoRA(configuration: ProfileLoRAConfiguration(modelID: entry.id, scale: 0.8), localURL: file)
        for modelID in LoRACatalog.ltxModelIDs {
            try LTXVideoGenerationService.validateLoRAs([lora], modelID: modelID)
        }
        #expect(throws: LTXVideoRuntimeError.self) {
            try LTXVideoGenerationService.validateLoRAs([lora], modelID: "city96/ltx-video-0.9.6-distilled-gguf@q4_k_m")
        }
        var invalid = lora
        invalid.conditioning = .sourceImageCanny
        #expect(throws: LTXVideoRuntimeError.self) {
            try LTXVideoGenerationService.validateLoRAs([invalid], modelID: LoRACatalog.ltxModelIDs[0])
        }
        invalid = lora
        invalid.scale = .nan
        #expect(throws: LTXVideoRuntimeError.self) {
            try LTXVideoGenerationService.validateLoRAs([invalid], modelID: LoRACatalog.ltxModelIDs[0])
        }
        invalid = lora
        invalid.modelID = LoRACatalog.entries[0].id
        #expect(throws: LTXVideoRuntimeError.self) {
            try LTXVideoGenerationService.validateLoRAs([invalid], modelID: LoRACatalog.ltxModelIDs[0])
        }
    }

    // Opt-in smoke check uses the real installer and the user's existing model directory.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_LORA_INSTALL_ROOT"] != nil))
    func downloadsAndRediscoversImageAndVideoAdapters() async throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_LORA_INSTALL_ROOT"]))
        let ids = ["renderartist/Saturday-Morning-Z-Image-Turbo", "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In"]
        for id in ids {
            let installed = try await HuggingFaceModelInstaller().install(modelID: id, rootURL: root, progress: { _ in })
            let verified = try HuggingFaceModelInstaller.verify(modelID: id, rootURL: root)
            #expect(installed == verified)
            let entry = try #require(LoRACatalog.entry(for: id))
            #expect(LocalModelDiscovery.discoverLoRAs(at: root).contains {
                $0.localURL == installed.resolvingSymlinksInPath().standardizedFileURL && $0.compatibleCapabilities == [entry.capability]
            })
        }
    }
}
