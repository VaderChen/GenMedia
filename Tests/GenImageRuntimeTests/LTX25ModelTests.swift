import Foundation
import GenImageCore
import Testing
@testable import GenImageRuntime

struct LTX25ModelTests {
    @Test func distilledProfileRejectsOtherSamplingAndAdapters() throws {
        var options = VideoGenerationOptions(prompt: "A red teapot.", width: 256, height: 256, steps: 8,
            outputCount: 1, frameCount: 9, frameRate: 24, seed: 42)
        try LTXVideoGenerationService.validateLTX25(options, capability: .textToVideo)
        options.steps = 4
        #expect(throws: LTXVideoRuntimeError.self) { try LTXVideoGenerationService.validateLTX25(options, capability: .textToVideo) }
        options.steps = 8; options.width = 272
        #expect(throws: LTXVideoRuntimeError.self) { try LTXVideoGenerationService.validateLTX25(options, capability: .textToVideo) }
        options.width = 256
        #expect(throws: LTXVideoRuntimeError.self) { try LTXVideoGenerationService.validateLTX25(options, capability: .imageToVideo) }
        let lora = VideoGenerationLoRA(configuration: .init(modelID: LoRACatalog.videoProfiles[0].loras[0].modelID, scale: 1),
            localURL: URL(fileURLWithPath: "/lora"))
        #expect(throws: LTXVideoRuntimeError.self) { try LTXVideoGenerationService.validateLoRAs([lora], modelID: LTX25Model.id) }
    }

    @Test func catalogAndDiscoveryRequireTheNewTextEncoder() throws {
        #expect(LTX25Model.profile.defaults.steps == 8)
        #expect(LTX25Model.profile.requiredModelIDs == [LTX25Model.id])
        #expect(LTX25Model.descriptor.capabilities == [.textToVideo])
        #expect(HuggingFaceModelInstaller.supports(modelID: LTX25Model.id))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent(LTX25Model.directoryName)
        for file in LTX25Model.requiredFiles {
            let path = model.appendingPathComponent(file)
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data([1]).write(to: path)
        }
        try JSONSerialization.data(withJSONObject: ["modelID": LTX25Model.id]).write(to: model.appendingPathComponent("genimage-model.json"))
        #expect(LocalModelDiscovery.discover(at: root).models.contains { $0.id == LTX25Model.id })
        try FileManager.default.removeItem(at: model.appendingPathComponent("gemma4-12b-ltx-v1/model.safetensors"))
        #expect(!LocalModelDiscovery.discover(at: root).models.contains { $0.id == LTX25Model.id })
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_LTX25_INSTALL_ROOT"] != nil))
    func installsAndVerifiesOnlyTheDistilledPack() async throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_LTX25_INSTALL_ROOT"]))
        let installed = try await HuggingFaceModelInstaller().install(modelID: LTX25Model.id, rootURL: root, progress: { _ in })
        #expect(try HuggingFaceModelInstaller.verify(modelID: LTX25Model.id, rootURL: root) == installed)
        #expect(LocalModelDiscovery.discover(at: root).models.contains { $0.id == LTX25Model.id && $0.localURL == installed })
        #expect(!FileManager.default.fileExists(atPath: installed.appendingPathComponent("transformer-dev.safetensors").path))
    }
}
