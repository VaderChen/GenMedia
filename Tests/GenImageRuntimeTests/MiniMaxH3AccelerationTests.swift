import Foundation
import GenImageCore
import Testing
@testable import GenImageRuntime

struct MiniMaxH3AccelerationTests {
    @Test func acceptsOnlyTheAdaptersSamplingAndBaseContracts() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data([1]).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        for acceleration in MiniMaxH3Acceleration.allCases {
            let lora = VideoGenerationLoRA(configuration: .init(modelID: acceleration.modelID, scale: 1), localURL: file)
            for base in acceleration.compatibleModelIDs {
                for steps in acceleration.allowedSteps {
                    #expect(try MiniMaxH3VideoGenerationService.validatedAcceleration(
                        [lora], modelID: base, capability: .textToVideo, steps: steps) == acceleration)
                }
            }
            for steps in [0, 3, 9, 30] {
                #expect(throws: MiniMaxH3VideoRuntimeError.self) {
                    try MiniMaxH3VideoGenerationService.validatedAcceleration(
                        [lora], modelID: acceleration.compatibleModelIDs[0], capability: .textToVideo, steps: steps)
                }
            }
            #expect(throws: LTXVideoRuntimeError.self) {
                try LTXVideoGenerationService.validateLoRAs([lora], modelID: LoRACatalog.ltxModelIDs[0])
            }
        }
        #expect(try MiniMaxH3VideoGenerationService.validatedAcceleration([], modelID: "base", capability: .textToVideo, steps: 30) == nil)
    }

    @Test(arguments: ["pruned", "ref2va", "ltx", "image-to-video", "mixed", "conditioning", "conditioning-scale", "zero", "nan", "too-large", "missing", "unknown"])
    func rejectsIncompatibleRequests(reason: String) throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try Data([1]).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        var lora = VideoGenerationLoRA(configuration: .init(modelID: MiniMaxH3Acceleration.larryV4.modelID, scale: 1), localURL: file)
        var base = MiniMaxH3Acceleration.fullModelIDs[0]
        var capability = ModelCapability.textToVideo
        switch reason {
        case "pruned": base = MiniMaxH3Acceleration.prunedModelID
        case "ref2va": base = "Abiray/MiniMax-H3-GGUF@ref2va-Q4_0"
        case "ltx": base = LoRACatalog.ltxModelIDs[0]
        case "image-to-video": capability = .imageToVideo
        case "conditioning": lora.conditioning = .sourceImageCanny
        case "conditioning-scale": lora.conditioningScale = 0.5
        case "zero": lora.scale = 0
        case "nan": lora.scale = .nan
        case "too-large": lora.scale = 2
        case "missing": lora.localURL = file.appendingPathExtension("missing")
        case "unknown": lora.modelID = "custom/unknown"
        default: break
        }
        #expect(throws: MiniMaxH3VideoRuntimeError.self) {
            try MiniMaxH3VideoGenerationService.validatedAcceleration(
                reason == "mixed" ? [lora, lora] : [lora], modelID: base, capability: capability, steps: 6)
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_H3_LORA_INSTALL_ROOT"] != nil))
    func downloadsAndRediscoversAccelerationAdapters() async throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_H3_LORA_INSTALL_ROOT"]))
        for acceleration in [MiniMaxH3Acceleration.lightX2V8Step768, .lightX2V4StepV12, .larryV4] {
            let entry = acceleration.entry
            let installed = try await HuggingFaceModelInstaller().install(modelID: entry.id, rootURL: root, progress: { _ in })
            #expect(try HuggingFaceModelInstaller.verify(modelID: entry.id, rootURL: root) == installed)
            #expect(LocalModelDiscovery.discoverLoRAs(at: root).contains {
                $0.localURL == installed.resolvingSymlinksInPath().standardizedFileURL && $0.compatibleCapabilities == [.textToVideo]
            })
        }
    }
}
