import GenImageCore
import Testing
@testable import GenImageApp

@MainActor
struct LoRAProfileTests {
    @Test func installingBaseModelsKeepsEveryCameraProfileAvailable() throws {
        let baseProfiles = try LoRACatalog.ltxModelIDs.map { id in
            try #require(ModelCatalog.builtInProfiles.first {
                $0.modelID == id && $0.capability == .textToVideo && $0.loras.isEmpty
            })
        }
        let merged = AppStore.mergedProfiles(discovered: DiscoveredModelCatalog(profiles: baseProfiles))
        for camera in LoRACatalog.videoProfiles {
            #expect(merged.contains { $0.name == camera.name && $0.loras == camera.loras })
        }
        for id in LoRACatalog.ltxModelIDs {
            #expect(merged.filter { $0.modelID == id && $0.capability == .textToVideo && $0.loras.isEmpty }.count == 1)
        }
    }

    @Test func identicalDiscoveredCameraProfileIsNotDuplicated() throws {
        let camera = try #require(LoRACatalog.videoProfiles.first)
        let merged = AppStore.mergedProfiles(discovered: DiscoveredModelCatalog(profiles: [camera]))
        #expect(merged.filter {
            $0.modelID == camera.modelID && $0.capability == camera.capability && $0.loras == camera.loras
        }.count == 1)
    }
}
