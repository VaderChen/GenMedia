import Foundation
import GenImageCore
import Testing
@testable import GenImageApp

@MainActor
struct LoRAProfileTests {
    @Test func workspaceRestoresTheSelectedVariantBeforeMatchingItsBaseModel() throws {
        let profiles = ModelCatalog.builtInProfiles.filter { $0.modelID == QwenImage21Model.id }
        let selected = try #require(profiles.first {
            $0.capability == .textToImage && $0.promptEnhancerModelID != nil && !$0.loras.isEmpty
        })
        var assignment: [String: Any] = ["profileID": selected.id.uuidString,
            "capability": selected.capability.rawValue, "modelID": selected.modelID,
            "modelRevision": selected.modelRevision, "architecture": selected.architecture.rawValue]
        #expect(AppStore.profileForWorkspaceAssignment(assignment, profiles: profiles)?.id == selected.id)
        assignment["profileID"] = UUID().uuidString
        #expect(AppStore.profileForWorkspaceAssignment(assignment, profiles: profiles) == nil)
        assignment["profileName"] = selected.name
        #expect(AppStore.profileForWorkspaceAssignment(assignment, profiles: profiles)?.id == selected.id)
        assignment["capability"] = ModelCapability.textToVideo.rawValue
        #expect(AppStore.profileForWorkspaceAssignment(assignment, profiles: profiles) == nil)
    }

    @Test func installedQwenBaseKeepsTurboAndEnhancementProfilesDistinct() {
        let merged = AppStore.mergedProfiles(discovered: DiscoveredModelCatalog(profiles: QwenImage21Model.profiles))
            .filter { $0.modelID == QwenImage21Model.id }
        #expect(merged.count == 8)
        #expect(merged.filter { $0.promptEnhancerModelID != nil }.count == 4)
        #expect(merged.filter { !$0.loras.isEmpty }.count == 4)
        for profile in merged {
            let rediscovered = AppStore.mergedProfiles(discovered: DiscoveredModelCatalog(profiles: [profile]))
                .filter { $0.modelID == profile.modelID && $0.capability == profile.capability
                    && $0.loras == profile.loras && $0.promptEnhancerModelID == profile.promptEnhancerModelID }
            #expect(rediscovered.count == 1)
        }
    }

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
