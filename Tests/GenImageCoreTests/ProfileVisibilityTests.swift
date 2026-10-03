import Foundation
import Testing
@testable import GenImageCore

struct ProfileVisibilityTests {
    @Test func batchFilteringMatchesIndividualRulesAndKeepsOrder() {
        var large = ModelCatalog.builtIn[0]
        large.recommendedMemoryGB = 96
        large.localURL = URL(fileURLWithPath: "/tmp/models/../large")
        var small = large
        small.recommendedMemoryGB = 64
        let models = [small, large] + ModelCatalog.builtIn
        var profiles = ModelCatalog.builtInProfiles
        for id in [large.id, "/tmp/large", "unknown"] {
            profiles.append(InferenceProfile(name: id, capability: .textToImage,
                modelID: id, architecture: .mlxSwift))
        }
        var enhanced = profiles.last!
        enhanced.promptEnhancerModelID = large.id
        profiles.append(enhanced)
        let expected = profiles.filter { ProfileVisibility.isVisible($0, models: models) }
        #expect(ProfileVisibility.visibleProfiles(profiles, models: models) == expected)
        #expect(ProfileVisibility.visibleProfiles(profiles, models: []).count == profiles.count)
        #expect(ProfileVisibility.visibleProfiles([], models: models).isEmpty)
    }

    @Test func keeps64GBAndUnknownModelsButHidesLargerModelsAndLocalPaths() {
        var model = ModelCatalog.builtIn[0]
        model.localURL = URL(fileURLWithPath: "/tmp/local-model")
        var profile = InferenceProfile(name: "test", capability: .textToImage,
            modelID: model.id, architecture: .mlxSwift)
        model.recommendedMemoryGB = 64
        #expect(ProfileVisibility.isVisible(profile, models: [model]))
        model.recommendedMemoryGB = 96
        #expect(!ProfileVisibility.isVisible(profile, models: [model]))
        profile.modelID = "/tmp/local-model"
        #expect(!ProfileVisibility.isVisible(profile, models: [model]))
        profile.modelID = "unknown"
        #expect(ProfileVisibility.isVisible(profile, models: [model]))
    }
}
