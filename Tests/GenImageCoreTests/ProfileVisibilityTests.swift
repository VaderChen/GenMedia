import Foundation
import Testing
@testable import GenImageCore

struct ProfileVisibilityTests {
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
