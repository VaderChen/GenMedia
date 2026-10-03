import Testing
@testable import GenImageCore

struct MiniMaxH3AccelerationCatalogTests {
    @Test func lowStepProfilesResolveToCompatibleBasesAndSamplingDefaults() throws {
        let profiles = MiniMaxH3Acceleration.allCases.flatMap(\.profiles)
        #expect(profiles.count == 11)
        #expect(LoRACatalog.videoProfiles.count == 19)
        for acceleration in MiniMaxH3Acceleration.allCases {
            let entry = try #require(LoRACatalog.entry(for: acceleration.modelID))
            #expect(!entry.repository.contains("@"))
            #expect(entry.descriptor.sourceURL?.absoluteString == "https://huggingface.co/" + entry.repository)
            for profile in acceleration.profiles {
                #expect(entry.supports(modelID: profile.modelID, capability: .textToVideo))
                #expect(profile.defaults.steps == acceleration.defaultSteps)
                #expect(profile.loras.first?.scale == 1)
                #expect(ModelCatalog.builtInProfiles.contains { $0.name == profile.name })
            }
            #expect(!entry.supports(modelID: LoRACatalog.ltxModelIDs[0], capability: .textToVideo))
            #expect(!entry.supports(modelID: MiniMaxH3Acceleration.fullModelIDs[0], capability: .imageToVideo))
            #expect(!entry.supports(modelID: "Abiray/MiniMax-H3-GGUF@ref2va-Q4_0", capability: .textToVideo))
        }
    }

    @Test func onlyAttentionAndMLPAdaptersCanUsePrunedTimeConditioning() {
        let pruned = MiniMaxH3Acceleration.prunedModelID
        #expect(MiniMaxH3Acceleration.lightX2V4Step768.entry.supports(modelID: pruned, capability: .textToVideo))
        #expect(MiniMaxH3Acceleration.lightX2V8Step768.entry.supports(modelID: pruned, capability: .textToVideo))
        #expect(!MiniMaxH3Acceleration.larryV4.entry.supports(modelID: pruned, capability: .textToVideo))
        #expect(MiniMaxH3Acceleration.lightX2V4Step768.allowedSteps == 4...4)
        #expect(MiniMaxH3Acceleration.lightX2V8Step768.allowedSteps == 8...8)
        #expect(MiniMaxH3Acceleration.larryV4.allowedSteps == 4...8)
        #expect(MiniMaxH3Acceleration.larryV4.defaultSteps == 6)
    }
}
