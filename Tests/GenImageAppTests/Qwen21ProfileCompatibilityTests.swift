import GenImageCore
import Testing
@testable import GenImageApp

struct Qwen21ProfileCompatibilityTests {
    @Test func qwen21BuiltInProfilesAllowNativeMLX() {
        let profiles = QwenImage21Model.profiles
        #expect(Set(profiles.map(\.capability)) == [.textToImage, .imageToImage])
        for profile in profiles {
            #expect(profile.architecture == .mlxSwift)
            #expect(AppStore.profileArchitectureCompatibilityError(profile) == nil)
        }
    }

    @Test func qwen2511BuiltInProfilesStillAllowExternalCLI() {
        let profiles = ModelCatalog.builtInProfiles.filter {
            $0.modelID.hasPrefix("qwen-image-edit-2511@") && $0.capability == .imageToImage
        }
        #expect(profiles.count == 3)
        for profile in profiles {
            #expect(profile.architecture == .externalCLI)
            #expect(AppStore.profileArchitectureCompatibilityError(profile) == nil)
        }
    }

    @Test func imageEditingRejectsCoreML() throws {
        var profile = try #require(QwenImage21Model.profiles.first { $0.capability == .imageToImage })
        profile.architecture = .coreML
        #expect(AppStore.profileArchitectureCompatibilityError(profile) != nil)
    }
}
