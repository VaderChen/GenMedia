import Foundation
import GenImageCore
import Testing
@testable import GenImageRuntime

struct Qwen21RoutingTests {
    @Test func installerRecognizesPinnedModelAndRejectsIncompleteInstallation() throws {
        #expect(HuggingFaceModelInstaller.supports(modelID: QwenImage21Model.id))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        #expect(HuggingFaceModelInstaller.installationDirectory(modelID: QwenImage21Model.id, rootURL: root)
            == root.appendingPathComponent(QwenImage21Model.directoryName, isDirectory: true))
        #expect(throws: (any Error).self) {
            try HuggingFaceModelInstaller.verify(modelID: QwenImage21Model.id, rootURL: root)
        }
    }

    @Test func routesAfterProfileModelIDBecomesLocalPath() {
        var profile = QwenImage21Model.profiles[0]
        profile.modelID = "/Models/custom-directory"
        let recipe = GenerationRecipe(prompt: "test", modelID: QwenImage21Model.id)
        #expect(ImageGenerationRouter.isQwen21(recipe: recipe, profile: profile))
        var other = recipe
        other.modelID = "mzbac/z-image-turbo-8bit"
        #expect(!ImageGenerationRouter.isQwen21(recipe: other, profile: profile))
    }

    @Test func rejectsNegativePromptWithoutLaunchingWorker() async throws {
        let service = Qwen21ImageService(outputDirectory: URL(fileURLWithPath: "/missing"),
            workerExecutable: URL(fileURLWithPath: "/missing-worker"))
        let profile = QwenImage21Model.profiles[0]
        let recipe = GenerationRecipe(prompt: "test", negativePrompt: "blur", modelID: QwenImage21Model.id)
        do {
            _ = try await service.generate(request: TextToImageRequest(projectID: UUID(), recipe: recipe, profile: profile), progress: { _ in })
            Issue.record("Expected unsupported option error")
        } catch {
            #expect(error.localizedDescription.contains("負面提示詞"))
        }
    }
}
