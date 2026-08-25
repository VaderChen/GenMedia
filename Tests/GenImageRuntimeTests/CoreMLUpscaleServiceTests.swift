import Foundation
import GenImageCore
import GenImageRuntime
import Testing

struct CoreMLUpscaleServiceTests {
    @Test func installerSupportsEveryCatalogDownload() {
        #expect(ModelCatalog.builtIn.allSatisfy {
            HuggingFaceModelInstaller.supports(modelID: $0.id)
        })
    }

    @Test func rejectsNonUpscaleProfilesBeforeLoadingModel() async {
        let projectID = UUID()
        let asset = MediaAsset(
            projectID: projectID,
            kind: .imported,
            title: "fixture",
            fileURL: URL(fileURLWithPath: "/tmp/missing.png"),
            pixelWidth: 512,
            pixelHeight: 512
        )
        let profile = InferenceProfile(
            name: "wrong",
            capability: .textToImage,
            modelID: "/tmp/missing.mlmodel",
            architecture: .coreML
        )
        let service = CoreMLUpscaleService(outputDirectory: FileManager.default.temporaryDirectory)

        await #expect(throws: CoreMLUpscaleError.self) {
            _ = try await service.upscale(
                request: UpscaleRequest(asset: asset, profile: profile),
                progress: { _ in }
            )
        }
    }
}
