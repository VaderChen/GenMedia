import Foundation
import GenImageCore

/// Keep the selected model identity before AppStore rewrites the profile's modelID to a local path.
public actor ImageGenerationRouter: TextToImageGenerating, ImageToImageGenerating {
    private nonisolated let zImage: ZImageTextToImageService
    private nonisolated let qwen2511: Qwen2511ImageToImageService
    private nonisolated let qwen21: Qwen21ImageService

    public init(outputDirectory: URL) {
        zImage = ZImageTextToImageService(outputDirectory: outputDirectory)
        qwen2511 = Qwen2511ImageToImageService(outputDirectory: outputDirectory)
        qwen21 = Qwen21ImageService(outputDirectory: outputDirectory)
    }
    public nonisolated func setOutputDirectory(_ url: URL) {
        zImage.setOutputDirectory(url); qwen2511.setOutputDirectory(url); qwen21.setOutputDirectory(url)
    }
    public func unload() async { await zImage.unload() }

    static func isQwen21(recipe: GenerationRecipe, profile: InferenceProfile) -> Bool {
        profile.modelID == QwenImage21Model.id
            || (NSString(string: profile.modelID).isAbsolutePath
                && (QwenImage21Model.isModelDirectory(URL(fileURLWithPath: profile.modelID))
                    || recipe.modelID == QwenImage21Model.id))
    }
    public func generate(request: TextToImageRequest, progress: @escaping @Sendable (Double) -> Void) async throws -> [MediaAsset] {
        if Self.isQwen21(recipe: request.recipe, profile: request.profile) {
            await zImage.unload()
            return try await qwen21.generate(request: request, progress: progress)
        }
        return try await zImage.generate(request: request, progress: progress)
    }
    public func generate(request: ImageToImageRequest, progress: @escaping @Sendable (Double) -> Void) async throws -> MediaAsset {
        if Self.isQwen21(recipe: request.recipe, profile: request.profile) {
            await zImage.unload()
            return try await qwen21.generate(request: request, progress: progress)
        }
        return try await qwen2511.generate(request: request, progress: progress)
    }
}
