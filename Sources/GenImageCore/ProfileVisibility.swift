import Foundation

/// Catalog recommendations describe the complete runtime requirement, not a
/// sum of component weights (many runtimes load components in separate stages).
public enum ProfileVisibility {
    public static let maximumRecommendedMemoryGB = 64

    public static func isVisible(_ profile: InferenceProfile, models: [ModelDescriptor]) -> Bool {
        !profile.requiredModelIDs.contains { modelID in
            models.contains { model in
                let matches = model.id == modelID || model.localURL?.standardizedFileURL.path == modelID
                return matches && model.recommendedMemoryGB > maximumRecommendedMemoryGB
            }
        }
    }
}
