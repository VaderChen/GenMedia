import Foundation

/// Catalog recommendations describe the complete runtime requirement, not a
/// sum of component weights (many runtimes load components in separate stages).
public enum ProfileVisibility {
    public static let maximumRecommendedMemoryGB = 64

    /// Resolve model aliases once when filtering the entire profile catalog.
    public static func visibleProfiles(
        _ profiles: [InferenceProfile], models: [ModelDescriptor]
    ) -> [InferenceProfile] {
        var blockedIDs = Set<String>()
        for model in models where model.recommendedMemoryGB > maximumRecommendedMemoryGB {
            blockedIDs.insert(model.id)
            if let localURL = model.localURL {
                blockedIDs.insert(localURL.standardizedFileURL.path)
            }
        }
        return profiles.filter { profile in
            !profile.requiredModelIDs.contains { blockedIDs.contains($0) }
        }
    }

    public static func isVisible(_ profile: InferenceProfile, models: [ModelDescriptor]) -> Bool {
        !profile.requiredModelIDs.contains { modelID in
            models.contains { model in
                let matches = model.id == modelID || model.localURL?.standardizedFileURL.path == modelID
                return matches && model.recommendedMemoryGB > maximumRecommendedMemoryGB
            }
        }
    }
}
