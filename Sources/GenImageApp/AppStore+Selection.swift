import Foundation
import GenImageCore

// 目前選取狀態衍生出來的唯讀資料。
extension AppStore {
    var selectedProject: Project? {
        projects.first { $0.id == selectedProjectID }
    }

    var projectAssets: [MediaAsset] {
        assets
            .filter { $0.projectID == selectedProjectID }
            .sorted { $0.createdAt < $1.createdAt }
    }

    var selectedAsset: MediaAsset? {
        guard let selectedAssetID else { return nil }
        return assets.first { $0.id == selectedAssetID }
    }

    var selectedSourceImage: MediaAsset? {
        guard let asset = selectedAsset,
              asset.kind.isImage,
              let fileURL = asset.fileURL,
              FileManager.default.fileExists(atPath: fileURL.path) else {
            return nil
        }
        return asset
    }

    func sourceImages(for assetIDs: [UUID]) -> [MediaAsset] {
        var seen = Set<UUID>()
        return assetIDs.compactMap { assetID in
            guard seen.insert(assetID).inserted,
                  let asset = assets.first(where: { $0.id == assetID }),
                  asset.kind.isImage,
                  let fileURL = asset.fileURL,
                  FileManager.default.fileExists(atPath: fileURL.path) else {
                return nil
            }
            return asset
        }
    }

    var comparisonAsset: MediaAsset? {
        guard let comparisonAssetID else { return nil }
        return assets.first { $0.id == comparisonAssetID }
    }

    var filteredModels: [ModelDescriptor] {
        models.filter { model in
            let matchesFilter = modelFilter.capability.map { model.capabilities.contains($0) } ?? true
            let search = modelSearch.trimmingCharacters(in: .whitespacesAndNewlines)
            let matchesSearch = search.isEmpty
                || model.displayName.localizedCaseInsensitiveContains(search)
                || model.publisher.localizedCaseInsensitiveContains(search)
            return matchesFilter && matchesSearch
        }
    }

    func profiles(for capability: ModelCapability) -> [InferenceProfile] {
        profiles.filter { $0.capability == capability }
    }

    func activeProfile(for capability: ModelCapability) -> InferenceProfile? {
        guard let id = activeProfileIDs[capability] else { return nil }
        return profiles.first { $0.id == id }
    }

    var preferredVideoProfile: InferenceProfile? {
        if selectedSourceImage != nil, let profile = activeProfile(for: .imageToVideo) {
            return profile
        }
        return activeProfile(for: .textToVideo) ?? activeProfile(for: .imageToVideo)
    }
}
