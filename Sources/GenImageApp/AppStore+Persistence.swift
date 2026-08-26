import Foundation
import GenImageCore

// 讀寫 UserDefaults 的設定持久化，以及還原時的驗證。全部是純函式，不碰 AppStore 的狀態。
extension AppStore {
    static func loadRecipeSettings() -> PersistedRecipeSettings? {
        guard let data = UserDefaults.standard.data(forKey: recipeSettingsKey) else { return nil }
        return try? JSONDecoder().decode(PersistedRecipeSettings.self, from: data)
    }

    static func persistRecipeSettings(_ recipe: GenerationRecipe) {
        guard let data = try? JSONEncoder().encode(PersistedRecipeSettings(recipe: recipe)) else { return }
        UserDefaults.standard.set(data, forKey: recipeSettingsKey)
    }

    static func loadVideoOutputSettings() -> VideoOutputSettings? {
        guard let data = UserDefaults.standard.data(forKey: videoOutputSettingsKey) else { return nil }
        return try? JSONDecoder().decode(VideoOutputSettings.self, from: data)
    }

    static func persistVideoOutputSettings(_ settings: VideoOutputSettings) {
        guard let data = try? JSONEncoder().encode(settings) else { return }
        UserDefaults.standard.set(data, forKey: videoOutputSettingsKey)
    }

    static func loadMusicOutputSettings() -> MusicOutputSettings? {
        guard let data = UserDefaults.standard.data(forKey: musicOutputSettingsKey) else { return nil }
        return try? JSONDecoder().decode(MusicOutputSettings.self, from: data)
    }

    static func persistMusicOutputSettings(_ settings: MusicOutputSettings) {
        guard let data = try? JSONEncoder().encode(settings) else { return }
        UserDefaults.standard.set(data, forKey: musicOutputSettingsKey)
    }

    static func mergedModels(discovered: DiscoveredModelCatalog) -> [ModelDescriptor] {
        let builtInModels = ModelCatalog.builtIn.filter { model in
            if discovered.models.contains(where: { $0.id == model.id }) {
                return false
            }
            if model.id == "local-captioner-3b@q4" {
                return !discovered.models.contains { $0.capabilities.contains(.imageToText) }
            }
            if model.id == "qwen3-vl-8b-nsfw-caption-v45@mxfp4" {
                return !discovered.models.contains { $0.id == model.id }
            }
            if model.id == "realesrgan-x4@coreml" {
                return !discovered.models.contains {
                    $0.capabilities.contains(.upscale) && $0.displayName.contains("Real-ESRGAN 4×")
                }
            }
            if model.id == "realesrgan-x2@coreml" {
                return !discovered.models.contains {
                    $0.capabilities.contains(.upscale) && $0.displayName.contains("Real-ESRGAN 2×")
                }
            }
            return true
        }
        return discovered.models + builtInModels
    }

    static func mergedProfiles(discovered: DiscoveredModelCatalog) -> [InferenceProfile] {
        let builtInProfiles = ModelCatalog.builtInProfiles.filter { profile in
            if discovered.profiles.contains(where: {
                $0.modelID == profile.modelID && $0.capability == profile.capability
            }) {
                return false
            }
            if profile.modelID == "local-captioner-3b@q4" {
                return !discovered.profiles.contains { $0.capability == .imageToText }
            }
            if profile.modelID == "qwen3-vl-8b-nsfw-caption-v45@mxfp4" {
                return !discovered.profiles.contains { $0.modelID == profile.modelID }
            }
            if profile.modelID == "realesrgan-x4@coreml" {
                return !discovered.profiles.contains {
                    $0.capability == .upscale && $0.defaults.upscaleScale == 4
                }
            }
            if profile.modelID == "realesrgan-x2@coreml" {
                return !discovered.profiles.contains {
                    $0.capability == .upscale && $0.defaults.upscaleScale == 2
                }
            }
            return true
        }
        return discovered.profiles + builtInProfiles
    }

    static func installations(
        for models: [ModelDescriptor],
        preserving previous: [String: ModelInstallation] = [:]
    ) -> [String: ModelInstallation] {
        Dictionary(
            uniqueKeysWithValues: models.map { model in
                if model.localURL != nil {
                    return (
                        model.id,
                        ModelInstallation(
                            phase: .installed,
                            progress: 1,
                            downloadedGB: model.approximateDownloadGB
                        )
                    )
                }
                if previous[model.id]?.phase == .installed {
                    return (model.id, ModelInstallation())
                }
                return (model.id, previous[model.id] ?? ModelInstallation())
            }
        )
    }

    private static func profileSignature(_ profile: InferenceProfile) -> String {
        [
            profile.capability.rawValue,
            profile.modelID,
            profile.modelRevision,
            profile.name
        ].joined(separator: "\u{1F}")
    }

    private static func disabledProfileSignatures() -> Set<String> {
        Set(UserDefaults.standard.stringArray(forKey: disabledProfilesKey) ?? [])
    }

    static func disabledProfileIDs(in profiles: [InferenceProfile]) -> Set<UUID> {
        let signatures = disabledProfileSignatures()
        return Set(profiles.filter { signatures.contains(profileSignature($0)) }.map(\.id))
    }

    static func persistDisabledProfiles(
        _ disabledProfileIDs: Set<UUID>,
        in profiles: [InferenceProfile]
    ) {
        let signatures = profiles
            .filter { disabledProfileIDs.contains($0.id) }
            .map(profileSignature)
            .sorted()
        UserDefaults.standard.set(
            signatures,
            forKey: disabledProfilesKey
        )
    }

    static func persistedActiveProfileIDs(
        in profiles: [InferenceProfile],
        models: [ModelDescriptor]
    ) -> [ModelCapability: UUID] {
        guard let signatures = UserDefaults.standard.dictionary(forKey: activeProfilesKey)
            as? [String: String] else { return [:] }

        return Dictionary(
            uniqueKeysWithValues: ModelCapability.allCases.compactMap { capability in
                guard let signature = signatures[capability.rawValue],
                      let profile = profiles.first(where: {
                          $0.capability == capability && profileSignature($0) == signature
                      }),
                      profile.requiredModelIDs.allSatisfy({ requiredModelID in
                          models.contains(where: {
                              $0.id == requiredModelID && $0.localURL != nil
                          })
                      }) else { return nil }
                return (capability, profile.id)
            }
        )
    }

    static func persistActiveProfiles(
        _ activeProfileIDs: [ModelCapability: UUID],
        in profiles: [InferenceProfile]
    ) {
        let signaturePairs: [(String, String)] = activeProfileIDs.compactMap { entry in
            let (capability, profileID) = entry
            guard let profile = profiles.first(where: {
                $0.id == profileID && $0.capability == capability
            }) else { return nil }
            return (capability.rawValue, profileSignature(profile))
        }
        let signatures = Dictionary(uniqueKeysWithValues: signaturePairs)
        UserDefaults.standard.set(signatures, forKey: activeProfilesKey)
    }

    static func persistedDimension(_ value: Int?, fallback: Int) -> Int {
        guard let value, OutputGeometry.isSupported(value) else {
            return fallback
        }
        return value
    }

    static func validatedVideoOutputSettings(
        _ saved: VideoOutputSettings?,
        defaults: ProfileDefaults
    ) -> VideoOutputSettings {
        VideoOutputSettings(
            width: persistedDimension(saved?.width, fallback: defaults.width ?? 1280),
            height: persistedDimension(saved?.height, fallback: defaults.height ?? 720),
            steps: persistedValue(saved?.steps, range: 1...100, fallback: defaults.steps ?? 8),
            outputCount: persistedValue(
                saved?.outputCount,
                range: 1...8,
                fallback: defaults.outputCount ?? 1
            ),
            frameCount: persistedValue(
                saved?.frameCount,
                range: 1...512,
                fallback: defaults.frameCount ?? 97
            ),
            frameRate: persistedValue(
                saved?.frameRate,
                range: 1...120,
                fallback: defaults.frameRate ?? 24
            ),
            seed: saved?.seed ?? UInt64.random(in: 0...UInt64.max)
        )
    }

    static func validatedMusicOutputSettings(
        _ saved: MusicOutputSettings?,
        defaults: ProfileDefaults
    ) -> MusicOutputSettings {
        MusicOutputSettings(
            prompt: saved?.prompt ?? "",
            lyrics: saved?.lyrics ?? "",
            style: saved?.style ?? .pop,
            durationSeconds: persistedValue(
                saved?.durationSeconds,
                range: MusicGenerationOptions.supportedDurationSeconds,
                fallback: defaults.durationSeconds ?? 10
            ),
            steps: persistedValue(
                saved?.steps,
                range: 1...100,
                fallback: defaults.steps ?? 20
            ),
            seed: saved?.seed ?? UInt64.random(in: 0...UInt64.max),
            format: saved?.format ?? .mp3
        )
    }

    static func validatedPersistedLoRA(
        _ selection: LoRASelection?,
        available: [LoRADescriptor]
    ) -> LoRASelection? {
        guard let selection,
              selection.scale.isFinite,
              (0...1).contains(selection.scale),
              let descriptor = available.first(where: { $0.id == selection.adapterID }) else {
            return nil
        }
        return LoRASelection(
            adapterID: descriptor.id,
            localURL: descriptor.localURL,
            scale: selection.scale
        )
    }

    static func persistedValue(
        _ value: Int?,
        range: ClosedRange<Int>,
        fallback: Int
    ) -> Int {
        guard let value else { return fallback }
        return min(max(value, range.lowerBound), range.upperBound)
    }
}
