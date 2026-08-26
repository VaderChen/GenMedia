import Foundation
import GenImageCore

// Profile 的選用、相容性檢查與增刪改。
extension AppStore {
    func ensureInferenceIdle() -> Bool {
        guard !jobs.contains(where: { [.queued, .running, .cancelling].contains($0.state) }) else {
            statusMessage = "已有生成或辨識任務正在執行，請完成或取消後再開始新任務。"
            return false
        }
        return true
    }

    func selectProfile(_ profileID: UUID, for capability: ModelCapability) {
        guard let profile = profiles.first(where: { $0.id == profileID && $0.capability == capability }) else {
            return
        }
        let missingModels = missingProfileModels(profile)
        guard missingModels.isEmpty else {
            statusMessage = "請先下載並驗證「\(profile.name)」的相關模型：\(missingModels.map(\.displayName).joined(separator: "、"))。"
            return
        }
        if let compatibilityError = profileCompatibilityError(profile) {
            statusMessage = compatibilityError
            return
        }
        let previousProfileID = activeProfileIDs[capability]
        activeProfileIDs[capability] = nil
        disabledProfileIDs.remove(profileID)
        Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)
        activeProfileIDs[capability] = profileID
        Self.persistActiveProfiles(activeProfileIDs, in: profiles)

        if capability == .textToImage {
            recipe.profileID = profileID
            recipe.modelID = profile.modelID
        }
        if previousProfileID != profileID {
            statusMessage = "已啟用「\(profile.name)」；相容性檢查通過，同類型 Profile 維持互斥。"
            releaseNonFocusedModelsIfNeeded(focusing: capability)
        }
    }

    /// 切換 Profile 時避免多個大型 Runtime 同時常駐。記憶體讀值超過
    /// 90% 才執行釋放，而且保留目前切換到的能力所需 Runtime。
    private func releaseNonFocusedModelsIfNeeded(focusing capability: ModelCapability) {
        guard let ramUsage = SystemMetricsReader.read().ramUsage,
              ramUsage > 0.90 else {
            return
        }

        let usagePercent = Int((ramUsage * 100).rounded())
        let textToImage = textToImageService
        let imageToText = imageToTextService
        let upscale = upscaleService
        let subtitles = subtitleGenerationService

        Task { @MainActor [weak self, textToImage, imageToText, upscale, subtitles] in
            var released: [String] = []
            if capability != .textToImage {
                await textToImage.unload()
                released.append("文生圖")
            }
            if capability != .imageToText {
                await imageToText.unload()
                released.append("圖生文")
            }
            if capability != .upscale {
                await upscale.unload()
                released.append("Upscale")
            }
            if capability != .videoToText {
                await subtitles.unload()
                released.append("字幕生成")
            }
            guard !released.isEmpty, let self else { return }
            self.statusMessage = "記憶體使用率約 \(usagePercent)%；已釋放非焦點模型：\(released.joined(separator: "、"))。"
        }
    }

    private func profileCompatibilityError(
        _ profile: InferenceProfile,
        modelOverride: ModelDescriptor? = nil
    ) -> String? {
        guard let model = modelOverride ?? models.first(where: { $0.id == profile.modelID }) else {
            return "Profile「\(profile.name)」找不到指定模型。"
        }
        guard let localURL = model.localURL,
              FileManager.default.fileExists(atPath: localURL.path) else {
            return "Profile「\(profile.name)」的模型路徑不存在。"
        }
        guard model.capabilities.contains(profile.capability) else {
            return "Profile「\(profile.name)」的模型不支援「\(profile.capability.title)」。"
        }

        let compatibleArchitectures: Set<InferenceArchitecture>?
        switch profile.capability {
        case .textToImage, .imageToText, .textToText:
            compatibleArchitectures = [.mlxSwift]
        case .upscale:
            compatibleArchitectures = [.coreML]
        case .imageToImage, .imageToVideo, .textToVideo:
            compatibleArchitectures = [.externalCLI]
        case .textToMusic:
            compatibleArchitectures = [.mlxSwift, .externalCLI]
        case .videoToText:
            compatibleArchitectures = [.coreML]
        case .controlNet, .lora:
            compatibleArchitectures = nil
        }
        if let compatibleArchitectures,
           !compatibleArchitectures.contains(profile.architecture) {
            let architectureNames = compatibleArchitectures
                .map(\.title)
                .sorted()
                .joined(separator: "、")
            return "Profile「\(profile.name)」的架構「\(profile.architecture.title)」與目前 Runtime 不相容；需要「\(architectureNames)」。"
        }

        for configuration in profile.loras {
            guard let loraModel = models.first(where: { $0.id == configuration.modelID }),
                  let loraURL = loraModel.localURL else {
                return "Profile「\(profile.name)」找不到 LoRA「\(configuration.modelID)」。"
            }
            if let error = loraCompatibilityError(at: loraURL) {
                return "Profile「\(profile.name)」的 LoRA 不相容：\(error)"
            }
        }
        return nil
    }

    func loraCompatibilityError(at url: URL) -> String? {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return "找不到權重路徑：\(url.path)"
        }

        let candidates: [URL]
        if isDirectory.boolValue {
            candidates = (FileManager.default.enumerator(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )?.compactMap { $0 as? URL }.filter { candidate in
                (try? candidate.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
            }) ?? []
        } else {
            candidates = [url]
        }

        var sawReadableHeader = false
        for candidate in candidates {
            guard let keys = safetensorHeaderKeys(at: candidate) else { continue }
            sawReadableHeader = true
            if hasLoRAPairs(keys) { return nil }
        }
        return sawReadableHeader
            ? "找不到可套用的 LoRA 權重配對（A/B 或 LoKr）。"
            : "無法讀取可辨識的權重標頭。"
    }

    private func safetensorHeaderKeys(at url: URL) -> [String]? {
        guard let data = try? Data(contentsOf: url, options: [.mappedIfSafe]), data.count >= 8 else {
            return nil
        }
        var headerLength: UInt64 = 0
        for (index, byte) in data.prefix(8).enumerated() {
            headerLength |= UInt64(byte) << UInt64(index * 8)
        }
        guard headerLength <= UInt64(data.count - 8), headerLength <= UInt64(Int.max) else {
            return nil
        }
        let start = data.index(data.startIndex, offsetBy: 8)
        let end = data.index(start, offsetBy: Int(headerLength))
        guard let header = try? JSONSerialization.jsonObject(with: data[start..<end]) as? [String: Any] else {
            return nil
        }
        return header.keys.filter { $0 != "__metadata__" }
    }

    private func hasLoRAPairs(_ keys: [String]) -> Bool {
        let suffixPairs = [(".lora_down", ".lora_up"), (".lora_A", ".lora_B")]
        for (downSuffix, upSuffix) in suffixPairs {
            let downBases = Set(keys.compactMap { loraBase($0, suffix: downSuffix) })
            let upBases = Set(keys.compactMap { loraBase($0, suffix: upSuffix) })
            if !downBases.isDisjoint(with: upBases) { return true }
        }
        let w1Bases = Set(keys.compactMap { loraBase($0, suffix: ".lokr_w1") })
        let w2Bases = Set(keys.compactMap { loraBase($0, suffix: ".lokr_w2") })
        return !w1Bases.isDisjoint(with: w2Bases)
    }

    private func loraBase(_ key: String, suffix: String) -> String? {
        let rawKey = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
        guard rawKey.hasSuffix(suffix) else { return nil }
        return String(rawKey.dropLast(suffix.count))
    }

    func deactivateProfiles(usingModelID modelID: String) {
        var changed = false
        for profile in profiles where profile.requiredModelIDs.contains(modelID) {
            guard activeProfileIDs[profile.capability] == profile.id else { continue }
            activeProfileIDs[profile.capability] = nil
            disabledProfileIDs.insert(profile.id)
            if profile.capability == .textToImage {
                recipe.profileID = nil
            }
            changed = true
        }
        if changed {
            Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)
            Self.persistActiveProfiles(activeProfileIDs, in: profiles)
        }
    }

    func deactivateProfile(_ profileID: UUID, for capability: ModelCapability) {
        guard activeProfileIDs[capability] == profileID,
              let profile = profiles.first(where: { $0.id == profileID && $0.capability == capability }) else {
            return
        }
        disabledProfileIDs.insert(profileID)
        Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)
        activeProfileIDs[capability] = nil
        Self.persistActiveProfiles(activeProfileIDs, in: profiles)
        if capability == .textToImage {
            recipe.profileID = nil
        }
        let hasAvailableProfile = profiles.contains {
            $0.capability == capability && !disabledProfileIDs.contains($0.id)
        }
        statusMessage = hasAvailableProfile
            ? "已停用「\(profile.name)」；左側工具已隱藏，可到 Profiles 頁啟用其他 Profile。"
            : "已停用「\(profile.name)」；此類型已全部停用，左側工具已隱藏。"
    }

    func duplicateProfile(_ profile: InferenceProfile) {
        let copy = InferenceProfile(
            name: "\(profile.name) 副本",
            capability: profile.capability,
            modelID: profile.modelID,
            modelRevision: profile.modelRevision,
            architecture: profile.architecture,
            defaults: profile.defaults,
            loras: profile.loras,
            profileRevision: 1,
            notes: profile.notes,
            isBuiltIn: false
        )
        profiles.append(copy)
        selectProfile(copy.id, for: copy.capability)
    }

    func createProfile(for capability: ModelCapability) {
        let isVideoCapability = capability == .imageToVideo || capability == .textToVideo
        let isMusicCapability = capability == .textToMusic
        let model = models.first(where: { $0.capabilities.contains(capability) })
            ?? (capability == .imageToImage
                ? models.first(where: { $0.capabilities.contains(.textToImage) })
                : nil)
        let templateProfile = profiles.first { $0.capability == capability }
        guard let modelID = model?.id ?? templateProfile?.modelID else {
            statusMessage = "找不到可作為 \(capability.title) Profile 初值的模型。"
            return
        }
        let defaults: ProfileDefaults
        if capability == .imageToImage {
            defaults = ProfileDefaults(
                width: recipe.width,
                height: recipe.height,
                steps: recipe.steps,
                outputCount: recipe.outputCount
            )
        } else if (isVideoCapability || isMusicCapability), let templateProfile {
            defaults = templateProfile.defaults
        } else {
            defaults = ProfileDefaults()
        }
        let architecture: InferenceArchitecture
        if (isVideoCapability || isMusicCapability), let templateProfile {
            architecture = templateProfile.architecture
        } else if capability == .imageToImage {
            architecture = .externalCLI
        } else {
            architecture = model?.quantization == .coreML ? .coreML : .mlxSwift
        }
        let profile = InferenceProfile(
            name: "新的 \(capability.title) Profile",
            capability: capability,
            modelID: modelID,
            architecture: architecture,
            defaults: defaults,
            loras: isVideoCapability ? (templateProfile?.loras ?? []) : [],
            notes: capability == .imageToImage || isVideoCapability || isMusicCapability
                ? "請設定支援此生成能力的模型版本與推論架構。"
                : "",
            isBuiltIn: false
        )
        profiles.append(profile)
        selectProfile(profile.id, for: capability)
    }

    func updateProfile(
        id: UUID,
        name: String,
        modelID: String,
        modelRevision: String,
        architecture: InferenceArchitecture,
        loras: [ProfileLoRAConfiguration]
    ) {
        guard let index = profiles.firstIndex(where: { $0.id == id }), !profiles[index].isBuiltIn else {
            return
        }

        guard loras.allSatisfy({ configuration in
            !configuration.modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && configuration.scale.isFinite
                && (0...1).contains(configuration.scale)
                && configuration.conditioningScale.isFinite
                && (0...1).contains(configuration.conditioningScale)
        }) else {
            statusMessage = "LoRA 模型 ID 不可空白，權重與控制強度必須介於 0 到 1。"
            return
        }

        var candidate = profiles[index]
        candidate.name = name
        candidate.modelID = modelID
        candidate.modelRevision = modelRevision
        candidate.architecture = architecture
        candidate.loras = loras
        candidate.profileRevision += 1

        // 編輯中的 Profile 尚未寫回陣列，將候選模型明確傳入檢查，避免
        // 使用中的 Profile 被更新成不存在或不相容的模型／架構。
        let candidateModel = models.first(where: { $0.id == modelID })
        if let compatibilityError = profileCompatibilityError(candidate, modelOverride: candidateModel) {
            statusMessage = compatibilityError
            return
        }

        profiles[index] = candidate
        Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)

        let capability = profiles[index].capability
        if activeProfileIDs[capability] == id {
            let isInstalled = missingProfileModels(profiles[index]).isEmpty
            if isInstalled {
                if capability == .textToImage {
                    recipe.modelID = modelID
                }
            } else {
                activeProfileIDs[capability] = nil
                if capability == .textToImage {
                    recipe.profileID = nil
                }
                statusMessage = "Profile 已更新；新模型尚未安裝，因此未設為使用中。"
            }
        }
        Self.persistActiveProfiles(activeProfileIDs, in: profiles)
    }

    func deleteProfile(_ id: UUID) {
        guard let profile = profiles.first(where: { $0.id == id }), !profile.isBuiltIn else { return }
        disabledProfileIDs.remove(id)
        profiles.removeAll { $0.id == id }
        Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)

        if activeProfileIDs[profile.capability] == id {
            let replacement = profiles.first { candidate in
                candidate.capability == profile.capability
                    && !disabledProfileIDs.contains(candidate.id)
                    && missingProfileModels(candidate).isEmpty
            }
            activeProfileIDs[profile.capability] = replacement?.id
            if profile.capability == .textToImage {
                recipe.profileID = replacement?.id
                if let replacement {
                    recipe.modelID = replacement.modelID
                }
            }
        }
        Self.persistActiveProfiles(activeProfileIDs, in: profiles)
    }

    func isProfileReady(_ profile: InferenceProfile) -> Bool {
        let missingModels = missingProfileModels(profile)
        guard missingModels.isEmpty else {
            statusMessage = "請先安裝 Profile 的相關模型：\(missingModels.map(\.displayName).joined(separator: "、"))。"
            return false
        }
        return true
    }

    func isProfileReadyWithoutStatus(_ profile: InferenceProfile) -> Bool {
        missingProfileModels(profile).isEmpty
    }

    private func missingProfileModels(_ profile: InferenceProfile) -> [ModelDescriptor] {
        profile.requiredModelIDs.compactMap { requiredModelID in
            guard let model = models.first(where: { $0.id == requiredModelID }) else {
                return ModelDescriptor(
                    id: requiredModelID,
                    displayName: requiredModelID,
                    publisher: "",
                    summary: "",
                    capabilities: [],
                    quantization: .lora,
                    approximateDownloadGB: 0,
                    recommendedMemoryGB: 0,
                    licenseName: ""
                )
            }
            guard model.localURL != nil,
                  installation(for: requiredModelID).phase == .installed else {
                return model
            }
            return nil
        }
    }

    func resolvedVideoLoRAs(for profile: InferenceProfile) -> [VideoGenerationLoRA]? {
        var result: [VideoGenerationLoRA] = []
        for configuration in profile.loras {
            guard let model = models.first(where: { $0.id == configuration.modelID }),
                  let localURL = model.localURL,
                  FileManager.default.fileExists(atPath: localURL.path) else {
                statusMessage = "找不到 Profile LoRA 的本機檔案：\(configuration.modelID)"
                return nil
            }
            result.append(VideoGenerationLoRA(configuration: configuration, localURL: localURL))
        }
        return result
    }

    func runtimeProfile(from profile: InferenceProfile) -> InferenceProfile? {
        guard let model = models.first(where: { $0.id == profile.modelID }) else {
            statusMessage = "找不到 Profile 指定的模型：\(profile.modelID)"
            return nil
        }
        guard let localURL = model.localURL,
              FileManager.default.fileExists(atPath: localURL.path) else {
            statusMessage = "找不到「\(model.displayName)」的本機 Runtime 路徑。"
            return nil
        }
        var executionProfile = profile
        executionProfile.modelID = localURL.path
        return executionProfile
    }
}
