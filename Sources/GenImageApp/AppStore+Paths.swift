import AppKit
import Foundation
import GenImageCore
import GenImageRuntime

// 模型根目錄、輸出目錄與系統資源監看。
extension AppStore {
    @discardableResult
    func setOutputDirectory(_ rawPath: String) -> Bool {
        let trimmedPath = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let expandedPath = (trimmedPath as NSString).expandingTildeInPath
        guard !expandedPath.isEmpty, NSString(string: expandedPath).isAbsolutePath else {
            statusMessage = "輸出目錄必須是絕對路徑。"
            return false
        }

        let outputURL = URL(fileURLWithPath: expandedPath, isDirectory: true).standardizedFileURL
        do {
            try FileManager.default.createDirectory(
                at: outputURL,
                withIntermediateDirectories: true
            )
        } catch {
            statusMessage = "無法建立輸出目錄：\(error.localizedDescription)"
            return false
        }

        textToImageService.setOutputDirectory(outputURL)
        imageToImageService.setOutputDirectory(outputURL)
        upscaleService.setOutputDirectory(outputURL)
        subtitleGenerationService.setOutputDirectory(outputURL)
        videoGenerationService = VideoGenerationRouter(outputDirectory: outputURL)
        musicGenerationService = Self.makeMusicGenerationService(outputDirectory: outputURL)
        mediaCompositionService = MediaCompositionService(outputDirectory: outputURL)
        outputDirectoryPath = outputURL.path
        UserDefaults.standard.set(outputURL.path, forKey: Self.outputDirectoryKey)
        statusMessage = "輸出目錄已更新：\(outputURL.path)"
        return true
    }

    func revealOutputDirectory() {
        let outputURL = URL(fileURLWithPath: outputDirectoryPath, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
            guard NSWorkspace.shared.open(outputURL) else {
                throw CocoaError(.fileNoSuchFile)
            }
        } catch {
            statusMessage = "無法開啟輸出目錄：\(error.localizedDescription)"
        }
    }

    @discardableResult
    func setModelRoot(_ rawPath: String) -> Bool {
        let trimmedPath = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let expandedPath = (trimmedPath as NSString).expandingTildeInPath
        guard !expandedPath.isEmpty, NSString(string: expandedPath).isAbsolutePath else {
            statusMessage = "模型路徑必須是絕對路徑。"
            return false
        }

        let rootURL = URL(fileURLWithPath: expandedPath, isDirectory: true).standardizedFileURL
        loadModelCatalog(at: rootURL)
        return true
    }

    func loadModelCatalog(at rootURL: URL, initialLoad: Bool = false, allowMissingRoot: Bool = false) {
        let scanningMessage = "正在掃描模型目錄：\(rootURL.path)"
        if !initialLoad || statusMessage == nil { statusMessage = scanningMessage }
        // Removal may already be inside an uninterruptible filesystem call.
        // Let it publish its final state before scanning, even if the new root fails.
        modelOperationQueue.cancelAll(except: Set(modelRemovalTokens.keys))
        modelDiscovery.load(at: rootURL, allowMissingRoot: allowMissingRoot,
            beforeDiscovery: { [modelOperationQueue] in await modelOperationQueue.waitForAll() }
        ) { [weak self] result in
            guard let self else { return }
            switch result {
            case let .success(discovered):
                self.applyModelCatalog(discovered, rootURL: rootURL)
                if self.statusMessage == scanningMessage || self.statusMessage == "請等待模型目錄掃描完成。" {
                    self.statusMessage = "模型掃描完成，偵測到 \(discovered.models.count) 個本機模型與 \(discovered.loras.count) 個 LoRA。"
                }
            case let .failure(error):
                let message = "無法讀取模型目錄：\(rootURL.path)；\(error.localizedDescription)"
                if !initialLoad || self.statusMessage == scanningMessage || self.statusMessage == nil {
                    self.statusMessage = message
                } else if initialLoad {
                    self.statusMessage = (self.statusMessage ?? "") + "\n" + message
                }
            }
        }
    }

    private func applyModelCatalog(_ discovered: DiscoveredModelCatalog, rootURL: URL) {
        let refreshedModels = Self.mergedModels(discovered: discovered)
        let customProfiles = profiles.filter { !$0.isBuiltIn }
        let refreshedProfiles = Self.mergedProfiles(discovered: discovered) + customProfiles
        var previousInstallations = modelRootPath == rootURL.path ? installations : [:]
        for modelID in modelTasks.keys {
            if var installation = previousInstallations[modelID] {
                installation.phase = .paused
                previousInstallations[modelID] = installation
            }
        }
        loraDiscovery.cancel()
        modelOperationQueue.cancelAll()
        modelTasks.values.forEach { $0.cancel() }
        modelTasks.removeAll()
        modelTaskTokens.removeAll()
        modelRootPath = rootURL.path
        UserDefaults.standard.set(rootURL.path, forKey: Self.modelRootKey)
        models = refreshedModels
        loras = discovered.loras
        profiles = refreshedProfiles
        disabledProfileIDs = Self.disabledProfileIDs(in: refreshedProfiles)
        installations = Self.installations(for: refreshedModels, preserving: previousInstallations)
        activeProfileIDs = Self.persistedActiveProfileIDs(in: refreshedProfiles, models: refreshedModels)
        recipe.lora = Self.validatedPersistedLoRA(recipe.lora, available: discovered.loras)
        if let generationProfile = activeProfile(for: .textToImage) {
            recipe.profileID = generationProfile.id
            recipe.modelID = generationProfile.modelID
        } else {
            recipe.profileID = nil
        }
    }

    func refreshLoRAs(at rootURL: URL) {
        loraDiscovery.load(at: rootURL) { [weak self] result in
            guard let self, self.modelRootPath == rootURL.path else { return }
            if case let .success(catalog) = result {
                self.loras = catalog.loras
                self.recipe.lora = Self.validatedPersistedLoRA(self.recipe.lora, available: catalog.loras)
            }
        }
    }

    func startSystemMetricsUpdates() {
        systemMetricsTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                let snapshot = await Task.detached(priority: .utility) {
                    SystemMetricsReader.read()
                }.value
                guard !Task.isCancelled, let store = self else { return }
                store.systemMetrics = snapshot
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }
}
