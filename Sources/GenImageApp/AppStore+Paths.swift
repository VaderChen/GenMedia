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

        outputDirectoryPath = outputURL.path
        UserDefaults.standard.set(outputURL.path, forKey: Self.outputDirectoryKey)
        let textToImageService = textToImageService
        let imageToImageService = imageToImageService
        let upscaleService = upscaleService
        Task {
            await textToImageService.setOutputDirectory(outputURL)
            await imageToImageService.setOutputDirectory(outputURL)
            await upscaleService.setOutputDirectory(outputURL)
        }
        videoGenerationService = LTXVideoGenerationService(outputDirectory: outputURL)
        musicGenerationService = Self.makeMusicGenerationService(outputDirectory: outputURL)
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
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: rootURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            statusMessage = "找不到模型目錄：\(rootURL.path)"
            return false
        }

        let discovered = LocalModelDiscovery.discover(at: rootURL)
        let refreshedModels = Self.mergedModels(discovered: discovered)
        let customProfiles = profiles.filter { !$0.isBuiltIn }
        let refreshedProfiles = Self.mergedProfiles(discovered: discovered) + customProfiles
        let previousInstallations = installations
        let refreshedDisabledProfileIDs = Self.disabledProfileIDs(in: refreshedProfiles)

        modelTasks.values.forEach { $0.cancel() }
        modelTasks.removeAll()
        modelTaskTokens.removeAll()
        modelRootPath = rootURL.path
        UserDefaults.standard.set(rootURL.path, forKey: Self.modelRootKey)
        models = refreshedModels
        loras = discovered.loras
        profiles = refreshedProfiles
        disabledProfileIDs = refreshedDisabledProfileIDs
        installations = Self.installations(for: refreshedModels, preserving: previousInstallations)
        activeProfileIDs = Self.persistedActiveProfileIDs(
            in: refreshedProfiles,
            models: refreshedModels
        )

        recipe.lora = Self.validatedPersistedLoRA(recipe.lora, available: discovered.loras)
        if let generationProfile = activeProfile(for: .textToImage) {
            recipe.profileID = generationProfile.id
            recipe.modelID = generationProfile.modelID
        } else {
            recipe.profileID = nil
        }
        statusMessage = "已切換模型路徑，偵測到 \(discovered.models.count) 個本機模型與 \(discovered.loras.count) 個 LoRA。"
        return true
    }

    func startSystemMetricsUpdates() {
        systemMetrics = SystemMetricsReader.read()
        systemMetricsTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                if Task.isCancelled { return }
                self?.systemMetrics = SystemMetricsReader.read()
            }
        }
    }
}
