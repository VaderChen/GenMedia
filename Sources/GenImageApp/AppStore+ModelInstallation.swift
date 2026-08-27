import AppKit
import Foundation
import GenImageCore
import GenImageRuntime

// 模型的安裝、暫停、移除與修復。
extension AppStore {
    func installation(for modelID: String) -> ModelInstallation {
        installations[modelID] ?? ModelInstallation()
    }

    func installProfileModels(_ profileID: UUID) {
        guard let profile = profiles.first(where: { $0.id == profileID }) else { return }
        let requiredModels = profile.requiredModelIDs.compactMap { requiredModelID in
            models.first { $0.id == requiredModelID }
        }
        guard requiredModels.count == profile.requiredModelIDs.count else {
            let knownIDs = Set(requiredModels.map(\.id))
            let missingIDs = profile.requiredModelIDs.filter { !knownIDs.contains($0) }
            statusMessage = "找不到 Profile 的相關模型：\(missingIDs.joined(separator: "、"))"
            return
        }

        var startedCount = 0
        for model in requiredModels where installation(for: model.id).phase != .installed {
            installModel(model)
            startedCount += 1
        }
        statusMessage = startedCount > 0
            ? "已開始下載「\(profile.name)」所需的 \(startedCount) 個相關模型；已安裝項目會自動去重。"
            : "「\(profile.name)」的相關模型皆已安裝。"
    }

    func installModel(_ model: ModelDescriptor) {
        if model.localURL != nil {
            installations[model.id] = ModelInstallation(
                phase: .installed,
                progress: 1,
                downloadedGB: model.approximateDownloadGB
            )
            return
        }
        guard HuggingFaceModelInstaller.supports(modelID: model.id) else {
            let message = "此模型尚未提供可執行的自動下載方案。"
            installations[model.id] = ModelInstallation(phase: .failed, errorMessage: message)
            statusMessage = message
            return
        }
        guard modelTasks[model.id] == nil else {
            statusMessage = "「\(model.displayName)」已有下載任務正在執行。"
            return
        }
        let taskToken = UUID()
        modelTaskTokens[model.id] = taskToken
        let startProgress = min(1, max(0, installations[model.id]?.progress ?? 0))
        installations[model.id] = ModelInstallation(
            phase: .downloading,
            progress: startProgress,
            downloadedGB: model.approximateDownloadGB * startProgress
        )
        let progressGate = ModelProgressGate(interval: Self.modelProgressUpdateInterval)

        modelTasks[model.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            defer {
                if modelTaskTokens[model.id] == taskToken {
                    modelTasks[model.id] = nil
                    modelTaskTokens[model.id] = nil
                }
            }
            do {
                let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
                let localURL = try await modelInstaller.install(
                    modelID: model.id,
                    rootURL: rootURL,
                    progress: { [weak self] update in
                        guard progressGate.shouldEmit(update) else { return }
                        Task { @MainActor [weak self] in
                            guard let self, modelTaskTokens[model.id] == taskToken else { return }
                            // The resolved Hugging Face file list is authoritative;
                            // keep the catalog estimate in sync so repositories that
                            // add or remove shards do not show e.g. 9.6 GB / 7.6 GB.
                            let resolvedTotalGB = Double(update.totalBytes) / 1_073_741_824
                            if resolvedTotalGB > 0,
                               abs(models.first(where: { $0.id == model.id })?.approximateDownloadGB ?? 0
                                   - resolvedTotalGB) > 0.01 {
                                if let index = models.firstIndex(where: { $0.id == model.id }) {
                                    models[index].approximateDownloadGB = resolvedTotalGB
                                }
                            }
                            let downloadedGB = Double(update.downloadedBytes) / 1_073_741_824
                            installations[model.id] = ModelInstallation(
                                phase: .downloading,
                                progress: update.fractionCompleted,
                                downloadedGB: downloadedGB
                            )
                        }
                    }
                )
                try Task.checkCancellation()
                guard modelTaskTokens[model.id] == taskToken else { return }
                let resolvedDownloadGB = models.first(where: { $0.id == model.id })?.approximateDownloadGB
                    ?? model.approximateDownloadGB
                installations[model.id] = ModelInstallation(
                    phase: .verifying,
                    progress: 1,
                    downloadedGB: resolvedDownloadGB
                )
                _ = try HuggingFaceModelInstaller.verify(modelID: model.id, rootURL: rootURL)
                if let index = models.firstIndex(where: { $0.id == model.id }) {
                    models[index].localURL = localURL
                }
                loras = LocalModelDiscovery.discover(at: rootURL).loras
                installations[model.id] = ModelInstallation(
                    phase: .installed,
                    progress: 1,
                    downloadedGB: resolvedDownloadGB
                )
                statusMessage = "「\(model.displayName)」已下載並驗證完成。"
            } catch is CancellationError {
                guard modelTaskTokens[model.id] == taskToken else { return }
                var installation = installation(for: model.id)
                installation.phase = .paused
                installations[model.id] = installation
            } catch {
                guard modelTaskTokens[model.id] == taskToken else { return }
                var installation = installation(for: model.id)
                installation.phase = .failed
                installation.errorMessage = error.localizedDescription
                installations[model.id] = installation
                if let installerError = error as? ModelInstallerError,
                   case .authenticationRequired = installerError {
                    statusMessage = "Civitai LoRA 下載需要有效的 API Token，請至設定輸入後重試。"
                } else {
                    statusMessage = "模型下載失敗：\(error.localizedDescription)"
                }
            }
        }
    }

    func pauseModel(_ model: ModelDescriptor) {
        modelTaskTokens[model.id] = nil
        modelTasks[model.id]?.cancel()
        modelTasks[model.id] = nil
        var installation = installation(for: model.id)
        installation.phase = .paused
        installations[model.id] = installation
    }

    func removeModel(_ model: ModelDescriptor) {
        guard confirmModelRemoval(model) else { return }
        modelTaskTokens[model.id] = nil
        modelTasks[model.id]?.cancel()
        modelTasks[model.id] = nil
        do {
            if HuggingFaceModelInstaller.supports(modelID: model.id) {
                let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
                try HuggingFaceModelInstaller.remove(modelID: model.id, rootURL: rootURL)
            } else if let localURL = model.localURL {
                let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
                    .resolvingSymlinksInPath()
                    .standardizedFileURL
                let targetURL = localURL
                    .resolvingSymlinksInPath()
                    .standardizedFileURL
                let rootPath = rootURL.path.hasSuffix("/") ? rootURL.path : "\(rootURL.path)/"
                guard targetURL.path != rootURL.path,
                      targetURL.path.hasPrefix(rootPath) else {
                    throw NSError(
                        domain: "GenImage.ModelRemoval",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "模型檔案不在目前的模型目錄內，為避免誤刪除已取消操作。"]
                    )
                }
                try FileManager.default.removeItem(at: targetURL)
            }
            if let index = models.firstIndex(where: { $0.id == model.id }) {
                models[index].localURL = nil
            }
            deactivateProfiles(usingModelID: model.id)
            let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
            loras = LocalModelDiscovery.discover(at: rootURL).loras
            installations[model.id] = ModelInstallation()
            statusMessage = "已移除「\(model.displayName)」。"
        } catch {
            installations[model.id] = ModelInstallation(
                phase: .failed,
                errorMessage: error.localizedDescription
            )
            statusMessage = "移除模型失敗：\(error.localizedDescription)"
        }
    }

    private func confirmModelRemoval(_ model: ModelDescriptor) -> Bool {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "確定要移除模型嗎？"
        let location = model.localURL?.path ?? model.id
        alert.informativeText = "將移除「\(model.displayName)」及其本機檔案。\n\n路徑：\(location)\n\n此操作無法復原。"
        alert.addButton(withTitle: "移除")
        alert.addButton(withTitle: "取消")
        return alert.runModal() == .alertFirstButtonReturn
    }

    func repairModel(_ model: ModelDescriptor) {
        if HuggingFaceModelInstaller.supports(modelID: model.id) {
            let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
            do {
                let localURL = try HuggingFaceModelInstaller.verify(
                    modelID: model.id,
                    rootURL: rootURL
                )
                if let index = models.firstIndex(where: { $0.id == model.id }) {
                    models[index].localURL = localURL
                }
                installations[model.id] = ModelInstallation(
                    phase: .installed,
                    progress: 1,
                    downloadedGB: model.approximateDownloadGB
                )
                statusMessage = "「\(model.displayName)」驗證完成。"
            } catch {
                installations[model.id] = ModelInstallation(phase: .queued)
                deactivateProfiles(usingModelID: model.id)
                statusMessage = "偵測到缺少檔案，開始續傳修復。"
                if let index = models.firstIndex(where: { $0.id == model.id }) {
                    models[index].localURL = nil
                }
                var downloadableModel = model
                downloadableModel.localURL = nil
                installModel(downloadableModel)
            }
            return
        }
        if let localURL = model.localURL {
            if FileManager.default.fileExists(atPath: localURL.path) {
                installations[model.id] = ModelInstallation(
                    phase: .installed,
                    progress: 1,
                    downloadedGB: model.approximateDownloadGB
                )
                statusMessage = "本機模型檔案仍然存在。"
            } else {
                installations[model.id] = ModelInstallation(
                    phase: .failed,
                    errorMessage: "找不到本機模型路徑。"
                )
                statusMessage = "找不到本機模型路徑：\(localURL.path)"
            }
            return
        }
        installations[model.id] = ModelInstallation(
            phase: .failed,
            errorMessage: "此模型沒有可驗證的下載方案。"
        )
    }
}

/// 節流下載回呼，避免大檔案下載時每個網路區塊都排入主執行緒，
/// 造成整個 WebUI 頻繁重繪而無法操作。
private final class ModelProgressGate: @unchecked Sendable {
    private let interval: TimeInterval
    private let lock = NSLock()
    private var lastReportedAt = Date.distantPast
    private var lastFraction = -1.0

    init(interval: TimeInterval) {
        self.interval = interval
    }

    func shouldEmit(_ update: ModelInstallProgress) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        let now = Date()
        let shouldEmit = update.fractionCompleted >= 1
            || update.fractionCompleted - lastFraction >= 0.01
            || now.timeIntervalSince(lastReportedAt) >= interval
        guard shouldEmit else { return false }
        lastReportedAt = now
        lastFraction = update.fractionCompleted
        return true
    }
}
