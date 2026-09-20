import Foundation
import GenImageCore
import GenImageRuntime

// 模型的安裝、暫停、移除與修復。
extension AppStore {
    private func enqueueModelTask(
        _ modelID: String, token: UUID,
        operation: @escaping @MainActor () async -> Void
    ) {
        modelTaskTokens[modelID] = token
        modelTasks[modelID] = modelOperationQueue.replace(for: modelID, operation: operation) { [weak self] in
            guard let self else { return }
            if self.modelRemovalTokens[modelID] == token { self.modelRemovalTokens[modelID] = nil }
            guard self.modelTaskTokens[modelID] == token else { return }
            if Task.isCancelled, var installation = self.installations[modelID],
               [.queued, .downloading, .verifying].contains(installation.phase) {
                installation.phase = .paused
                self.installations[modelID] = installation
            }
            self.modelTasks[modelID] = nil
            self.modelTaskTokens[modelID] = nil
        }
    }

    func installation(for modelID: String) -> ModelInstallation {
        installations[modelID] ?? ModelInstallation()
    }

    func installProfileModels(
        _ profileID: UUID,
        civitaiToken: String? = nil,
        huggingFaceToken: String? = nil
    ) {
        guard !modelDiscovery.isLoading else {
            statusMessage = "請等待模型目錄掃描完成。"
            return
        }
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
            installModel(
                model,
                civitaiToken: civitaiToken,
                huggingFaceToken: huggingFaceToken
            )
            startedCount += 1
        }
        statusMessage = startedCount > 0
            ? "已開始下載「\(profile.name)」所需的 \(startedCount) 個相關模型；已安裝項目會自動去重。"
            : "「\(profile.name)」的相關模型皆已安裝。"
    }

    func installModel(
        _ model: ModelDescriptor,
        civitaiToken: String? = nil,
        huggingFaceToken: String? = nil
    ) {
        guard !modelDiscovery.isLoading else {
            statusMessage = "請等待模型目錄掃描完成。"
            return
        }
        guard modelTasks[model.id] == nil, modelRemovalTokens[model.id] == nil else {
            statusMessage = "「\(model.displayName)」已有模型操作正在執行。"
            return
        }
        let resolvedHuggingFaceToken = huggingFaceToken ?? HuggingFaceTokenStore.token()
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
        let taskToken = UUID()
        modelTaskTokens[model.id] = taskToken
        let startProgress = min(1, max(0, installations[model.id]?.progress ?? 0))
        installations[model.id] = ModelInstallation(
            phase: .downloading,
            progress: startProgress,
            downloadedGB: model.approximateDownloadGB * startProgress
        )
        let progressGate = ModelProgressGate(interval: Self.modelProgressUpdateInterval)

        let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
        enqueueModelTask(model.id, token: taskToken) { [weak self] in
            guard let self else { return }
            do {
                try Task.checkCancellation()
                let localURL = try await modelInstaller.install(
                    modelID: model.id,
                    rootURL: rootURL,
                    civitaiToken: civitaiToken,
                    huggingFaceToken: resolvedHuggingFaceToken,
                    progress: { [weak self] update in
                        guard progressGate.shouldEmit(update) else { return }
                        Task { @MainActor [weak self] in
                            guard let self, modelTaskTokens[model.id] == taskToken,
                                  installations[model.id]?.phase == .downloading else { return }
                            // The resolved Hugging Face file list is authoritative;
                            // keep the catalog estimate in sync so repositories that
                            // add or remove shards do not show e.g. 9.6 GB / 7.6 GB.
                            let resolvedTotalGB = Double(update.totalBytes) / 1_073_741_824
                            if resolvedTotalGB > 0,
                               abs((models.first(where: { $0.id == model.id })?.approximateDownloadGB ?? 0)
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
                _ = try await BackgroundTask.run {
                    try HuggingFaceModelInstaller.verify(modelID: model.id, rootURL: rootURL)
                }
                guard modelTaskTokens[model.id] == taskToken, modelRootPath == rootURL.path else { return }
                if let index = models.firstIndex(where: { $0.id == model.id }) {
                    models[index].localURL = localURL
                }
                refreshLoRAs(at: rootURL)
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
                } else if let installerError = error as? ModelInstallerError,
                          case .httpStatus(401, _) = installerError {
                    statusMessage = HuggingFaceTokenStore.isConfigured()
                        ? "Hugging Face Token 已設定，但模型伺服器仍回傳 HTTP 401；請重新儲存 Token 後重試。"
                        : "模型需要 Hugging Face API Token，請點選下載並輸入金鑰後重試。"
                } else {
                    statusMessage = "模型下載失敗：\(error.localizedDescription)"
                }
            }
        }
    }

    func pauseModel(_ model: ModelDescriptor) {
        guard modelRemovalTokens[model.id] == nil else {
            statusMessage = "模型正在移除，請等待檔案清理完成。"
            return
        }
        modelTaskTokens[model.id] = nil
        modelTasks[model.id]?.cancel()
        modelTasks[model.id] = nil
        var installation = installation(for: model.id)
        installation.phase = .paused
        installations[model.id] = installation
    }

    func removeModel(_ model: ModelDescriptor) {
        guard !modelDiscovery.isLoading else {
            statusMessage = "請等待模型目錄掃描完成。"
            return
        }
        guard ensureInferenceIdle() else { return }
        let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
        let token = UUID()
        modelRemovalTokens[model.id] = token
        installations[model.id] = ModelInstallation(phase: .verifying)
        statusMessage = "正在停止相關任務並移除「\(model.displayName)」…"
        enqueueModelTask(model.id, token: token) { [weak self] in
            guard let self else { return }
            do {
                try await BackgroundTask.run {
                    if HuggingFaceModelInstaller.supports(modelID: model.id) {
                        try HuggingFaceModelInstaller.remove(modelID: model.id, rootURL: rootURL)
                    } else if let localURL = model.localURL {
                        let root = rootURL.resolvingSymlinksInPath().standardizedFileURL
                        let target = localURL.resolvingSymlinksInPath().standardizedFileURL
                        let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
                        guard target.path != root.path, target.path.hasPrefix(prefix) else {
                            throw NSError(domain: "GenImage.ModelRemoval", code: 1,
                                userInfo: [NSLocalizedDescriptionKey: "模型檔案不在目前的模型目錄內，為避免誤刪除已取消操作。"])
                        }
                        try Task.checkCancellation()
                        try FileManager.default.removeItem(at: target)
                    }
                }
                guard modelTaskTokens[model.id] == token, modelRootPath == rootURL.path else { return }
                if let index = models.firstIndex(where: { $0.id == model.id }) {
                    models[index].localURL = nil
                }
                deactivateProfiles(usingModelID: model.id)
                refreshLoRAs(at: rootURL)
                installations[model.id] = ModelInstallation()
                statusMessage = "已移除「\(model.displayName)」。"
            } catch is CancellationError {
                guard modelTaskTokens[model.id] == token else { return }
                installations[model.id]?.phase = .paused
            } catch {
                guard modelTaskTokens[model.id] == token, modelRootPath == rootURL.path else { return }
                installations[model.id] = ModelInstallation(phase: .failed, errorMessage: error.localizedDescription)
                statusMessage = "移除模型失敗：\(error.localizedDescription)"
            }
        }
    }

    func repairModel(
        _ model: ModelDescriptor,
        civitaiToken: String? = nil,
        huggingFaceToken: String? = nil
    ) {
        guard !modelDiscovery.isLoading else {
            statusMessage = "請等待模型目錄掃描完成。"
            return
        }
        guard modelTasks[model.id] == nil else {
            statusMessage = "請等待目前的模型安裝任務完成。"
            return
        }
        if HuggingFaceModelInstaller.supports(modelID: model.id) {
            let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
            let token = UUID()
            modelTaskTokens[model.id] = token
            installations[model.id] = ModelInstallation(phase: .verifying, progress: 1,
                downloadedGB: model.approximateDownloadGB)
            enqueueModelTask(model.id, token: token) { [weak self] in
                guard let self else { return }
                do {
                    let localURL = try await BackgroundTask.run {
                        try HuggingFaceModelInstaller.verify(modelID: model.id, rootURL: rootURL)
                    }
                    guard modelTaskTokens[model.id] == token, modelRootPath == rootURL.path else { return }
                    if let index = models.firstIndex(where: { $0.id == model.id }) {
                        models[index].localURL = localURL
                    }
                    installations[model.id] = ModelInstallation(phase: .installed, progress: 1,
                        downloadedGB: model.approximateDownloadGB)
                    refreshLoRAs(at: rootURL)
                    statusMessage = "「\(model.displayName)」驗證完成。"
                } catch is CancellationError {
                    guard modelTaskTokens[model.id] == token else { return }
                    installations[model.id]?.phase = .paused
                } catch {
                    guard modelTaskTokens[model.id] == token, modelRootPath == rootURL.path else { return }
                    guard !modelDiscovery.isLoading else {
                        installations[model.id]?.phase = .paused
                        return
                    }
                    // Release this verification slot before creating its replacement download.
                    modelTasks[model.id] = nil
                    modelTaskTokens[model.id] = nil
                    installations[model.id] = ModelInstallation(phase: .queued)
                    deactivateProfiles(usingModelID: model.id)
                    statusMessage = "偵測到缺少檔案，開始續傳修復。"
                    if let index = models.firstIndex(where: { $0.id == model.id }) {
                        models[index].localURL = nil
                    }
                    var downloadableModel = model
                    downloadableModel.localURL = nil
                    installModel(downloadableModel, civitaiToken: civitaiToken,
                        huggingFaceToken: huggingFaceToken)
                }
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
