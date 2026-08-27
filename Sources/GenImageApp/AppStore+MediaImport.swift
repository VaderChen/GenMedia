import Foundation
import GenImageCore
import GenImageRuntime

// 媒體匯入與相容性轉檔共用生成工作佇列，讓長時間匯入可顯示進度並支援取消。
extension AppStore {
    func requestMediaImport(
        sourceURL: URL,
        requiresAudio: Bool,
        completion: (@MainActor @Sendable (UUID) -> Void)? = nil
    ) {
        guard ensureInferenceIdle() else { return }

        let assetID = UUID()
        let job = GenerationJob(
            action: .importMedia,
            title: "匯入「\(sourceURL.lastPathComponent)」"
        )
        jobs.append(job)
        updateJob(job.id) {
            $0.state = .running
            $0.progress = 0.001
        }

        let jobID = job.id
        let task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let prepared = try await MediaSourceCompatibilityService.prepare(
                    sourceURL: sourceURL,
                    assetID: assetID,
                    requiresAudio: requiresAudio,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(jobID, value: value)
                        }
                    }
                )
                try Task.checkCancellation()

                await MainActor.run { [weak self] in
                    guard let self else { return }
                    let importedAssetID = importMedia(
                        id: assetID,
                        url: sourceURL,
                        playbackURL: prepared.compatibilityURL,
                        kind: prepared.kind == .video ? .importedVideo : .importedAudio,
                        pixelWidth: prepared.pixelWidth,
                        pixelHeight: prepared.pixelHeight,
                        durationSeconds: prepared.durationSeconds,
                        compatibilityPrepared: prepared.preparation != .original
                    )
                    updateJob(jobID) {
                        $0.progress = 1
                        $0.state = .completed
                    }
                    statusMessage = "媒體匯入完成。"
                    completion?(importedAssetID)
                }
            } catch is CancellationError {
                await MainActor.run { [weak self] in
                    guard let self,
                          jobs.first(where: { $0.id == jobID }).map({
                              [.running, .cancelling].contains($0.state)
                          }) == true else { return }
                    updateJob(jobID) { $0.state = .cancelled }
                }
            } catch {
                let message = error.localizedDescription
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    updateJob(jobID) {
                        $0.state = .failed
                        $0.errorMessage = message
                    }
                    statusMessage = "無法匯入來源媒體：\(message)"
                }
            }
            await MainActor.run { [weak self] in
                self?.finishJobTask(jobID)
            }
        }
        jobTasks[jobID] = task
        statusMessage = "正在以 FFmpeg 分析並準備「\(sourceURL.lastPathComponent)」…"
    }
}
