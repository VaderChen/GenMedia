import Foundation
import GenImageCore

// 多媒體語音辨識與字幕輸出流程。
extension AppStore {
    func requestSubtitleGeneration(
        sourceAssetID: UUID? = nil,
        format: SubtitleFormat = .srt,
        targetLanguage: SubtitleTranslationLanguage? = nil
    ) {
        guard ensureInferenceIdle() else { return }
        guard let profile = activeProfile(for: .videoToText) else {
            statusMessage = "請先啟用影生文 Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        let sourceAsset: MediaAsset?
        if let sourceAssetID {
            sourceAsset = subtitleSourceAsset(id: sourceAssetID)
        } else {
            sourceAsset = selectedSubtitleSource
        }
        guard let sourceAsset else {
            statusMessage = "字幕生成需要先匯入或選取包含聲音的影片或音訊。"
            return
        }
        guard let model = models.first(where: { $0.id == profile.modelID }),
              let modelURL = model.localURL else {
            statusMessage = "找不到字幕 Profile 指定模型的本機安裝路徑。"
            return
        }

        let translation: SubtitleTranslationConfiguration?
        if let targetLanguage {
            guard let translationProfile = activeProfile(for: .textToText) else {
                statusMessage = "字幕翻譯需要先啟用文生文 Profile。"
                return
            }
            guard isProfileReady(translationProfile) else { return }
            guard let translationModel = models.first(where: {
                $0.id == translationProfile.modelID
            }), let translationModelURL = translationModel.localURL else {
                statusMessage = "找不到文生文 Profile 指定模型的本機安裝路徑。"
                return
            }
            translation = SubtitleTranslationConfiguration(
                targetLanguage: targetLanguage,
                profile: translationProfile,
                modelURL: translationModelURL
            )
        } else {
            translation = nil
        }

        let projectID = selectedProjectID
        let translationTitle = targetLanguage.map { "並翻譯為 \($0.rawValue)" } ?? ""
        let job = GenerationJob(
            action: .generateSubtitles,
            title: "為「\(sourceAsset.title)」生成 \(format.displayName) 字幕\(translationTitle)"
        )
        jobs.append(job)
        updateJob(job.id) {
            $0.state = .running
            $0.progress = 0.001
        }

        let service = subtitleGenerationService
        let request = SubtitleGenerationRequest(
            projectID: projectID,
            sourceAsset: sourceAsset,
            profile: profile,
            modelURL: modelURL,
            format: format,
            translation: translation
        )
        let jobID = job.id
        let subtitleTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                var result = try await service.generate(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(jobID, value: value)
                        }
                    }
                )
                try Task.checkCancellation()

                let operationID = UUID()
                result.asset.operationID = operationID
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    assets.append(result.asset)
                    operations.append(
                        WorkflowOperation(
                            id: operationID,
                            projectID: projectID,
                            action: .generateSubtitles,
                            inputAssetID: sourceAsset.id,
                            outputAssetIDs: [result.asset.id],
                            profileSnapshot: profile
                        )
                    )
                    comparisonAssetID = nil
                    selectedAssetID = result.asset.id
                    previewMode = .single
                    updateJob(jobID) {
                        $0.progress = 1
                        $0.state = .completed
                    }
                    if let targetLanguage {
                        statusMessage = "字幕生成完成，已翻譯為 \(targetLanguage.rawValue) 並輸出 \(format.displayName)。"
                    } else {
                        statusMessage = "字幕生成完成，已輸出 \(format.displayName)。"
                    }
                }
            } catch is CancellationError {
                await MainActor.run { [weak self] in
                    guard let self,
                          jobs.first(where: { $0.id == jobID })?.state == .running else {
                        return
                    }
                    updateJob(jobID) { $0.state = .cancelled }
                }
            } catch {
                let message = error.localizedDescription
                await MainActor.run { [weak self] in
                    guard let self,
                          jobs.first(where: { $0.id == jobID }).map({
                              [.running, .cancelling].contains($0.state)
                          }) == true else {
                        return
                    }
                    updateJob(jobID) {
                        $0.state = .failed
                        $0.errorMessage = message
                    }
                    statusMessage = "字幕生成失敗：\(message)"
                }
            }
            await MainActor.run { [weak self] in
                self?.finishJobTask(jobID)
            }
        }
        jobTasks[jobID] = subtitleTask
    }
}
