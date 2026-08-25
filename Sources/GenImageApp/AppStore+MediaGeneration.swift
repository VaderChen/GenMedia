import Foundation
import GenImageCore

// 影片與音樂的生成流程。
extension AppStore {
    func requestVideoGeneration(sourceAssetIDs: [UUID] = []) {
        guard ensureInferenceIdle() else { return }
        let profile: InferenceProfile
        let sourceAssets: [MediaAsset]
        if sourceAssetIDs.isEmpty {
            guard let preferredVideoProfile else {
                statusMessage = "請先啟用文生影或圖生影 Profile。"
                return
            }
            profile = preferredVideoProfile
            if profile.capability == .imageToVideo {
                guard let selectedSourceImage else {
                    statusMessage = "圖生影需要先匯入或選取至少一張圖片。"
                    return
                }
                sourceAssets = [selectedSourceImage]
            } else {
                sourceAssets = []
            }
        } else {
            let resolvedSourceAssets = sourceImages(for: sourceAssetIDs)
            guard resolvedSourceAssets.count == Set(sourceAssetIDs).count else {
                statusMessage = "部分圖生影錨點已不存在或不是可用圖片，請重新選取。"
                return
            }
            guard let imageToVideoProfile = activeProfile(for: .imageToVideo) else {
                statusMessage = "已選取圖片錨點，請先啟用圖生影 Profile。"
                return
            }
            profile = imageToVideoProfile
            sourceAssets = resolvedSourceAssets
        }
        guard isProfileReady(profile) else { return }
        guard let model = models.first(where: { $0.id == profile.modelID }) else {
            statusMessage = "找不到影片 Profile 指定的模型。"
            return
        }
        guard let modelURL = model.localURL else {
            statusMessage = "找不到影片模型的本機安裝路徑。"
            return
        }
        guard let profileLoRAs = resolvedVideoLoRAs(for: profile) else { return }

        let normalizedFrameCount = normalizedVideoFrameCount(
            videoOutputSettings.frameCount,
            for: profile
        )
        if normalizedFrameCount != videoOutputSettings.frameCount {
            videoOutputSettings.frameCount = normalizedFrameCount
        }

        let options = VideoGenerationOptions(
            prompt: recipe.prompt,
            width: videoOutputSettings.width,
            height: videoOutputSettings.height,
            steps: videoOutputSettings.steps,
            outputCount: videoOutputSettings.outputCount,
            frameCount: normalizedFrameCount,
            frameRate: videoOutputSettings.frameRate,
            seed: videoOutputSettings.seed
        )
        do {
            try options.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }

        let projectID = selectedProjectID
        let recipeID = recipe.id
        let jobTitle: String
        if profile.capability == .imageToVideo {
            jobTitle = sourceAssets.count == 1
                ? "從「\(sourceAssets[0].title)」生成影片"
                : "使用 \(sourceAssets.count) 張圖片錨點生成影片"
        } else {
            jobTitle = "生成 \(options.outputCount) 部影片"
        }
        let job = GenerationJob(
            action: .generateVideo,
            title: jobTitle
        )
        jobs.append(job)
        updateJob(job.id) {
            $0.state = .running
            $0.progress = 0.001
        }
        let service = videoGenerationService
        let request = VideoGenerationRequest(
            projectID: projectID,
            recipeID: recipeID,
            sourceAsset: sourceAssets.first,
            sourceAssets: sourceAssets,
            options: options,
            profile: profile,
            modelURL: modelURL,
            profileLoRAs: profileLoRAs
        )

        let jobID = job.id
        let videoTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                var newAssets = try await service.generate(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(jobID, value: value)
                        }
                    }
                )
                try Task.checkCancellation()

                let operationID = UUID()
                for index in newAssets.indices {
                    newAssets[index].operationID = operationID
                }
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    assets.append(contentsOf: newAssets)
                    operations.append(
                        WorkflowOperation(
                            id: operationID,
                            projectID: projectID,
                            action: .generateVideo,
                            inputAssetID: sourceAssets.first?.id,
                            outputAssetIDs: newAssets.map(\.id),
                            recipeID: recipeID,
                            profileSnapshot: profile
                        )
                    )
                    comparisonAssetID = nil
                    selectedAssetID = newAssets.first?.id
                    previewMode = .single
                    updateJob(jobID) {
                        $0.progress = 1
                        $0.state = .completed
                    }
                    statusMessage = profile.capability == .imageToVideo
                        ? "圖生影完成。"
                        : "文生影完成。"
                }
            } catch is CancellationError {
                await MainActor.run { [weak self] in
                    guard let self,
                          jobs.first(where: { [.running, .cancelling].contains($0.state) }) != nil else {
                        return
                    }
                    updateJob(jobID) { $0.state = .cancelled }
                }
            } catch {
                let message = error.localizedDescription
                await MainActor.run { [weak self] in
                    guard let self,
                          jobs.first(where: { [.running, .cancelling].contains($0.state) }) != nil else {
                        return
                    }
                    updateJob(jobID) {
                        $0.state = .failed
                        $0.errorMessage = message
                    }
                    statusMessage = "影片生成失敗：\(message)"
                }
            }
            await MainActor.run { [weak self] in
                self?.finishJobTask(jobID)
            }
        }
        jobTasks[jobID] = videoTask

        Task.detached(priority: .utility) { [weak self] in
            try? await Task.sleep(for: .seconds(15))
            guard !Task.isCancelled else { return }
            await self?.failVideoJobIfStartupStalled(jobID)
        }
    }

    private func failVideoJobIfStartupStalled(_ jobID: UUID) {
        guard let job = jobs.first(where: { $0.id == jobID }),
              job.state == .running,
              job.progress < 0.01 else {
            return
        }
        jobTasks[jobID]?.cancel()
        jobTasks[jobID] = nil
        let message = "影片生成 Runtime 在 15 秒內未啟動，任務已自動停止；請重新執行。"
        updateJob(jobID) {
            $0.state = .failed
            $0.errorMessage = message
        }
        statusMessage = message
    }

    private static func formattedMusicDuration(_ durationSeconds: Double) -> String {
        let totalSeconds = max(0, Int(durationSeconds.rounded()))
        let minutes = totalSeconds / 60
        let seconds = totalSeconds % 60
        if minutes > 0, seconds > 0 {
            return "\(minutes) 分 \(seconds) 秒"
        }
        if minutes > 0 {
            return "\(minutes) 分"
        }
        return "\(seconds) 秒"
    }

    func requestMusicGeneration() {
        guard ensureInferenceIdle() else { return }
        guard let profile = activeProfile(for: .textToMusic) else {
            statusMessage = "請先啟用文生音樂 Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let model = models.first(where: { $0.id == profile.modelID }),
              let modelURL = model.localURL else {
            statusMessage = "找不到音樂模型的本機安裝路徑。"
            return
        }
        let musicPrompt = musicOutputSettings.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let options = MusicGenerationOptions(
            prompt: musicPrompt.isEmpty ? musicOutputSettings.style.prompt : musicPrompt,
            lyrics: musicOutputSettings.lyrics,
            durationSeconds: musicOutputSettings.durationSeconds,
            steps: musicOutputSettings.steps,
            seed: musicOutputSettings.seed,
            format: musicOutputSettings.format
        )
        do {
            try options.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }

        let projectID = selectedProjectID
        let recipeID = recipe.id
        let requestedDuration = "\(options.durationSeconds) 秒"
        let durationTitle = profile.music?.durationSemantics == .maximum
            ? "最長 \(requestedDuration)"
            : requestedDuration
        let job = GenerationJob(
            action: .generateMusic,
            title: "生成 \(durationTitle) \(options.format.displayName) 音樂"
        )
        jobs.append(job)
        updateJob(job.id) {
            $0.state = .running
            $0.progress = 0.001
        }
        let service = musicGenerationService
        let request = MusicGenerationRequest(
            projectID: projectID,
            recipeID: recipeID,
            options: options,
            profile: profile,
            modelURL: modelURL
        )
        let jobID = job.id
        let musicTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                var asset = try await service.generate(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJob(jobID) {
                                $0.progress = max($0.progress, min(1, max(0, value)))
                            }
                        }
                    }
                )
                try Task.checkCancellation()
                let operationID = UUID()
                asset.operationID = operationID
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    assets.append(asset)
                    operations.append(
                        WorkflowOperation(
                            id: operationID,
                            projectID: projectID,
                            action: .generateMusic,
                            outputAssetIDs: [asset.id],
                            recipeID: recipeID,
                            profileSnapshot: profile
                        )
                    )
                    comparisonAssetID = nil
                    selectedAssetID = asset.id
                    previewMode = .single
                    updateJob(jobID) {
                        $0.progress = 1
                        $0.state = .completed
                    }
                    let actualDuration = asset.mediaDurationSeconds.map(Self.formattedMusicDuration)
                        ?? "未知"
                    let requestedLabel = profile.music?.durationSemantics == .maximum
                        ? "設定最長 \(requestedDuration)"
                        : "設定 \(requestedDuration)"
                    statusMessage = "文生音樂完成，\(requestedLabel)，實際生成 \(actualDuration)，已輸出 \(options.format.displayName)。"
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
                          jobs.first(where: { $0.id == jobID })?.state == .running else {
                        return
                    }
                    updateJob(jobID) {
                        $0.state = .failed
                        $0.errorMessage = message
                    }
                    statusMessage = "音樂生成失敗：\(message)"
                }
            }
            await MainActor.run { [weak self] in
                self?.jobTasks[jobID] = nil
            }
        }
        jobTasks[jobID] = musicTask
    }
}
