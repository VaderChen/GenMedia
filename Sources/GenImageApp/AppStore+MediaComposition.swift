import Foundation
import GenImageCore
import GenImageRuntime

// 不需要模型的媒體工作：圖片循環影片與影音合併。
extension AppStore {
    func requestImageLoop(
        sourceAssetIDs: [UUID],
        options: ImageLoopOptions
    ) {
        guard ensureInferenceIdle() else { return }
        let resolvedSources = sourceImages(for: sourceAssetIDs)
        guard !resolvedSources.isEmpty,
              resolvedSources.count == Set(sourceAssetIDs).count else {
            statusMessage = "圖片循環需要至少一張仍存在的來源圖片。"
            return
        }
        do {
            try options.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }

        let projectID = selectedProjectID
        let job = GenerationJob(
            action: .createImageLoop,
            title: "使用 \(resolvedSources.count) 張圖片建立循環影片"
        )
        jobs.append(job)
        updateJob(job.id) {
            $0.state = .running
            $0.progress = 0.001
        }

        let service = mediaCompositionService
        let jobID = job.id
        let task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                var output = try await service.createImageLoop(
                    projectID: projectID,
                    sourceAssets: resolvedSources,
                    options: options,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(jobID, value: value)
                        }
                    }
                )
                try Task.checkCancellation()
                let operationID = UUID()
                output.operationID = operationID
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    assets.append(output)
                    operations.append(
                        WorkflowOperation(
                            id: operationID,
                            projectID: projectID,
                            action: .createImageLoop,
                            inputAssetID: resolvedSources.first?.id,
                            inputAssetIDs: resolvedSources.map(\.id),
                            outputAssetIDs: [output.id]
                        )
                    )
                    comparisonAssetID = nil
                    selectedAssetID = output.id
                    previewMode = .single
                    updateJob(jobID) {
                        $0.progress = 1
                        $0.state = .completed
                    }
                    statusMessage = "圖片循環影片已生成。"
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
                    statusMessage = "圖片循環失敗：\(message)"
                }
            }
            await MainActor.run { [weak self] in
                self?.finishJobTask(jobID)
            }
        }
        jobTasks[jobID] = task
    }

    func requestMediaMerge(
        videoAssetID: UUID,
        audioAssetID: UUID,
        options: MediaMergeOptions
    ) {
        guard ensureInferenceIdle() else { return }
        guard let videoAsset = assets.first(where: {
            $0.id == videoAssetID && [.importedVideo, .generatedVideo].contains($0.kind)
        }) else {
            statusMessage = "影音合併需要先選擇影片來源。"
            return
        }
        guard let audioAsset = assets.first(where: {
            $0.id == audioAssetID && [.importedAudio, .generatedAudio].contains($0.kind)
        }) else {
            statusMessage = "影音合併需要先選擇音訊來源。"
            return
        }
        do {
            try options.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }

        let projectID = selectedProjectID
        let job = GenerationJob(
            action: .mergeMedia,
            title: "合併「\(videoAsset.title)」與「\(audioAsset.title)」"
        )
        jobs.append(job)
        updateJob(job.id) {
            $0.state = .running
            $0.progress = 0.001
        }

        let service = mediaCompositionService
        let jobID = job.id
        let task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                var output = try await service.mergeMedia(
                    projectID: projectID,
                    videoAsset: videoAsset,
                    audioAsset: audioAsset,
                    options: options,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(jobID, value: value)
                        }
                    }
                )
                try Task.checkCancellation()
                let operationID = UUID()
                output.operationID = operationID
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    assets.append(output)
                    operations.append(
                        WorkflowOperation(
                            id: operationID,
                            projectID: projectID,
                            action: .mergeMedia,
                            inputAssetID: videoAsset.id,
                            inputAssetIDs: [videoAsset.id, audioAsset.id],
                            outputAssetIDs: [output.id]
                        )
                    )
                    comparisonAssetID = nil
                    selectedAssetID = output.id
                    previewMode = .single
                    updateJob(jobID) {
                        $0.progress = 1
                        $0.state = .completed
                    }
                    statusMessage = "影音合併完成。"
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
                    statusMessage = "影音合併失敗：\(message)"
                }
            }
            await MainActor.run { [weak self] in
                self?.finishJobTask(jobID)
            }
        }
        jobTasks[jobID] = task
    }
}
