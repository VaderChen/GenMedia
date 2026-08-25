import Foundation
import GenImageCore

// 圖生文、文生圖、圖生圖與 Upscale 的執行流程。
extension AppStore {
    func describeSelected() {
        guard ensureInferenceIdle() else { return }
        guard let input = selectedSourceImage else {
            statusMessage = "請先匯入或選取一張圖片。"
            return
        }
        guard let profile = activeProfile(for: .imageToText) else {
            statusMessage = "請先設定圖生文 Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let executionProfile = runtimeProfile(from: profile) else { return }

        let job = GenerationJob(action: .describe, title: "描述「\(input.title)」")
        jobs.append(job)
        updateJob(job.id) { $0.state = .running }
        let service = imageToTextService
        let request = ImageDescriptionRequest(
            asset: input,
            profile: executionProfile,
            languageCode: executionProfile.defaults.languageCode ?? "zh-Hant"
        )

        jobTasks[job.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let description = try await service.describe(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(job.id, value: value)
                        }
                    }
                )
                try Task.checkCancellation()
                recipe.prompt = description
                operations.append(
                    WorkflowOperation(
                        projectID: selectedProjectID,
                        action: .describe,
                        inputAssetID: input.id,
                        recipeID: recipe.id,
                        profileSnapshot: profile
                    )
                )
                updateJob(job.id) {
                    $0.progress = 1
                    $0.state = .completed
                }
                statusMessage = "圖生文完成，描述已放入 Prompt，可直接修改或生成。"
            } catch is CancellationError {
                updateJob(job.id) { $0.state = .cancelled }
            } catch {
                let message = error.localizedDescription
                updateJob(job.id) {
                    $0.state = .failed
                    $0.errorMessage = message
                }
                statusMessage = "圖生文失敗：\(message)"
            }
            finishJobTask(job.id)
        }
    }

    func generate(linkToSelectedAsset: Bool = false) {
        guard ensureInferenceIdle() else { return }
        do {
            try recipe.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }
        guard let profile = activeProfile(for: .textToImage) else {
            statusMessage = "請先設定文生圖 Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let executionProfile = runtimeProfile(from: profile) else { return }

        let input = linkToSelectedAsset ? selectedAsset : nil
        let recipeSnapshot = recipe
        let projectID = selectedProjectID
        let job = GenerationJob(action: .generate, title: "生成 \(recipeSnapshot.outputCount) 張圖片")
        jobs.append(job)
        updateJob(job.id) { $0.state = .running }
        let service = textToImageService
        let request = TextToImageRequest(
            projectID: projectID,
            recipe: recipeSnapshot,
            profile: executionProfile,
            sourceAsset: input
        )

        jobTasks[job.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                var newAssets = try await service.generate(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(job.id, value: value)
                        }
                    }
                )
                try Task.checkCancellation()

                let operationID = UUID()
                for index in newAssets.indices {
                    newAssets[index].operationID = operationID
                }
                assets.append(contentsOf: newAssets)
                operations.append(
                    WorkflowOperation(
                        id: operationID,
                        projectID: projectID,
                        action: .generate,
                        inputAssetID: input?.id,
                        outputAssetIDs: newAssets.map(\.id),
                        recipeID: recipeSnapshot.id,
                        profileSnapshot: profile
                    )
                )
                comparisonAssetID = newAssets.dropFirst().first?.id
                selectedAssetID = newAssets.first?.id
                previewMode = .grid
                updateJob(job.id) {
                    $0.progress = 1
                    $0.state = .completed
                }
                statusMessage = input == nil ? "獨立文生圖完成。" : "已從選取圖片建立新的生成分支。"
            } catch is CancellationError {
                updateJob(job.id) { $0.state = .cancelled }
            } catch {
                let message = error.localizedDescription
                updateJob(job.id) {
                    $0.state = .failed
                    $0.errorMessage = message
                }
                statusMessage = "文生圖失敗：\(message)"
            }
            finishJobTask(job.id)
        }
    }

    func imageToImageSelected() {
        guard ensureInferenceIdle() else { return }
        guard let input = selectedSourceImage else {
            statusMessage = "請先匯入或選取一張圖片。"
            return
        }
        do {
            try recipe.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }
        guard let profile = activeProfile(for: .imageToImage) else {
            statusMessage = "請先啟用圖生圖 Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let model = models.first(where: { $0.id == profile.modelID }) else {
            statusMessage = "找不到圖生圖 Profile 指定的模型。"
            return
        }
        guard let modelURL = model.localURL else {
            statusMessage = "找不到圖生圖模型的本機安裝路徑。"
            return
        }

        let recipeSnapshot = recipe
        let projectID = selectedProjectID
        let job = GenerationJob(action: .imageToImage, title: "編輯「\(input.title)」")
        jobs.append(job)
        updateJob(job.id) { $0.state = .running }
        let service = imageToImageService
        let request = ImageToImageRequest(
            projectID: projectID,
            sourceAsset: input,
            recipe: recipeSnapshot,
            profile: profile,
            modelURL: modelURL,
            quantization: model.quantization
        )

        jobTasks[job.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                var output = try await service.generate(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(job.id, value: value)
                        }
                    }
                )
                try Task.checkCancellation()
                let operationID = UUID()
                output.operationID = operationID
                assets.append(output)
                operations.append(
                    WorkflowOperation(
                        id: operationID,
                        projectID: projectID,
                        action: .imageToImage,
                        inputAssetID: input.id,
                        outputAssetIDs: [output.id],
                        recipeID: recipeSnapshot.id,
                        profileSnapshot: profile
                    )
                )
                comparisonAssetID = input.id
                selectedAssetID = output.id
                previewMode = .compare
                updateJob(job.id) {
                    $0.progress = 1
                    $0.state = .completed
                }
                statusMessage = "圖生圖完成，已切換至前後比較。"
            } catch is CancellationError {
                updateJob(job.id) { $0.state = .cancelled }
            } catch {
                let message = error.localizedDescription
                updateJob(job.id) {
                    $0.state = .failed
                    $0.errorMessage = message
                }
                statusMessage = "圖生圖失敗：\(message)"
            }
            finishJobTask(job.id)
        }
    }

    func upscaleSelected() {
        guard ensureInferenceIdle() else { return }
        guard let input = selectedSourceImage else {
            statusMessage = "請先匯入或選取一張圖片。"
            return
        }
        guard let profile = activeProfile(for: .upscale) else {
            statusMessage = "請先設定 Upscale Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let executionProfile = runtimeProfile(from: profile) else { return }
        let scale = executionProfile.defaults.upscaleScale ?? 4
        let job = GenerationJob(action: .upscale, title: "放大「\(input.title)」\(scale)×")
        jobs.append(job)
        updateJob(job.id) { $0.state = .running }
        let service = upscaleService
        let request = UpscaleRequest(asset: input, profile: executionProfile, scale: scale)

        jobTasks[job.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                var output = try await service.upscale(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(job.id, value: value)
                        }
                    }
                )
                try Task.checkCancellation()

                let operationID = UUID()
                output.operationID = operationID
                assets.append(output)
                operations.append(
                    WorkflowOperation(
                        id: operationID,
                        projectID: selectedProjectID,
                        action: .upscale,
                        inputAssetID: input.id,
                        outputAssetIDs: [output.id],
                        recipeID: input.recipeID,
                        profileSnapshot: profile
                    )
                )
                comparisonAssetID = input.id
                selectedAssetID = output.id
                previewMode = .compare
                updateJob(job.id) {
                    $0.progress = 1
                    $0.state = .completed
                }
                statusMessage = "Upscale 完成，已切換至前後比較。"
            } catch is CancellationError {
                updateJob(job.id) { $0.state = .cancelled }
            } catch {
                let message = error.localizedDescription
                updateJob(job.id) {
                    $0.state = .failed
                    $0.errorMessage = message
                }
                statusMessage = "Upscale 失敗：\(message)"
            }
            finishJobTask(job.id)
        }
    }
}
