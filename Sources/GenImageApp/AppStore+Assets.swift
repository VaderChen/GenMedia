import Foundation
import GenImageCore

// 工作區資產的匯入、選取與移除。
extension AppStore {
    func selectAsset(_ id: UUID) {
        if selectedAssetID != id {
            comparisonAssetID = selectedAssetID
            selectedAssetID = id
        }
    }

    func renameAsset(_ id: UUID, toFileName requestedName: String) throws {
        try ensureAssetMutationAllowed()
        assets = try MediaAssetFiles.rename(assetID: id, to: requestedName, in: assets)
        if let name = assets.first(where: { $0.id == id })?.fileURL?.lastPathComponent {
            statusMessage = "已將檔案重新命名為「\(name)」。"
        }
    }

    func removeAsset(
        _ id: UUID,
        selecting replacementID: UUID?,
        deleteFile: Bool = false
    ) throws {
        try ensureAssetMutationAllowed()
        guard let removedAsset = assets.first(where: { $0.id == id }) else { return }

        operations = operations.compactMap { operation in
            let referencedInput = operation.inputAssetID == id
            let referencedInputs = operation.inputAssetIDs?.contains(id) == true
            let referencedOutput = operation.outputAssetIDs.contains(id)
            guard referencedInput || referencedInputs || referencedOutput else { return operation }

            var updated = operation
            if referencedInput { updated.inputAssetID = nil }
            if referencedInputs {
                updated.inputAssetIDs?.removeAll { $0 == id }
                if updated.inputAssetIDs?.isEmpty == true { updated.inputAssetIDs = nil }
            }
            updated.outputAssetIDs.removeAll { $0 == id }

            if updated.outputAssetIDs.isEmpty,
               operation.action != .describe || referencedInput {
                return nil
            }
            return updated
        }

        assets.removeAll { $0.id == id }
        for index in assets.indices where assets[index].parentAssetID == id {
            assets[index].parentAssetID = nil
        }

        if selectedAssetID == id {
            selectedAssetID = replacementID.flatMap { replacement in
                assets.contains(where: { $0.id == replacement }) ? replacement : nil
            }
        }
        if comparisonAssetID == id || comparisonAssetID == selectedAssetID {
            comparisonAssetID = nil
        }

        let compatibilityRemovalError = removeCompatibilityFiles(for: [removedAsset])
        var sourceRemovalError: Error?
        var sourceRetained = false
        if deleteFile, let source = removedAsset.fileURL {
            do {
                sourceRetained = try MediaAssetFiles.remove(at: source,
                    preserving: MediaAssetFiles.references(in: assets)) == .retained
            } catch { sourceRemovalError = error }
        }
        if let sourceRemovalError {
            statusMessage = "已從工作區移除「\(removedAsset.title)」，但無法刪除檔案：\(sourceRemovalError.localizedDescription)"
        } else if let compatibilityRemovalError {
            statusMessage = "已從工作區移除「\(removedAsset.title)」，但無法清除媒體相容快取：\(compatibilityRemovalError.localizedDescription)"
        } else if sourceRetained {
            statusMessage = "已從工作區移除「\(removedAsset.title)」；檔案仍由其他資產使用，已保留。"
        } else if deleteFile {
            statusMessage = "已從工作區移除並刪除「\(removedAsset.title)」。"
        } else {
            statusMessage = "已從工作區移除「\(removedAsset.title)」；檔案仍保留於磁碟。"
        }
    }

    func closeWorkspaceProject(assetIDs: [UUID]) throws {
        try ensureAssetMutationAllowed()
        let closedAssetIDs = Set(assetIDs).intersection(Set(assets.map(\.id)))
        guard !closedAssetIDs.isEmpty else { return }
        let closedAssets = assets.filter { closedAssetIDs.contains($0.id) }

        operations.removeAll { operation in
            operation.inputAssetID.map(closedAssetIDs.contains) == true
                || operation.inputAssetIDs.map { !closedAssetIDs.isDisjoint(with: $0) } == true
                || !closedAssetIDs.isDisjoint(with: operation.outputAssetIDs)
        }
        assets.removeAll { closedAssetIDs.contains($0.id) }
        for index in assets.indices where assets[index].parentAssetID.map(closedAssetIDs.contains) == true {
            assets[index].parentAssetID = nil
        }
        if selectedAssetID.map(closedAssetIDs.contains) == true {
            selectedAssetID = nil
        }
        if comparisonAssetID.map(closedAssetIDs.contains) == true {
            comparisonAssetID = nil
        }
        _ = removeCompatibilityFiles(for: closedAssets)
        statusMessage = "已關閉生成專案分頁；\(closedAssetIDs.count) 個結果已從工作區移除，輸出檔案仍保留於磁碟。"
    }

    @discardableResult
    func importImage(url: URL, pixelWidth: Int, pixelHeight: Int) -> UUID {
        let asset = MediaAsset(
            projectID: selectedProjectID,
            kind: .imported,
            title: url.deletingPathExtension().lastPathComponent,
            fileURL: url,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight
        )
        assets.append(asset)
        operations.append(
            WorkflowOperation(
                projectID: selectedProjectID,
                action: .importImage,
                outputAssetIDs: [asset.id]
            )
        )
        selectAsset(asset.id)
        statusMessage = "已匯入「\(asset.title)」；可以執行圖生文、圖生圖、圖生影或 Upscale。"
        return asset.id
    }

    @discardableResult
    func importMedia(
        id: UUID = UUID(),
        url: URL,
        playbackURL: URL? = nil,
        kind: AssetKind,
        pixelWidth: Int,
        pixelHeight: Int,
        durationSeconds: Double,
        compatibilityPrepared: Bool = false
    ) -> UUID {
        precondition(kind == .importedVideo || kind == .importedAudio)
        let asset = MediaAsset(
            id: id,
            projectID: selectedProjectID,
            kind: kind,
            title: url.deletingPathExtension().lastPathComponent,
            fileURL: url,
            playbackURL: playbackURL,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight,
            mediaDurationSeconds: durationSeconds
        )
        assets.append(asset)
        operations.append(
            WorkflowOperation(
                projectID: selectedProjectID,
                action: .importMedia,
                outputAssetIDs: [asset.id]
            )
        )
        selectAsset(asset.id)
        statusMessage = compatibilityPrepared
            ? "已匯入「\(asset.title)」，並完成 FFmpeg 相容處理；可以播放或生成字幕。"
            : "已匯入「\(asset.title)」；可以播放或生成字幕。"
        return asset.id
    }

    /// Only unreferenced playback proxies inside MediaCache may be removed
    /// automatically. Sources remain protected even after their asset is closed.
    @discardableResult
    func removeCompatibilityFiles(for removedAssets: [MediaAsset]) -> Error? {
        let references = MediaAssetFiles.references(in: assets) + removedAssets.compactMap(\.fileURL)
        var firstError: Error?
        for url in Set(removedAssets.compactMap(\.playbackURL)) {
            do {
                try MediaAssetFiles.remove(at: url, preserving: references,
                    within: ApplicationSupport.directory(.mediaCache))
            } catch { if firstError == nil { firstError = error } }
        }
        return firstError
    }

    private func ensureAssetMutationAllowed() throws {
        guard !jobs.contains(where: { [.queued, .running, .cancelling].contains($0.state) }) else {
            throw NSError(domain: "GenImage.AssetMutation", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "任務執行或取消中，完成後才能移除或重新命名媒體。"])
        }
    }
}
