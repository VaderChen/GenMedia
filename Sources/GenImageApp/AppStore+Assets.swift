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

    func removeAsset(_ id: UUID, selecting replacementID: UUID?) {
        guard let removedAsset = assets.first(where: { $0.id == id }) else { return }

        operations = operations.compactMap { operation in
            let referencedInput = operation.inputAssetID == id
            let referencedOutput = operation.outputAssetIDs.contains(id)
            guard referencedInput || referencedOutput else { return operation }

            var updated = operation
            if referencedInput { updated.inputAssetID = nil }
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

        let fileRemovalError = removeManagedAssetFile(at: removedAsset.fileURL)
        if let fileRemovalError {
            statusMessage = "已從工作區移除「\(removedAsset.title)」，但無法刪除應用程式副本：\(fileRemovalError.localizedDescription)"
        } else {
            statusMessage = "已移除「\(removedAsset.title)」。"
        }
    }

    func closeWorkspaceProject(assetIDs: [UUID]) {
        let closedAssetIDs = Set(assetIDs).intersection(Set(assets.map(\.id)))
        guard !closedAssetIDs.isEmpty else { return }

        operations.removeAll { operation in
            operation.inputAssetID.map(closedAssetIDs.contains) == true
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
        url: URL,
        kind: AssetKind,
        pixelWidth: Int,
        pixelHeight: Int,
        durationSeconds: Double
    ) -> UUID {
        precondition(kind == .importedVideo || kind == .importedAudio)
        let asset = MediaAsset(
            projectID: selectedProjectID,
            kind: kind,
            title: url.deletingPathExtension().lastPathComponent,
            fileURL: url,
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
        statusMessage = "已匯入「\(asset.title)」；可以開始生成字幕。"
        return asset.id
    }

    private func removeManagedAssetFile(at fileURL: URL?) -> Error? {
        guard let fileURL else { return nil }
        let fileManager = FileManager.default
        let candidate = fileURL.resolvingSymlinksInPath().standardizedFileURL

        guard ApplicationSupport.managesFile(at: candidate, fileManager: fileManager),
              fileManager.fileExists(atPath: candidate.path)
        else {
            return nil
        }
        do {
            try fileManager.removeItem(at: candidate)
            return nil
        } catch {
            return error
        }
    }
}
