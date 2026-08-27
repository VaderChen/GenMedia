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
        guard let index = assets.firstIndex(where: { $0.id == id }),
              let sourceURL = assets[index].fileURL else {
            throw AssetRenameError.fileUnavailable
        }

        let trimmedName = requestedName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty,
              trimmedName != ".",
              trimmedName != "..",
              URL(fileURLWithPath: trimmedName).lastPathComponent == trimmedName else {
            throw AssetRenameError.invalidName
        }
        guard FileManager.default.fileExists(atPath: sourceURL.path) else {
            throw AssetRenameError.fileUnavailable
        }

        let fileName = URL(fileURLWithPath: trimmedName).pathExtension.isEmpty
            && !sourceURL.pathExtension.isEmpty
            ? "\(trimmedName).\(sourceURL.pathExtension)"
            : trimmedName
        let destinationURL = sourceURL
            .deletingLastPathComponent()
            .appendingPathComponent(fileName, isDirectory: false)

        guard destinationURL.standardizedFileURL != sourceURL.standardizedFileURL else { return }
        guard !FileManager.default.fileExists(atPath: destinationURL.path) else {
            throw AssetRenameError.destinationExists
        }

        do {
            try FileManager.default.moveItem(at: sourceURL, to: destinationURL)
        } catch {
            throw AssetRenameError.moveFailed(error.localizedDescription)
        }

        var asset = assets[index]
        asset.fileURL = destinationURL
        if asset.playbackURL?.standardizedFileURL == sourceURL.standardizedFileURL {
            asset.playbackURL = destinationURL
        }
        asset.title = destinationURL.deletingPathExtension().lastPathComponent
        assets[index] = asset
        statusMessage = "已將檔案重新命名為「\(destinationURL.lastPathComponent)」。"
    }

    func removeAsset(
        _ id: UUID,
        selecting replacementID: UUID?,
        deleteFile: Bool = false
    ) {
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

        let compatibilityRemovalError = removeCompatibilityFile(at: removedAsset.playbackURL)
        let sourceRemovalError = deleteFile ? removeAssetFile(at: removedAsset.fileURL) : nil
        if let sourceRemovalError {
            statusMessage = "已從工作區移除「\(removedAsset.title)」，但無法刪除檔案：\(sourceRemovalError.localizedDescription)"
        } else if let compatibilityRemovalError {
            statusMessage = "已從工作區移除「\(removedAsset.title)」，但無法清除媒體相容快取：\(compatibilityRemovalError.localizedDescription)"
        } else if deleteFile {
            statusMessage = "已從工作區移除並刪除「\(removedAsset.title)」。"
        } else {
            statusMessage = "已從工作區移除「\(removedAsset.title)」；檔案仍保留於磁碟。"
        }
    }

    func closeWorkspaceProject(assetIDs: [UUID]) {
        let closedAssetIDs = Set(assetIDs).intersection(Set(assets.map(\.id)))
        guard !closedAssetIDs.isEmpty else { return }
        let compatibilityURLs = assets.compactMap { asset in
            closedAssetIDs.contains(asset.id) ? asset.playbackURL : nil
        }

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
        compatibilityURLs.forEach { _ = removeCompatibilityFile(at: $0) }
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

    private func removeCompatibilityFile(at fileURL: URL?) -> Error? {
        guard let fileURL else { return nil }
        let fileManager = FileManager.default
        guard ApplicationSupport.managesFile(at: fileURL, fileManager: fileManager) else {
            return CocoaError(.fileWriteNoPermission)
        }
        return removeAssetFile(at: fileURL)
    }

    private func removeAssetFile(at fileURL: URL?) -> Error? {
        guard let fileURL else { return nil }
        let fileManager = FileManager.default
        let candidate = fileURL.standardizedFileURL

        guard fileManager.fileExists(atPath: candidate.path) else { return nil }
        guard let values = try? candidate.resourceValues(forKeys: [.isDirectoryKey]),
              values.isDirectory != true else { return CocoaError(.fileWriteInvalidFileName) }
        do {
            try fileManager.removeItem(at: candidate)
            return nil
        } catch {
            return error
        }
    }
}

private enum AssetRenameError: LocalizedError {
    case fileUnavailable
    case invalidName
    case destinationExists
    case moveFailed(String)

    var errorDescription: String? {
        switch self {
        case .fileUnavailable:
            "找不到可重新命名的媒體檔案。"
        case .invalidName:
            "檔案名稱不可為空白，也不能包含路徑。"
        case .destinationExists:
            "相同檔名的檔案已存在，請使用其他名稱。"
        case let .moveFailed(message):
            "無法重新命名檔案：\(message)"
        }
    }
}
