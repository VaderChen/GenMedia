import Foundation

/// File operations shared by assets across all workspaces.
public enum MediaAssetFiles {
    public enum RemovalResult: Equatable, Sendable {
        case removed, missing, retained, outsideManagedDirectory
    }

    public static func references(in assets: [MediaAsset]) -> [URL] {
        assets.flatMap { [$0.fileURL, $0.playbackURL].compactMap { $0 } }
    }

    @discardableResult
    public static func remove(
        at url: URL,
        preserving references: [URL],
        within directory: URL? = nil
    ) throws -> RemovalResult {
        guard url.isFileURL else { throw CocoaError(.fileWriteNoPermission) }
        let canonical = url.resolvingSymlinksInPath().standardizedFileURL
        if let directory {
            let root = directory.resolvingSymlinksInPath().standardizedFileURL
            let prefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
            guard canonical.path.hasPrefix(prefix), canonical != root else { return .outsideManagedDirectory }
        }
        guard !references.contains(where: {
            $0.isFileURL && $0.resolvingSymlinksInPath().standardizedFileURL == canonical
        }) else { return .retained }
        let values: URLResourceValues
        do { values = try url.resourceValues(forKeys: [.isDirectoryKey]) }
        catch let error as CocoaError where error.code == .fileNoSuchFile || error.code == .fileReadNoSuchFile {
            return .missing
        }
        guard values.isDirectory != true else { throw CocoaError(.fileWriteInvalidFileName) }
        do { try FileManager.default.removeItem(at: url) }
        catch let error as CocoaError where error.code == .fileNoSuchFile || error.code == .fileReadNoSuchFile {
            return .missing
        }
        return .removed
    }

    public static func rename(
        assetID: UUID, to requestedName: String, in assets: [MediaAsset]
    ) throws -> [MediaAsset] {
        guard let source = assets.first(where: { $0.id == assetID })?.fileURL, source.isFileURL else {
            throw RenameError.fileUnavailable
        }
        let name = requestedName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/"), !name.contains("\0") else {
            throw RenameError.invalidName
        }
        guard let values = try? source.resourceValues(forKeys: [.isDirectoryKey]), values.isDirectory != true else {
            throw RenameError.fileUnavailable
        }
        let fileName = URL(fileURLWithPath: name).pathExtension.isEmpty && !source.pathExtension.isEmpty
            ? "\(name).\(source.pathExtension)" : name
        let destination = source.deletingLastPathComponent().appendingPathComponent(fileName, isDirectory: false)
        let sourceEntry = entryURL(source)
        guard entryURL(destination) != sourceEntry else { return assets }
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw RenameError.destinationExists }
        do { try FileManager.default.moveItem(at: source, to: destination) }
        catch { throw RenameError.moveFailed(error.localizedDescription) }
        return assets.map { asset in
            var updated = asset
            if let url = asset.fileURL, entryURL(url) == sourceEntry {
                updated.fileURL = destination
                updated.title = destination.deletingPathExtension().lastPathComponent
            }
            if let url = asset.playbackURL, entryURL(url) == sourceEntry { updated.playbackURL = destination }
            return updated
        }
    }

    // Moving a symlink moves that directory entry, not its target. Resolve only
    // parent directories so references to the target or another hard link stay put.
    private static func entryURL(_ url: URL) -> URL {
        url.deletingLastPathComponent().resolvingSymlinksInPath()
            .appendingPathComponent(url.lastPathComponent).standardizedFileURL
    }

    public enum RenameError: LocalizedError {
        case fileUnavailable, invalidName, destinationExists, moveFailed(String)

        public var errorDescription: String? {
            switch self {
            case .fileUnavailable: "找不到可重新命名的媒體檔案。"
            case .invalidName: "檔案名稱不可為空白，也不能包含路徑。"
            case .destinationExists: "相同檔名的檔案已存在，請使用其他名稱。"
            case let .moveFailed(message): "無法重新命名檔案：\(message)"
            }
        }
    }
}
