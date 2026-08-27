import Foundation

/// App 在 Application Support 底下的資料位置，唯一的定義處。
///
/// 這裡原本分裂成兩個根目錄：模型、Runtime、貼上的圖片與輸出在 `GenImage/`，而工作區索引在
/// `GenMedia/`。同一個 App 兩個根，備份、搬移與清理都會漏掉一半，刪檔前的「是不是我們管理的
/// 檔案」保護也只認得其中一個。
///
/// 統一到 `GenImage/`，而不是改名成與 App 一致的 `GenMedia/`：`Runtime/` 底下的 Python venv
/// 會把絕對路徑寫死在啟動 script 裡（`'exec' '…/GenImage/Runtime/ltx-2-mlx/.venv/bin/python3'`），
/// 改名會讓已安裝的 LTX 與 MiniMax Runtime 直接失效；`Models/` 也可能是數十 GB。改名要付的代價
/// 遠大於它解決的問題，所以目錄名稱維持不變，改由這個型別統一定義。
public enum ApplicationSupport: Sendable {
    public static let directoryName = "GenImage"
    /// 曾經寫入過、需要接回來的舊根目錄。
    public static let legacyDirectoryNames = ["GenMedia"]

    public enum Subdirectory: String, CaseIterable, Sendable {
        /// 下載安裝的模型權重。
        case models = "Models"
        /// 外部 Runtime（含內嵌絕對路徑的 Python venv）。
        case runtime = "Runtime"
        /// 從剪貼簿貼上的圖片。
        case pasted = "Pasted"
        /// FFmpeg 為 WebKit 建立的可播放媒體代理。
        case mediaCache = "MediaCache"
        /// 未指定輸出目錄時的預設輸出位置。
        case generated = "Generated"
        /// 開啟中生成專案的索引。
        case workspace = "Workspace"
    }

    public static func rootURL(fileManager: FileManager = .default) -> URL {
        supportURL(fileManager: fileManager)
            .appendingPathComponent(directoryName, isDirectory: true)
            .standardizedFileURL
    }

    public static func directory(
        _ subdirectory: Subdirectory,
        fileManager: FileManager = .default
    ) -> URL {
        rootURL(fileManager: fileManager)
            .appendingPathComponent(subdirectory.rawValue, isDirectory: true)
            .standardizedFileURL
    }

    public static func legacyRootURLs(fileManager: FileManager = .default) -> [URL] {
        legacyDirectoryNames.map {
            supportURL(fileManager: fileManager)
                .appendingPathComponent($0, isDirectory: true)
                .standardizedFileURL
        }
    }

    /// 這個檔案是否位於 App 管理的目錄內 —— 刪檔前用來確認不會動到使用者自己的檔案。
    public static func managesFile(at url: URL, fileManager: FileManager = .default) -> Bool {
        let root = rootURL(fileManager: fileManager).resolvingSymlinksInPath().standardizedFileURL
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return url.resolvingSymlinksInPath().standardizedFileURL.path.hasPrefix(rootPath)
    }

    /// 找出 MediaCache 中沒有對應工作區資產的暫存代理檔案。
    ///
    /// 只回報檔名主體是 UUID 的一般檔案，避免清掉非本 App 建立的內容或仍在寫入的 `.part` 檔案。
    public static func orphanMediaCacheFiles(
        in directory: URL,
        referencedAssetIDs: Set<UUID>,
        fileManager: FileManager = .default
    ) -> [URL] {
        guard let entries = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        return entries.filter { entry in
            guard let resourceValues = try? entry.resourceValues(forKeys: [.isRegularFileKey]),
                  resourceValues.isRegularFile == true,
                  let assetID = UUID(uuidString: entry.deletingPathExtension().lastPathComponent)
            else {
                return false
            }
            return !referencedAssetIDs.contains(assetID)
        }
    }

    /// 把舊根目錄的內容接到現在的根目錄下，回傳實際搬移的項目名稱。
    ///
    /// 只搬移目前根目錄還沒有的項目 —— 兩邊都有時保留現有的，不做合併也不覆蓋。搬完後清掉舊
    /// 根目錄底下剩下的空目錄，舊根目錄自己空了也一併移除。可重複執行。
    @discardableResult
    public static func adoptLegacyDirectories(
        fileManager: FileManager = .default,
        root: URL? = nil,
        legacyRoots: [URL]? = nil
    ) -> [String] {
        let destination = root ?? rootURL(fileManager: fileManager)
        let sources = legacyRoots ?? legacyRootURLs(fileManager: fileManager)
        var adopted: [String] = []

        for source in sources {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: source.path, isDirectory: &isDirectory),
                  isDirectory.boolValue,
                  source.standardizedFileURL != destination.standardizedFileURL
            else { continue }

            let entries = (try? fileManager.contentsOfDirectory(atPath: source.path)) ?? []
            for entry in entries where !entry.hasPrefix(".") {
                let from = source.appendingPathComponent(entry)
                let to = destination.appendingPathComponent(entry)
                if fileManager.fileExists(atPath: to.path) {
                    removeIfEmptyDirectory(from, fileManager: fileManager)
                    continue
                }
                try? fileManager.createDirectory(at: destination, withIntermediateDirectories: true)
                guard (try? fileManager.moveItem(at: from, to: to)) != nil else { continue }
                adopted.append(entry)
            }
            removeIfEmptyDirectory(source, fileManager: fileManager)
        }
        return adopted
    }

    private static func supportURL(fileManager: FileManager) -> URL {
        fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory
    }

    private static func removeIfEmptyDirectory(_ url: URL, fileManager: FileManager) {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue,
              let entries = try? fileManager.contentsOfDirectory(atPath: url.path),
              entries.allSatisfy({ $0 == ".DS_Store" })
        else { return }
        try? fileManager.removeItem(at: url)
    }
}
