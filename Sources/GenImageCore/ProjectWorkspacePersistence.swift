import Foundation

public struct ProjectWorkspaceSnapshot: Codable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var projects: [Project]
    public var selectedProjectID: UUID
    public var assets: [MediaAsset]
    public var operations: [WorkflowOperation]
    public var selectedAssetID: UUID?
    public var comparisonAssetID: UUID?

    public init(
        schemaVersion: Int = currentSchemaVersion,
        projects: [Project],
        selectedProjectID: UUID,
        assets: [MediaAsset],
        operations: [WorkflowOperation],
        selectedAssetID: UUID?,
        comparisonAssetID: UUID?
    ) {
        self.schemaVersion = schemaVersion
        self.projects = projects
        self.selectedProjectID = selectedProjectID
        self.assets = assets
        self.operations = operations
        self.selectedAssetID = selectedAssetID
        self.comparisonAssetID = comparisonAssetID
    }
}

public enum ProjectWorkspacePersistence {
    public struct Restoration: Sendable {
        public let snapshot: ProjectWorkspaceSnapshot?
        public let errorMessage: String?
        public var allowsPersistence: Bool { errorMessage == nil }
        public var allowsCacheCleanup: Bool { errorMessage == nil }
    }

    /// A missing workspace starts a new session; an unreadable one must never
    /// be overwritten or used as an empty asset list for orphan cleanup.
    public static func restore(from url: URL) -> Restoration {
        do {
            return Restoration(snapshot: try load(from: url), errorMessage: nil)
        } catch {
            return Restoration(snapshot: nil, errorMessage: error.localizedDescription)
        }
    }

    public static func defaultURL(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default
    ) -> URL {
        for key in ["GENMEDIA_WORKSPACE_STATE", "GENIMAGE_WORKSPACE_STATE"] {
            if let value = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
               !value.isEmpty {
                return URL(
                    fileURLWithPath: (value as NSString).expandingTildeInPath,
                    isDirectory: false
                ).standardizedFileURL
            }
        }
        return ApplicationSupport.directory(.workspace, fileManager: fileManager)
            .appendingPathComponent("open-projects.json", isDirectory: false)
            .standardizedFileURL
    }

    public static func load(from url: URL) throws -> ProjectWorkspaceSnapshot? {
        let data: Data
        do { data = try Data(contentsOf: url) }
        catch let error as CocoaError where error.code == .fileReadNoSuchFile { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let snapshot = try decoder.decode(ProjectWorkspaceSnapshot.self, from: data)
        guard snapshot.schemaVersion == ProjectWorkspaceSnapshot.currentSchemaVersion else {
            throw ProjectWorkspacePersistenceError.unsupportedSchema(snapshot.schemaVersion)
        }
        return snapshot
    }

    public static func save(_ snapshot: ProjectWorkspaceSnapshot, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(snapshot).write(to: url, options: .atomic)
    }
}

public enum ProjectWorkspacePersistenceError: LocalizedError, Sendable {
    case unsupportedSchema(Int)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedSchema(version):
            "不支援的專案工作區資料版本：\(version)。"
        }
    }
}
