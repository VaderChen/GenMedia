import Foundation

public enum ModelStorage {
    public static let environmentKey = "GENIMAGE_MODEL_ROOT"

    public static var defaultRootURL: URL {
        ApplicationSupport.directory(.models)
    }

    public static func rootURL(
        explicitPath: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL {
        let path = [explicitPath, environment[environmentKey]]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first { !$0.isEmpty }
        guard let path else { return defaultRootURL }
        return URL(
            fileURLWithPath: (path as NSString).expandingTildeInPath,
            isDirectory: true
        ).standardizedFileURL
    }
}
