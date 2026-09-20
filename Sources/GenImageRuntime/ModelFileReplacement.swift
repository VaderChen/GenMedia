import Darwin
import Foundation

/// Preparing on the destination volume keeps existing files intact on failure.
enum ModelFileReplacement {
    static func replace(at destination: URL, prepare: (URL) throws -> Void) throws {
        let staging = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(UUID()).download")
        defer { try? FileManager.default.removeItem(at: staging) }
        try prepare(staging)
        guard rename(staging.path, destination.path) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
    }
}
