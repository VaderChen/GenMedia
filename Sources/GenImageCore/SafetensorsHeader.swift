import Foundation

/// Read only the length prefix and bounded JSON metadata, never the weight payload.
public struct SafetensorsHeader: Sendable {
    public static let maximumHeaderBytes = 16 * 1_024 * 1_024
    public let jsonData: Data
    public let payloadOffset: UInt64
    public let fileSize: UInt64

    public static func read(from url: URL) throws -> Self {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let size = try handle.seekToEnd()
        try handle.seek(toOffset: 0)
        guard size >= 8, let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let count = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << ($1.offset * 8) }
        guard count > 0, count <= UInt64(maximumHeaderBytes), count <= size - 8,
              let data = try handle.read(upToCount: Int(count)), data.count == Int(count),
              (try JSONSerialization.jsonObject(with: data)) is [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return Self(jsonData: data, payloadOffset: count + 8, fileSize: size)
    }

    public func dictionary() throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return object
    }
}
