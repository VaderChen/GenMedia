import Darwin
import Foundation
import GenImageCore

enum ZImageLoRAAdapterNormalizer {
    static func normalize(_ url: URL, cacheDirectory: URL = FileManager.default.temporaryDirectory
        .appendingPathComponent("GenImage-LoRAAdapters", isDirectory: true)) throws -> URL {
        try Task.checkCancellation()
        let originalSignature = try signature(url)
        let source = try SafetensorsHeader.read(from: url)
        var header = try source.dictionary()
        var renamed = false
        for key in Array(header.keys) where key.hasSuffix(".lora_A") || key.hasSuffix(".lora_B") {
            let normalized = "\(key).weight"
            guard header[normalized] == nil, let value = header.removeValue(forKey: key) else { continue }
            header[normalized] = value
            renamed = true
        }
        guard renamed else { return url }

        var metadata = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        metadata.append(Data(repeating: 0x20, count: (8 - metadata.count % 8) % 8))
        guard metadata.count <= SafetensorsHeader.maximumHeaderBytes else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in ("stream-v2|" + originalSignature).utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let destination = cacheDirectory.appendingPathComponent("\(String(hash, radix: 16)).safetensors")
        let expectedSize = UInt64(8 + metadata.count) + source.fileSize - source.payloadOffset
        if let cached = try? SafetensorsHeader.read(from: destination),
           cached.fileSize == expectedSize, cached.jsonData == metadata {
            return destination
        }
        let temporary = cacheDirectory.appendingPathComponent("\(UUID()).partial")
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard FileManager.default.createFile(atPath: temporary.path, contents: nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        let input = try FileHandle(forReadingFrom: url)
        defer { try? input.close() }
        let output = try FileHandle(forWritingTo: temporary)
        defer { try? output.close() }
        var length = UInt64(metadata.count).littleEndian
        try withUnsafeBytes(of: &length) { try output.write(contentsOf: Data($0)) }
        try output.write(contentsOf: metadata)
        try input.seek(toOffset: source.payloadOffset)
        var remaining = source.fileSize - source.payloadOffset
        while remaining > 0 {
            try Task.checkCancellation()
            let copied: Int = try autoreleasepool {
                guard let chunk = try input.read(upToCount: Int(min(remaining, 1_024 * 1_024))),
                      !chunk.isEmpty else { throw CocoaError(.fileReadCorruptFile) }
                try output.write(contentsOf: chunk)
                return chunk.count
            }
            remaining -= UInt64(copied)
        }
        try output.close()
        try Task.checkCancellation()
        guard try signature(url) == originalSignature else { throw CocoaError(.fileReadCorruptFile) }
        // POSIX rename atomically publishes the complete file, including when
        // another App or MCP process has normalized the same source concurrently.
        guard rename(temporary.path, destination.path) == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        return destination
    }

    private static func signature(_ url: URL) throws -> String {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
        let modified = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let inode = (attributes[.systemFileNumber] as? NSNumber)?.uint64Value ?? 0
        return "\(url.standardizedFileURL.path)|\(size)|\(modified)|\(inode)"
    }
}
