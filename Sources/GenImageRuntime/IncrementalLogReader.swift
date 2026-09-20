import Foundation

/// Keeps only an unfinished line, never the complete log. A reader belongs to
/// one consumer; callers that share it must serialize access.
final class IncrementalLogReader {
    let url: URL
    private(set) var offset: UInt64 = 0
    private var pending = Data()
    private var discardingLongLine = false
    private let maximumLineBytes = 64 * 1_024

    init(url: URL) { self.url = url }

    func readLines(finalize: Bool = false) throws -> (lines: [Data], reset: Bool, hasMore: Bool) {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let size = try handle.seekToEnd()
        let reset = size < offset
        if reset {
            offset = 0
            pending.removeAll(keepingCapacity: true)
            discardingLongLine = false
        }
        try handle.seek(toOffset: offset)
        var lines: [Data] = []
        // Bound each poll's work even if a subprocess emits a flood of logs.
        let end = min(size, offset + 1_024 * 1_024)
        while offset < end {
            guard let data = try handle.read(upToCount: Int(min(64 * 1_024, end - offset))),
                  !data.isEmpty else { break }
            offset += UInt64(data.count)
            var start = data.startIndex
            for index in data.indices where data[index] == 10 || data[index] == 13 {
                append(data[start..<index])
                if !discardingLongLine, !pending.isEmpty { lines.append(pending) }
                pending = Data()
                discardingLongLine = false
                start = data.index(after: index)
            }
            append(data[start..<data.endIndex])
        }
        let hasMore = offset < size
        // Only a stopped writer can make its unterminated final line complete.
        if finalize, !hasMore {
            if !discardingLongLine, !pending.isEmpty { lines.append(pending) }
            pending = Data()
            discardingLongLine = false
        }
        return (lines, reset, hasMore)
    }

    private func append(_ bytes: Data.SubSequence) {
        guard !discardingLongLine else { return }
        guard pending.count + bytes.count <= maximumLineBytes else {
            pending.removeAll(keepingCapacity: false)
            discardingLongLine = true
            return
        }
        pending.append(bytes)
    }
}

final class RuntimeLogProgress: @unchecked Sendable {
    private let lock = NSLock()
    private let reader: IncrementalLogReader
    private var latest: Double?
    private var maximum: Double?

    init(url: URL) { reader = IncrementalLogReader(url: url) }

    func value(useMaximum: Bool, parse: (Data) -> Double?) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        if let batch = try? reader.readLines() {
            if batch.reset { latest = nil; maximum = nil }
            for line in batch.lines {
                guard let value = parse(line), value.isFinite else { continue }
                latest = value
                maximum = max(maximum ?? value, value)
            }
        }
        return useMaximum ? maximum : latest
    }
}
