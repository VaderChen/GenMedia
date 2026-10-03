import Foundation
import Testing
@testable import GenImageRuntime

struct IncrementalLogReaderBoundaryTests {
    @Test func preservesBytesAtChunkAndLineLimits() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        let maximum = Data(repeating: 0xff, count: 65_536)
        let unicode = Data("文字 👩🏽‍💻\u{0}尾端".utf8)
        try (maximum + Data("\r\n\n".utf8) + unicode).write(to: url)
        let reader = IncrementalLogReader(url: url)
        #expect(try reader.readLines().lines == [maximum])
        #expect(try reader.readLines(finalize: true).lines == [unicode])
        #expect(try reader.readLines(finalize: true).lines.isEmpty)
    }

    @Test func oversizedSplitLineDoesNotContaminateFollowingLine() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(repeating: 97, count: 65_536).write(to: url)
        let reader = IncrementalLogReader(url: url)
        #expect(try reader.readLines().lines.isEmpty)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: Data("b\rOK\n".utf8))
        #expect(try reader.readLines().lines == [Data("OK".utf8)])
        try handle.truncate(atOffset: 0)
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: Data("new\n".utf8))
        let batch = try reader.readLines()
        #expect(batch.reset)
        #expect(batch.lines == [Data("new".utf8)])
    }
}
