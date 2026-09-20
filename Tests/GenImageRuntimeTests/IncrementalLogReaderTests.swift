import Foundation
import Testing
@testable import GenImageRuntime

struct IncrementalLogReaderTests {
    @Test func waitsForCompleteLinesAndDoesNotReplayOldProgress() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("log-\(UUID())")
        defer { try? FileManager.default.removeItem(at: url) }
        let log = try RuntimeLog(at: url)
        defer { log.close() }
        log.handle.write(Data("{\"type\":\"progress\",\"value\":0.".utf8))
        #expect(log.latestProgress() == nil)
        log.handle.write(Data("25}\n".utf8))
        #expect(log.latestProgress() == 0.25)
        log.handle.write(Data("noise\n{\"type\":\"progress\",\"value\":0.1}\n".utf8))
        #expect(log.latestProgress() == 0.1)
        #expect(log.latestProgress(useMaximum: true) == 0.25)
        #expect(log.latestProgress() == 0.1)
        try log.handle.truncate(atOffset: 0)
        try log.handle.seek(toOffset: 0)
        log.handle.write(Data("{\"type\":\"progress\",\"value\":0.05}\n".utf8))
        #expect(log.latestProgress(useMaximum: true) == 0.05)
    }

    @Test func skipsOversizedLinesAndReadsOnlyNewBytes() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("log-\(UUID())")
        defer { try? FileManager.default.removeItem(at: url) }
        let data = Data(repeating: 65, count: 100_000) + Data("\nOK\r\n".utf8)
        try data.write(to: url)
        let reader = IncrementalLogReader(url: url)
        #expect(try reader.readLines().lines == [Data("OK".utf8)])
        #expect(try reader.readLines().lines.isEmpty)
        #expect(reader.offset == data.count)
    }

    @Test func finalDrainStaysBoundedAndEmitsTheLastLineOnlyOnce() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("log-\(UUID())")
        defer { try? FileManager.default.removeItem(at: url) }
        let noise = Data(repeating: 65, count: 2 * 1_024 * 1_024)
        try (noise + Data("\ncompleted".utf8)).write(to: url)
        let reader = IncrementalLogReader(url: url)
        let first = try reader.readLines(finalize: true)
        #expect(first.hasMore && first.lines.isEmpty)
        #expect(reader.offset == 1_024 * 1_024)
        let second = try reader.readLines(finalize: true)
        #expect(second.hasMore && second.lines.isEmpty)
        #expect(reader.offset == 2 * 1_024 * 1_024)
        let last = try reader.readLines(finalize: true)
        #expect(!last.hasMore)
        #expect(last.lines == [Data("completed".utf8)])
        #expect(try reader.readLines(finalize: true).lines.isEmpty)
    }

}
