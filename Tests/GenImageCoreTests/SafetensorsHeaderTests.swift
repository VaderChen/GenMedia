import Foundation
import Testing
@testable import GenImageCore

struct SafetensorsHeaderTests {
    @Test func readsOnlyMetadataFromAHugeSparseWeightFile() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("header-\(UUID())")
        defer { try? FileManager.default.removeItem(at: file) }
        let json = Data(#"{"test.lora_A":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#.utf8)
        var count = UInt64(json.count).littleEndian
        var data = withUnsafeBytes(of: &count) { Data($0) }
        data.append(json)
        try data.write(to: file)
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: 8 * 1_024 * 1_024 * 1_024)
        try handle.close()
        let header = try SafetensorsHeader.read(from: file)
        #expect(header.jsonData == json)
        #expect(header.payloadOffset == UInt64(json.count + 8))
        #expect(header.fileSize == 8 * 1_024 * 1_024 * 1_024)
        #expect(try header.dictionary()["test.lora_A"] != nil)
    }

    @Test(arguments: [UInt64.max, UInt64(SafetensorsHeader.maximumHeaderBytes + 1), 32, 0])
    func invalidLengthsAreRejectedBeforeAllocation(count: UInt64) throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("header-\(UUID())")
        defer { try? FileManager.default.removeItem(at: file) }
        var length = count.littleEndian
        try withUnsafeBytes(of: &length) { try Data($0).write(to: file) }
        #expect(throws: (any Error).self) { try SafetensorsHeader.read(from: file) }
    }
}
