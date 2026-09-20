import Foundation
import Testing
@testable import GenImageRuntime

struct ModelFileReplacementTests {
    @Test(arguments: ["success", "missingSource", "directory"])
    func reusingWeightsPreservesExistingDataOnFailure(mode: String) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source")
        let destination = root.appendingPathComponent("destination")
        if mode != "missingSource" { try Data("new weights".utf8).write(to: source) }
        let original: URL
        if mode == "directory" {
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            original = destination.appendingPathComponent("sentinel")
        } else {
            original = destination
        }
        try Data("preserve".utf8).write(to: original)
        let replace = {
            try ModelFileReplacement.replace(at: destination) { staging in
                try FileManager.default.linkItem(at: source, to: staging)
            }
        }
        if mode == "success" {
            try replace()
            #expect(try Data(contentsOf: destination) == Data("new weights".utf8))
            #expect(try Data(contentsOf: source) == Data("new weights".utf8))
            let sourceID = try FileManager.default.attributesOfItem(atPath: source.path)[.systemFileNumber] as? NSNumber
            let destinationID = try FileManager.default.attributesOfItem(atPath: destination.path)[.systemFileNumber] as? NSNumber
            #expect(sourceID != nil && sourceID == destinationID)
        } else {
            #expect(throws: (any Error).self) { try replace() }
            #expect(try Data(contentsOf: original) == Data("preserve".utf8))
        }
        let files = try FileManager.default.contentsOfDirectory(atPath: root.path)
        #expect(!files.contains { $0.hasSuffix(".download") })
    }
}
