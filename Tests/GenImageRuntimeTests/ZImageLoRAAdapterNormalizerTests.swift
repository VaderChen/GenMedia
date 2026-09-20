import Foundation
import GenImageCore
import Testing
@testable import GenImageRuntime

struct ZImageLoRAAdapterNormalizerTests {
    @Test func streamsWeightsAndPublishesOnlyCompleteAdapters() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("normalize-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let input = root.appendingPathComponent("source.safetensors")
        let cache = root.appendingPathComponent("cache")
        let json = Data(#"{"layer.lora_A":{"dtype":"U8","shape":[3145731],"data_offsets":[0,3145731]},"layer.lora_B":{"dtype":"U8","shape":[0],"data_offsets":[3145731,3145731]},"__metadata__":{"keep":"yes"}}"#.utf8)
        var length = UInt64(json.count).littleEndian
        var original = withUnsafeBytes(of: &length) { Data($0) }
        original.append(json)
        let payload = Data((0..<3_145_731).map { UInt8($0 % 251) })
        original.append(payload)
        try original.write(to: input)
        let normalized = try ZImageLoRAAdapterNormalizer.normalize(input, cacheDirectory: cache)
        let header = try SafetensorsHeader.read(from: normalized)
        let object = try header.dictionary()
        #expect(object["layer.lora_A"] == nil)
        #expect(object["layer.lora_B"] == nil)
        #expect(object["layer.lora_A.weight"] != nil)
        #expect(object["layer.lora_B.weight"] != nil)
        #expect((object["__metadata__"] as? [String: String])?["keep"] == "yes")
        #expect(header.payloadOffset % 8 == 0)
        #expect(try Data(contentsOf: normalized).dropFirst(Int(header.payloadOffset)) == payload)
        #expect(try Data(contentsOf: input) == original)
        #expect(try ZImageLoRAAdapterNormalizer.normalize(input, cacheDirectory: cache) == normalized)
        #expect(try FileManager.default.contentsOfDirectory(atPath: cache.path).count == 1)
        // A truncated cached adapter must be rebuilt rather than reused.
        try Data([0]).write(to: normalized)
        #expect(try ZImageLoRAAdapterNormalizer.normalize(input, cacheDirectory: cache) == normalized)
        #expect(try Data(contentsOf: normalized).dropFirst(Int(header.payloadOffset)) == payload)
    }
}
