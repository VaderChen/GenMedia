import CoreImage
import Foundation
import MLX
import MLXVLM
import Testing
@testable import QwenImage21Runtime

@Suite(.serialized)
struct Qwen21PixelTests {
    @Test func pngInputRetainsRowOrderAndStraightAlpha() throws {
        let values: [Float] = [1, 0, 0, 1, 0, 1, 0, 1,
                               0, 0, 1, 1, 0.8, 0.4, 0.2, 0.5]
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        defer { try? FileManager.default.removeItem(at: url) }
        try Qwen21Pipeline.writePNG(MLXArray(values).reshaped([1, 2, 2, 4]), to: url, width: 2, height: 2)
        let image = try #require(CIImage(contentsOf: url))
        let actual = Qwen21ImageProcessing.rgba(image)
        #expect(max(abs(actual - MLXArray(values).reshaped([1, 2, 2, 4]))).item(Float.self) < 0.01)
    }

    @Test func visionCompositesTransparentPixelsOverWhiteInSRGB() throws {
        let configuration = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: Data(
            #"{"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"merge_size":2,"patch_size":2,"temporal_patch_size":2,"image_processor_type":"Qwen2VLImageProcessorFast"}"#.utf8))
        let pixel: [Float] = [0, 0, 0, 0.5]
        let values = Array(repeating: pixel, count: 16).flatMap { $0 }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        defer { try? FileManager.default.removeItem(at: url) }
        try Qwen21Pipeline.writePNG(MLXArray(values).reshaped([1, 4, 4, 4]), to: url, width: 4, height: 4)
        let image = try #require(CIImage(contentsOf: url))
        let (pixels, _) = try Qwen21ImageProcessing.visionPixels(image, configuration: configuration)
        // Pillow's white.paste(image, mask=alpha): black at alpha=128 becomes 127/255.
        #expect(max(abs(pixels - (Float(127) / 255 * 2 - 1))).item(Float.self) < 0.01)
    }
}
