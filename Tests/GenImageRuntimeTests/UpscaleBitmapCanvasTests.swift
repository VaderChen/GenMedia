import CoreGraphics
import CoreImage
import Foundation
import Testing
@testable import GenImageRuntime

struct UpscaleBitmapCanvasTests {
    @Test func unevenEdgeTilesPreserveOrientationAndPixels() throws {
        let width = 7, height = 5
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        var bytes: [UInt8] = []
        for y in 0..<height {
            for x in 0..<width {
                bytes += [x.isMultiple(of: 2) ? 255 : 0, y.isMultiple(of: 2) ? 255 : 0,
                    x == y ? 255 : 0, 255]
            }
        }
        let source = CIImage(bitmapData: Data(bytes), bytesPerRow: width * 4,
            size: CGSize(width: width, height: height), format: .RGBA8, colorSpace: space)
        let renderer = CIContext(options: [.cacheIntermediates: false])
        let canvas = try UpscaleBitmapCanvas(width: width, height: height)
        for y in stride(from: 0, to: height, by: 2) {
            for x in stride(from: 0, to: width, by: 3) {
                let rect = CGRect(x: x, y: y, width: min(3, width - x), height: min(2, height - y))
                let tile = try #require(renderer.createCGImage(source.cropped(to: rect), from: rect))
                canvas.draw(tile, in: rect)
            }
        }
        let output = CIImage(cgImage: try canvas.image())
        var actual = [UInt8](repeating: 0, count: bytes.count)
        renderer.render(output, toBitmap: &actual, rowBytes: width * 4,
            bounds: CGRect(x: 0, y: 0, width: width, height: height), format: .RGBA8, colorSpace: space)
        #expect(actual == bytes)
    }
}
