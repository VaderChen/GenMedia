import CoreGraphics
import Foundation

/// Tiles are materialized directly into one destination. This avoids retaining
/// a Core Image graph and every prediction's pixel buffer until PNG encoding.
final class UpscaleBitmapCanvas: @unchecked Sendable {
    // CGContext is mutable; serialize access even when passed through a
    // synchronous autorelease pool closure on the inference actor.
    private let lock = NSLock()
    private let bitmap: CGContext

    init(width: Int, height: Int) throws {
        guard width > 0, height > 0,
              let bitmap = CGContext(data: nil, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw CoreMLUpscaleError.cannotCreateTile
        }
        self.bitmap = bitmap
        bitmap.interpolationQuality = .none
        bitmap.setBlendMode(.copy)
    }

    func draw(_ tile: CGImage, in rect: CGRect) {
        lock.lock()
        defer { lock.unlock() }
        bitmap.draw(tile, in: rect)
    }

    func image() throws -> CGImage {
        lock.lock()
        defer { lock.unlock() }
        guard let image = bitmap.makeImage() else { throw CoreMLUpscaleError.cannotCreateTile }
        return image
    }
}
