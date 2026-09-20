import Foundation
import ImageIO
import UniformTypeIdentifiers

/// A bounded cache of encoded thumbnails. ImageIO downsamples at decode time;
/// full-resolution pixel buffers never need to enter the web content process.
public final class ImageThumbnailCache: @unchecked Sendable {
    private let cache = NSCache<NSString, NSData>()
    private let lock = NSLock()
    public let maximumPixelSize: Int

    public init(maximumPixelSize: Int = 384, costLimit: Int = 16 * 1_024 * 1_024) {
        self.maximumPixelSize = max(1, maximumPixelSize)
        cache.totalCostLimit = costLimit
        cache.countLimit = 256
    }

    public func data(for url: URL) throws -> Data {
        let attributes = try url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
        let key = "\(url.standardizedFileURL.path)|\(attributes.contentModificationDate?.timeIntervalSince1970 ?? 0)|\(attributes.fileSize ?? 0)" as NSString
        // Serialize cache misses to bound simultaneous ImageIO decode buffers.
        lock.lock()
        defer { lock.unlock() }
        if let data = cache.object(forKey: key) { return data as Data }
        guard let source = CGImageSourceCreateWithURL(url as CFURL,
            [kCGImageSourceShouldCache: false] as CFDictionary),
            let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: maximumPixelSize,
                kCGImageSourceShouldCacheImmediately: true
            ] as CFDictionary) else { throw CocoaError(.fileReadCorruptFile) }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil)
        else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw CocoaError(.fileWriteUnknown) }
        let data = output as Data
        cache.setObject(data as NSData, forKey: key, cost: data.count)
        return data
    }
}
