import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// Turns decoded pixels into files on disk.
public enum MiniMaxH3VideoWriter {
    public enum WriterError: LocalizedError {
        case unexpectedShape([Int])
        case imageCreationFailed(Int)
        case writerSetupFailed(String)

        public var errorDescription: String? {
            switch self {
            case let .unexpectedShape(shape):
                "影格張量形狀不正確：\(shape)，預期 [1, 3, T, H, W]。"
            case let .imageCreationFailed(index):
                "第 \(index) 格影像建立失敗。"
            case let .writerSetupFailed(detail):
                "影片寫出失敗：\(detail)。"
            }
        }
    }

    /// Convert `[1, 3, T, H, W]` in [0, 1] into one CGImage per frame.
    public static func images(from pixels: MLXArray) throws -> [CGImage] {
        guard pixels.ndim == 5, pixels.shape[0] == 1, pixels.shape[1] == 3 else {
            throw WriterError.unexpectedShape(pixels.shape)
        }
        let frames = pixels.shape[2]
        let height = pixels.shape[3]
        let width = pixels.shape[4]

        // [1, 3, T, H, W] -> [T, H, W, 3] and quantize once for all frames.
        let planar = pixels[0].transposed(1, 2, 3, 0)
        let bytes = MLX.clip(planar * 255.0, min: 0, max: 255)
            .asType(.uint8)
        MLX.eval(bytes)
        let flat = bytes.asArray(UInt8.self)

        var result: [CGImage] = []
        result.reserveCapacity(frames)
        let bytesPerRow = width * 4
        for frame in 0 ..< frames {
            var rgba = [UInt8](repeating: 255, count: width * height * 4)
            let base = frame * height * width * 3
            for pixel in 0 ..< (width * height) {
                rgba[pixel * 4] = flat[base + pixel * 3]
                rgba[pixel * 4 + 1] = flat[base + pixel * 3 + 1]
                rgba[pixel * 4 + 2] = flat[base + pixel * 3 + 2]
            }
            guard let provider = CGDataProvider(data: Data(rgba) as CFData),
                  let image = CGImage(
                      width: width, height: height,
                      bitsPerComponent: 8, bitsPerPixel: 32,
                      bytesPerRow: bytesPerRow,
                      space: CGColorSpaceCreateDeviceRGB(),
                      bitmapInfo: CGBitmapInfo(
                          rawValue: CGImageAlphaInfo.noneSkipLast.rawValue
                      ),
                      provider: provider, decode: nil,
                      shouldInterpolate: false, intent: .defaultIntent
                  ) else {
                throw WriterError.imageCreationFailed(frame)
            }
            result.append(image)
        }
        return result
    }

    /// Write every frame as a PNG named `frame-000.png` and return the paths.
    @discardableResult
    public static func writePNGs(
        _ images: [CGImage],
        to directory: URL
    ) throws -> [URL] {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true
        )
        var written: [URL] = []
        for (index, image) in images.enumerated() {
            let url = directory.appendingPathComponent(
                String(format: "frame-%03d.png", index)
            )
            guard let destination = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil
            ) else {
                throw WriterError.imageCreationFailed(index)
            }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else {
                throw WriterError.imageCreationFailed(index)
            }
            written.append(url)
        }
        return written
    }

    /// Write an H.264 MP4 at `frameRate` frames per second.
    public static func writeMP4(
        _ images: [CGImage],
        to url: URL,
        frameRate: Int = 24
    ) throws {
        guard let first = images.first else {
            throw WriterError.writerSetupFailed("沒有影格")
        }
        try? FileManager.default.removeItem(at: url)

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: first.width,
            AVVideoHeightKey: first.height
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String:
                    Int(kCVPixelFormatType_32ARGB),
                kCVPixelBufferWidthKey as String: first.width,
                kCVPixelBufferHeightKey as String: first.height
            ]
        )
        guard writer.canAdd(input) else {
            throw WriterError.writerSetupFailed("無法加入 video input")
        }
        writer.add(input)
        guard writer.startWriting() else {
            throw WriterError.writerSetupFailed(
                writer.error?.localizedDescription ?? "startWriting"
            )
        }
        writer.startSession(atSourceTime: .zero)

        for (index, image) in images.enumerated() {
            guard let pool = adaptor.pixelBufferPool else {
                throw WriterError.writerSetupFailed("pixelBufferPool 不可用")
            }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let pixelBuffer = buffer else {
                throw WriterError.writerSetupFailed("無法建立 pixel buffer")
            }
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            if let context = CGContext(
                data: CVPixelBufferGetBaseAddress(pixelBuffer),
                width: image.width, height: image.height,
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue
            ) {
                context.draw(
                    image,
                    in: CGRect(x: 0, y: 0, width: image.width, height: image.height)
                )
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.005)
            }
            let time = CMTime(value: CMTimeValue(index), timescale: CMTimeScale(frameRate))
            adaptor.append(pixelBuffer, withPresentationTime: time)
        }

        input.markAsFinished()
        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting { semaphore.signal() }
        semaphore.wait()
        if writer.status != .completed {
            throw WriterError.writerSetupFailed(
                writer.error?.localizedDescription ?? "status \(writer.status.rawValue)"
            )
        }
    }
}
