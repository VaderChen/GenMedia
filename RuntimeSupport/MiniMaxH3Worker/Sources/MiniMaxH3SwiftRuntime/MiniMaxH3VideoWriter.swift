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

    /// Interleaved stereo PCM plus its sample rate.
    public struct Audio {
        /// `[1, channels, samples]` in [-1, 1], as the audio VAE produces it.
        public var waveform: MLXArray
        public var sampleRate: Int

        public init(waveform: MLXArray, sampleRate: Int) {
            self.waveform = waveform
            self.sampleRate = sampleRate
        }
    }

    /// Write an H.264 MP4 at `frameRate` frames per second, optionally with an
    /// AAC audio track.
    ///
    /// The audio is trimmed or zero-padded to the video's exact duration: the
    /// two streams are generated on separate schedules and their lengths only
    /// agree to within a latent frame.
    public static func writeMP4(
        _ images: [CGImage],
        to url: URL,
        frameRate: Int = 24,
        audio: Audio? = nil
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

        var audioInput: AVAssetWriterInput?
        var audioSamples: [Float] = []
        var audioChannels = 0
        if let audio {
            let prepared = try preparedAudio(
                audio, frameCount: images.count, frameRate: frameRate
            )
            audioSamples = prepared.interleaved
            audioChannels = prepared.channels
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: audio.sampleRate,
                AVNumberOfChannelsKey: prepared.channels,
                AVEncoderBitRateKey: 128_000
            ]
            let track = AVAssetWriterInput(mediaType: .audio, outputSettings: settings)
            track.expectsMediaDataInRealTime = false
            guard writer.canAdd(track) else {
                throw WriterError.writerSetupFailed("無法加入 audio input")
            }
            writer.add(track)
            audioInput = track
        }

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

        if let audioInput, let audio, !audioSamples.isEmpty {
            let buffer = try audioSampleBuffer(
                interleaved: audioSamples,
                channels: audioChannels,
                sampleRate: audio.sampleRate
            )
            while !audioInput.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.005)
            }
            guard audioInput.append(buffer) else {
                throw WriterError.writerSetupFailed(
                    writer.error?.localizedDescription ?? "audio append"
                )
            }
            audioInput.markAsFinished()
        }

        let semaphore = DispatchSemaphore(value: 0)
        writer.finishWriting { semaphore.signal() }
        semaphore.wait()
        if writer.status != .completed {
            throw WriterError.writerSetupFailed(
                writer.error?.localizedDescription ?? "status \(writer.status.rawValue)"
            )
        }
    }

    /// Interleave the waveform and match it to the video's duration.
    private static func preparedAudio(
        _ audio: Audio,
        frameCount: Int,
        frameRate: Int
    ) throws -> (interleaved: [Float], channels: Int) {
        let waveform = audio.waveform.ndim == 3
            ? audio.waveform[0]
            : audio.waveform
        guard waveform.ndim == 2 else {
            throw WriterError.unexpectedShape(audio.waveform.shape)
        }
        let channels = waveform.shape[0]
        let available = waveform.shape[1]
        let wanted = sampleCount(
            frameCount: frameCount, frameRate: frameRate,
            sampleRate: audio.sampleRate
        )

        let clipped = MLX.clip(waveform.asType(.float32), min: -1, max: 1)
        MLX.eval(clipped)
        let planar = clipped.asArray(Float.self)

        // Interleave, zero-filling if the audio is short of the video.
        var interleaved = [Float](repeating: 0, count: wanted * channels)
        let copyCount = min(available, wanted)
        for channel in 0 ..< channels {
            let base = channel * available
            for sample in 0 ..< copyCount {
                interleaved[sample * channels + channel] = planar[base + sample]
            }
        }
        return (interleaved, channels)
    }

    /// Samples needed to cover the video's exact duration.
    ///
    /// The two streams denoise on separate shifted schedules, so the audio VAE's
    /// output only matches the video length to within one latent frame (20 ms);
    /// this is what the trim/pad targets.
    static func sampleCount(
        frameCount: Int,
        frameRate: Int,
        sampleRate: Int
    ) -> Int {
        max(0, Int(
            (Double(frameCount) / Double(max(frameRate, 1)) * Double(sampleRate))
                .rounded()
        ))
    }

    /// Wrap interleaved float PCM in a CMSampleBuffer the writer can transcode.
    private static func audioSampleBuffer(
        interleaved: [Float],
        channels: Int,
        sampleRate: Int
    ) throws -> CMSampleBuffer {
        var description = AudioStreamBasicDescription(
            mSampleRate: Float64(sampleRate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(4 * channels),
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(4 * channels),
            mChannelsPerFrame: UInt32(channels),
            mBitsPerChannel: 32,
            mReserved: 0
        )
        var format: CMAudioFormatDescription?
        guard CMAudioFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            asbd: &description,
            layoutSize: 0, layout: nil, magicCookieSize: 0, magicCookie: nil,
            extensions: nil, formatDescriptionOut: &format
        ) == noErr, let format else {
            throw WriterError.writerSetupFailed("無法建立音訊格式描述")
        }

        let byteCount = interleaved.count * MemoryLayout<Float>.size
        var block: CMBlockBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault, memoryBlock: nil,
            blockLength: byteCount, blockAllocator: kCFAllocatorDefault,
            customBlockSource: nil, offsetToData: 0, dataLength: byteCount,
            flags: 0, blockBufferOut: &block
        ) == noErr, let block else {
            throw WriterError.writerSetupFailed("無法建立音訊緩衝區")
        }
        let copied = interleaved.withUnsafeBytes { pointer -> OSStatus in
            CMBlockBufferReplaceDataBytes(
                with: pointer.baseAddress!, blockBuffer: block,
                offsetIntoDestination: 0, dataLength: byteCount
            )
        }
        guard copied == noErr else {
            throw WriterError.writerSetupFailed("無法寫入音訊緩衝區")
        }

        let frameCount = interleaved.count / max(channels, 1)
        var buffer: CMSampleBuffer?
        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: CMTimeScale(sampleRate)),
            presentationTimeStamp: .zero,
            decodeTimeStamp: .invalid
        )
        guard CMSampleBufferCreate(
            allocator: kCFAllocatorDefault, dataBuffer: block, dataReady: true,
            makeDataReadyCallback: nil, refcon: nil, formatDescription: format,
            sampleCount: CMItemCount(frameCount), sampleTimingEntryCount: 1,
            sampleTimingArray: &timing, sampleSizeEntryCount: 1,
            sampleSizeArray: [MemoryLayout<Float>.size * channels],
            sampleBufferOut: &buffer
        ) == noErr, let buffer else {
            throw WriterError.writerSetupFailed("無法建立音訊取樣緩衝")
        }
        return buffer
    }
}
