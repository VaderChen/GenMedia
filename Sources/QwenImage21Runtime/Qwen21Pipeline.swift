import CoreGraphics
import CoreImage
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

public enum Qwen21Pipeline {
    /// Components are loaded in stages; the Qwen3-VL encoder is released before the DiT is loaded.
    public static func generate(_ request: Qwen21Request, progress: (Double) -> Void) throws {
        try request.validate()
        let root = URL(fileURLWithPath: request.modelDirectory, isDirectory: true)
        let reference = try request.inputPath.map { try referenceImage(path: $0, targetArea: request.width * request.height) }
        progress(0.02)
        let conditioning = try encode(root: root, prompt: request.prompt, reference: reference)
        Memory.clearCache()
        progress(0.15)
        let referenceLatents: MLXArray? = try reference.map { image in
            let vae = try Qwen21VAE(directory: root.appendingPathComponent("vae"))
            return try vae.encode(Qwen21ImageProcessing.rgba(image) * 2 - 1)
        }
        Memory.clearCache()
        let layout = try Qwen21Layout(imageSlots: conditioning.imageSlots,
            referenceHeight: reference.map { Int($0.extent.height) / 16 },
            referenceWidth: reference.map { Int($0.extent.width) / 16 },
            height: request.height / 16, width: request.width / 16)
        let scheduler = try readJSON(Qwen21Scheduler.self, at: root.appendingPathComponent("scheduler/scheduler_config.json"))
        let sigmas = try scheduler.sigmas(steps: request.steps, imageTokens: request.height * request.width / 256)
        for (index, output) in request.outputPaths.enumerated() {
            try Task.checkCancellation()
            let base = 0.2 + 0.8 * Double(index) / Double(request.outputPaths.count)
            let span = 0.8 / Double(request.outputPaths.count)
            let latent = try denoise(root: root, request: request, seed: request.seed &+ UInt64(index),
                conditioning: conditioning, reference: referenceLatents, layout: layout, sigmas: sigmas) { step in
                progress(base + span * 0.85 * Double(step) / Double(request.steps))
            }
            Memory.clearCache()
            let rgba = try decode(root: root, latent: latent, height: request.height, width: request.width)
            try writePNG(rgba, to: URL(fileURLWithPath: output), width: request.width, height: request.height)
            Memory.clearCache()
            progress(base + span)
        }
    }

    private static func encode(root: URL, prompt: String, reference: CIImage?) throws -> Qwen21Conditioning {
        try Qwen21TextEncoder(directory: root).encode(prompt: prompt, image: reference)
    }

    private static func denoise(root: URL, request: Qwen21Request, seed: UInt64,
                                conditioning: Qwen21Conditioning, reference: MLXArray?, layout: Qwen21Layout,
                                sigmas: [Float], progress: (Int) -> Void) throws -> MLXArray {
        let transformer = try Qwen21Transformer(directory: root.appendingPathComponent("transformer"))
        var x = MLXRandom.normal([1, request.height * request.width / 256, 64], key: MLXRandom.key(seed)).asType(.bfloat16)
        for step in 0..<request.steps {
            try Task.checkCancellation()
            let input = reference.map { concatenated([$0, x], axis: 1) } ?? x
            let velocity = try transformer(latents: input, conditioning: conditioning.hidden, sigma: sigmas[step], layout: layout)
            // Euler accumulation in fp32, then restore model precision.
            x = (x.asType(.float32) + velocity.asType(.float32) * (sigmas[step + 1] - sigmas[step])).asType(.bfloat16)
            eval(x)
            Memory.clearCache()
            progress(step + 1)
        }
        return x
    }

    private static func decode(root: URL, latent: MLXArray, height: Int, width: Int) throws -> MLXArray {
        try Qwen21VAE(directory: root.appendingPathComponent("vae")).decode(latent, height: height, width: width)
    }

    private static func referenceImage(path: String, targetArea: Int) throws -> CIImage {
        guard let source = CIImage(contentsOf: URL(fileURLWithPath: path)),
              source.extent.width > 0, source.extent.height > 0 else {
            throw Qwen21Error.invalid("無法讀取參考圖片。")
        }
        let scale = sqrt(Double(targetArea) / (source.extent.width * source.extent.height))
        let w = max(32, min(2048, Int((source.extent.width * scale / 32).rounded()) * 32))
        let h = max(32, min(2048, Int((source.extent.height * scale / 32).rounded()) * 32))
        return source.transformed(by: CGAffineTransform(scaleX: Double(w) / source.extent.width,
                                                        y: Double(h) / source.extent.height))
            .cropped(to: CGRect(x: 0, y: 0, width: w, height: h))
    }

    static func writePNG(_ rgba: MLXArray, to url: URL, width: Int, height: Int) throws {
        guard all(isFinite(rgba)).item(Bool.self) else {
            throw Qwen21Error.invalid("解碼結果包含非有限數值，未寫入圖片。")
        }
        let bytes = round(clip(rgba, min: 0, max: 1) * 255).asType(.uint8).asArray(UInt8.self)
        guard bytes.count == width * height * 4,
              let provider = CGDataProvider(data: Data(bytes) as CFData),
              let color = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: color, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw Qwen21Error.invalid("RGBA 輸出格式錯誤。")
        }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else {
            throw Qwen21Error.invalid("無法建立 PNG。")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw Qwen21Error.invalid("PNG 編碼失敗。") }
        try (data as Data).write(to: url, options: .atomic)
    }
}
