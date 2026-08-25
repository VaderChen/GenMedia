import CoreGraphics
import Darwin
import Foundation
import ImageIO
import MLX
import QwenImageEdit
import UniformTypeIdentifiers

private struct WorkerRequest: Decodable {
    enum Quantization: String, Decodable {
        case int4
        case int8
        case fp16
    }

    var modelDirectory: String
    var quantization: Quantization
    var inputPath: String
    var outputPath: String
    var prompt: String
    var negativePrompt: String
    var width: Int
    var height: Int
    var steps: Int
    var seed: UInt64
}

private struct OutputPlan {
    var generationWidth: Int
    var generationHeight: Int
    var outputWidth: Int
    var outputHeight: Int
}

private struct WorkerEvent: Encodable {
    var type: String
    var stage: String?
    var value: Double?
    var message: String?
    var width: Int?
    var height: Int?
}

private enum WorkerError: LocalizedError {
    case usage
    case missingFile(URL)
    case invalidImage(URL)
    case pngEncoding

    var errorDescription: String? {
        switch self {
        case .usage:
            "用法：GenImageQwen2511Worker --request <request.json>"
        case let .missingFile(url):
            "找不到必要檔案：\(url.path)"
        case let .invalidImage(url):
            "無法讀取輸入圖片：\(url.path)"
        case .pngEncoding:
            "無法編碼輸出 PNG。"
        }
    }
}

@main
private enum GenImageQwen2511Worker {
    /// diffusers' Qwen-Image-Edit canvas area — the size both the reference pipeline and
    /// the training recipe use for the target and the conditioning latents alike.
    private static let nativeGenerationArea = 1024 * 1024

    static func main() async {
        do {
            let requestURL = try requestURL(from: CommandLine.arguments)
            let data = try Data(contentsOf: requestURL)
            let request = try JSONDecoder().decode(WorkerRequest.self, from: data)
            try await run(request)
        } catch {
            emit(
                WorkerEvent(
                    type: "error",
                    message: error.localizedDescription
                ),
                to: .standardError
            )
            Darwin.exit(EXIT_FAILURE)
        }
    }

    private static func run(_ request: WorkerRequest) async throws {
        let modelDirectory = URL(fileURLWithPath: request.modelDirectory, isDirectory: true)
        let snapshot = modelDirectory.appendingPathComponent("snapshot", isDirectory: true)
        let quantized = modelDirectory.appendingPathComponent("quantized", isDirectory: true)
        let inputURL = URL(fileURLWithPath: request.inputPath)
        let outputURL = URL(fileURLWithPath: request.outputPath)

        try require(snapshot.appendingPathComponent("vae/config.json"))
        try require(snapshot.appendingPathComponent("text_encoder/config.json"))
        try require(snapshot.appendingPathComponent("processor/tokenizer.json"))
        try FileManager.default.createDirectory(at: quantized, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let quantizedDiT: URL?
        let quantizedEncoder: URL?
        switch request.quantization {
        case .int4:
            quantizedDiT = quantized.appendingPathComponent(
                "qie-2511-dit-int4-mod8.safetensors"
            )
            quantizedEncoder = quantized.appendingPathComponent(
                "qie-2511-vl7b-int4.safetensors"
            )
            try require(quantizedDiT!)
            try require(quantizedEncoder!)

        case .int8:
            try require(snapshot.appendingPathComponent("transformer/config.json"))
            let dit = quantized.appendingPathComponent(
                "qie-2511-dit-int8-mod8.safetensors"
            )
            let encoder = quantized.appendingPathComponent(
                "qie-2511-vl7b-int8.safetensors"
            )
            if !FileManager.default.fileExists(atPath: dit.path) {
                emitProgress(stage: "convertingDiT", value: 0.03)
                try QwenImageEditWeights.saveQuantizedDiT(
                    from: snapshot.appendingPathComponent("transformer", isDirectory: true),
                    to: dit,
                    config: .init(ditBits: 8, modulationBits: 8, groupSize: 64)
                )
                Memory.clearCache()
            }
            if !FileManager.default.fileExists(atPath: encoder.path) {
                emitProgress(stage: "convertingEncoder", value: 0.08)
                try QwenVLPromptEncoder.saveQuantizedTextModel(
                    snapshot: snapshot,
                    to: encoder,
                    bits: 8,
                    groupSize: 64
                )
                Memory.clearCache()
            }
            quantizedDiT = dit
            quantizedEncoder = encoder

        case .fp16:
            try require(snapshot.appendingPathComponent("transformer/config.json"))
            quantizedDiT = nil
            quantizedEncoder = nil
        }

        emitProgress(stage: "loadingModel", value: 0.10)
        let transformer: QwenImageTransformer2DModel
        if let quantizedDiT {
            transformer = try QwenImageEditWeights.loadQuantizedDiT(from: quantizedDiT)
        } else {
            transformer = try QwenImageEditWeights.loadDiTFromPT(
                directory: snapshot.appendingPathComponent("transformer", isDirectory: true),
                dtype: .bfloat16
            )
        }
        let vae = try QwenImageEditWeights.loadVAE(
            directory: snapshot.appendingPathComponent("vae", isDirectory: true),
            dtype: request.quantization == .fp16 ? .float32 : .bfloat16
        )
        let quantizedEncoderPath = quantizedEncoder?.path
        let generator = QwenImageEditGenerator(
            encoderProvider: {
                try await QwenVLPromptEncoder.load(
                    snapshot: snapshot,
                    quantizedTextModelPath: quantizedEncoderPath
                )
            },
            transformer: transformer,
            vae: vae
        )

        let input = try decodeRGB(inputURL)
        let plan = outputPlan(width: request.width, height: request.height)
        let fittedInput = padSourceToOutputAspect(
            input,
            targetWidth: plan.generationWidth,
            targetHeight: plan.generationHeight
        )
        emitProgress(stage: "generating", value: 0.15)
        let output = try await generator.generate(
            image: fittedInput,
            prompt: request.prompt,
            negativePrompt: request.negativePrompt.isEmpty ? " " : request.negativePrompt,
            width: plan.generationWidth,
            height: plan.generationHeight,
            steps: request.steps,
            trueCFGScale: 4,
            seed: request.seed,
            progress: { current, total in
                let denominator = max(1, total)
                let fraction = 0.15 + 0.80 * Double(current) / Double(denominator)
                emitProgress(stage: "denoising", value: fraction)
            }
        )
        let pixels: [UInt8]
        if output.width == plan.outputWidth, output.height == plan.outputHeight {
            pixels = output.pixels
        } else {
            emitProgress(stage: "resizing", value: 0.97)
            pixels = PILLanczosResize.resize(
                rgb: output.pixels,
                width: output.width,
                height: output.height,
                outWidth: plan.outputWidth,
                outHeight: plan.outputHeight
            )
        }
        try encodePNG(
            pixels: pixels,
            width: plan.outputWidth,
            height: plan.outputHeight
        ).write(to: outputURL, options: .atomic)
        emit(
            WorkerEvent(
                type: "completed",
                value: 1,
                width: plan.outputWidth,
                height: plan.outputHeight
            )
        )
    }

    /// The DiT denoises a latent grid of `width/16 × height/16` tokens and was trained at a
    /// ~1024² canvas (≈4096 tokens). The conditioning grid has to match the target grid or
    /// the centred RoPE positions only overlap over the middle of the source, so a small
    /// output would force a small conditioning grid too — and far below the trained token
    /// count the denoise degrades and then collapses into striping. Generate at
    /// the model's own canvas area in the requested aspect and resample the decode down to
    /// the requested size instead; requests at or above that area are generated as asked.
    private static func outputPlan(width: Int, height: Int) -> OutputPlan {
        let outputWidth = max(1, width)
        let outputHeight = max(1, height)
        // The Runtime floors the generation canvas to a multiple of 16.
        let quantizedWidth = max(outputWidth / 16, 1) * 16
        let quantizedHeight = max(outputHeight / 16, 1) * 16
        guard quantizedWidth * quantizedHeight < nativeGenerationArea else {
            return OutputPlan(
                generationWidth: quantizedWidth,
                generationHeight: quantizedHeight,
                outputWidth: outputWidth,
                outputHeight: outputHeight
            )
        }
        let (generationWidth, generationHeight) = QwenVLPromptEncoder.calculateDimensions(
            targetArea: nativeGenerationArea,
            ratio: Double(outputWidth) / Double(outputHeight)
        )
        return OutputPlan(
            generationWidth: max(generationWidth, 16),
            generationHeight: max(generationHeight, 16),
            outputWidth: outputWidth,
            outputHeight: outputHeight
        )
    }

    private static func requestURL(from arguments: [String]) throws -> URL {
        guard arguments.count == 3, arguments[1] == "--request" else {
            throw WorkerError.usage
        }
        return URL(fileURLWithPath: arguments[2])
    }

    private static func require(_ url: URL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WorkerError.missingFile(url)
        }
    }

    private static func decodeRGB(_ url: URL) throws -> (rgb: [UInt8], width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw WorkerError.invalidImage(url)
        }
        let width = image.width
        let height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &rgba,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw WorkerError.invalidImage(url) }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for index in 0..<(width * height) {
            rgb[index * 3] = rgba[index * 4]
            rgb[index * 3 + 1] = rgba[index * 4 + 1]
            rgb[index * 3 + 2] = rgba[index * 4 + 2]
        }
        return (rgb, width, height)
    }

    /// Extend the source to the output aspect ratio so nothing is cropped, keeping the
    /// source's own scale (capped) rather than pre-shrinking to the output size — the
    /// pipeline resizes this canvas to the target once, and the VL branch still sees a
    /// detailed image even when the requested output is small.
    private static func padSourceToOutputAspect(
        _ image: (rgb: [UInt8], width: Int, height: Int),
        targetWidth: Int,
        targetHeight: Int
    ) -> (rgb: [UInt8], width: Int, height: Int) {
        guard image.width > 0, image.height > 0, targetWidth > 0, targetHeight > 0 else {
            return image
        }

        // The Runtime floors the output to a multiple of 16; mirror that here so the
        // canvas matches the grid the pipeline actually generates.
        let outputWidth = max(targetWidth / 16, 1) * 16
        let outputHeight = max(targetHeight / 16, 1) * 16
        let sourceRatio = Double(image.width) / Double(image.height)
        let targetRatio = Double(outputWidth) / Double(outputHeight)
        var canvasWidth: Int
        var canvasHeight: Int
        if sourceRatio >= targetRatio {
            canvasWidth = image.width
            canvasHeight = max(1, Int((Double(image.width) / targetRatio).rounded()))
        } else {
            canvasHeight = image.height
            canvasWidth = max(1, Int((Double(image.height) * targetRatio).rounded()))
        }

        // An extreme aspect change on a very large source would otherwise balloon the
        // intermediate buffer; anything above the output size (or the VL branch's needs)
        // is discarded by the pipeline's own resize anyway.
        let canvasLimit = max(outputWidth, outputHeight, 1024)
        let longestSide = max(canvasWidth, canvasHeight)
        if longestSide > canvasLimit {
            let scale = Double(canvasLimit) / Double(longestSide)
            canvasWidth = max(1, Int((Double(canvasWidth) * scale).rounded()))
            canvasHeight = max(1, Int((Double(canvasHeight) * scale).rounded()))
        }

        let fittedWidth: Int
        let fittedHeight: Int
        if sourceRatio >= targetRatio {
            fittedWidth = canvasWidth
            fittedHeight = max(1, Int((Double(canvasWidth) / sourceRatio).rounded()))
        } else {
            fittedHeight = canvasHeight
            fittedWidth = max(1, Int((Double(canvasHeight) * sourceRatio).rounded()))
        }

        guard fittedWidth != image.width || fittedHeight != image.height
            || canvasWidth != image.width || canvasHeight != image.height
        else { return image }

        let resized = fittedWidth == image.width && fittedHeight == image.height
            ? image.rgb
            : PILLanczosResize.resize(
                rgb: image.rgb,
                width: image.width,
                height: image.height,
                outWidth: fittedWidth,
                outHeight: fittedHeight
            )
        guard fittedWidth != canvasWidth || fittedHeight != canvasHeight else {
            return (resized, fittedWidth, fittedHeight)
        }

        var canvas = [UInt8](repeating: 0, count: canvasWidth * canvasHeight * 3)
        let offsetX = (canvasWidth - fittedWidth) / 2
        let offsetY = (canvasHeight - fittedHeight) / 2
        for y in 0..<canvasHeight {
            let sourceY = min(max(y - offsetY, 0), fittedHeight - 1)
            for x in 0..<canvasWidth {
                let sourceX = min(max(x - offsetX, 0), fittedWidth - 1)
                let sourceIndex = (sourceY * fittedWidth + sourceX) * 3
                let targetIndex = (y * canvasWidth + x) * 3
                canvas[targetIndex] = resized[sourceIndex]
                canvas[targetIndex + 1] = resized[sourceIndex + 1]
                canvas[targetIndex + 2] = resized[sourceIndex + 2]
            }
        }
        return (canvas, canvasWidth, canvasHeight)
    }

    private static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        guard pixels.count == width * height * 3,
              let context = CGContext(
                data: nil,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ) else { throw WorkerError.pngEncoding }
        let buffer = context.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for index in 0..<(width * height) {
            buffer[index * 4] = pixels[index * 3]
            buffer[index * 4 + 1] = pixels[index * 3 + 1]
            buffer[index * 4 + 2] = pixels[index * 3 + 2]
            buffer[index * 4 + 3] = 255
        }
        guard let image = context.makeImage() else { throw WorkerError.pngEncoding }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else { throw WorkerError.pngEncoding }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw WorkerError.pngEncoding }
        return data as Data
    }

    private static func emitProgress(stage: String, value: Double) {
        emit(WorkerEvent(type: "progress", stage: stage, value: min(1, max(0, value))))
    }

    private static func emit(_ event: WorkerEvent, to fileHandle: FileHandle = .standardOutput) {
        guard let data = try? JSONEncoder().encode(event) else { return }
        fileHandle.write(data)
        fileHandle.write(Data([0x0A]))
        try? fileHandle.synchronize()
    }
}
