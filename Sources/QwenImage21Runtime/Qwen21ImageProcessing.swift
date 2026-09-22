import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// Normalize sRGB pixel values in tensor space. Applying a CIColorMatrix before rendering
/// would normalize Core Image's linear-light values instead of the checkpoint's sRGB values.
enum Qwen21ImageProcessing {
    static func rgba(_ image: CIImage) -> MLXArray {
        let w = Int(image.extent.width), h = Int(image.extent.height)
        var values = [Float](repeating: 0, count: w * h * 4)
        values.withUnsafeMutableBytes { buffer in
            // Let the renderer produce straight sRGB RGBA. Unpremultiplying a CIImage
            // beforehand happens in linear light, before the output color conversion.
            CIContext(options: [.cacheIntermediates: false, .outputPremultiplied: false]).render(
                image, toBitmap: buffer.baseAddress!, rowBytes: w * 16,
                bounds: image.extent, format: .RGBAf, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        }
        return MLXArray(values).reshaped([1, h, w, 4])
    }

    static func visionPixels(_ image: CIImage, configuration c: Qwen3VLProcessorConfiguration) throws -> (MLXArray, THW) {
        let w = Int(image.extent.width), h = Int(image.extent.height)
        let p = c.patchSize, m = c.mergeSize, t = c.temporalPatchSize
        guard p > 0, m > 0, t > 0, w % (p * m) == 0, h % (p * m) == 0,
              c.imageMean.count == 3, c.imageStd.count == 3, c.imageStd.allSatisfy({ $0 > 0 }) else {
            throw Qwen21Error.invalid("視覺前處理設定或參考圖片尺寸不符。")
        }
        let pixels = rgba(image)
        let alpha = pixels[0..., 0..., 0..., 3..<4]
        // Match the checkpoint's Pillow paste over white, which blends sRGB values.
        let rgb = pixels[0..., 0..., 0..., ..<3] * alpha + (1 - alpha)
        let mean = MLXArray(c.imageMean.map(Float.init)), deviation = MLXArray(c.imageStd.map(Float.init))
        let normalized = ((rgb - mean) / deviation).transposed(0, 3, 1, 2)
        let gridH = h / p, gridW = w / p
        // Duplicate a still frame for the temporal patch, then group spatial patches in 2x2 blocks.
        let patches = repeated(normalized, count: t, axis: 0)
            .reshaped([1, t, 3, gridH / m, m, p, gridW / m, m, p])
            .transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .reshaped([gridH * gridW, 3 * t * p * p])
        return (patches, THW(1, gridH, gridW))
    }
}
