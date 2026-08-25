import Foundation

/// The single source of truth for output-size arithmetic.
///
/// Three different sizes travel through one generation and they are not the same number:
///
///   - the size the user asked for — arbitrary, whatever the UI or an MCP caller sends;
///   - the canvas the Runtime can actually denoise — a multiple of `alignment`, because the
///     DiT carries one latent token per 16 px;
///   - the canvas the model was trained on — `nativeCanvasArea`, which is what the denoise
///     stays coherent at.
///
/// Those three drifting apart is what produced the image-to-image cropping and the striped
/// collapse, so every rule that converts between them lives here. The WebUI mirrors this file
/// in `Resources/WebUI/js/geometry.js`; keep the two in step.
public enum OutputGeometry: Sendable {
    /// One latent token per 16 px — every canvas has to land on a multiple of this.
    public static let alignment = 16
    public static let minimumDimension = 64
    public static let maximumDimension = 4096
    public static let supportedRange = minimumDimension...maximumDimension
    /// diffusers' Qwen-Image-Edit canvas area, used for the target and the conditioning
    /// latents alike (≈4096 latent tokens).
    public static let nativeCanvasArea = 1024 * 1024
    /// diffusers `calculate_dimensions` rounds to a multiple of 32, not 16.
    private static let canvasAlignment = 32

    public static func isSupported(_ value: Int) -> Bool {
        supportedRange.contains(value) && value.isMultiple(of: alignment)
    }

    public static func isSupported(width: Int, height: Int) -> Bool {
        isSupported(width) && isSupported(height)
    }

    /// The user-facing rule: nearest multiple of `alignment`, clamped into the supported
    /// range. Nearest rather than floor so a request lands on whichever side is closer.
    public static func quantize(_ value: Int) -> Int {
        guard value > 0 else { return minimumDimension }
        let rounded = (value + alignment / 2) / alignment * alignment
        return min(max(rounded, minimumDimension), maximumDimension)
    }

    public static func quantize(width: Int, height: Int) -> (width: Int, height: Int) {
        (quantize(width), quantize(height))
    }

    /// The Runtime's own rule: it floors the generation canvas to a multiple of `alignment`
    /// and has no upper bound of its own. A canvas handed to the Runtime must already satisfy
    /// this, or the size it generates is not the size that was planned.
    public static func runtimeAligned(_ value: Int) -> Int {
        max(value / alignment, 1) * alignment
    }

    /// diffusers `calculate_dimensions`: the `area`-sized canvas at `ratio`, each side rounded
    /// to a multiple of 32 (Python banker's rounding).
    public static func canvas(area: Int, ratio: Double) -> (width: Int, height: Int) {
        guard area > 0, ratio.isFinite, ratio > 0 else {
            return (canvasAlignment, canvasAlignment)
        }
        let width = (Double(area) * ratio).squareRoot()
        let height = width / ratio
        let quantized = { (value: Double) -> Int in
            max(Int((value / Double(canvasAlignment)).rounded(.toNearestOrEven)), 1)
                * canvasAlignment
        }
        return (quantized(width), quantized(height))
    }

    /// How one image-to-image request is split into a canvas to denoise and a file to write.
    public struct ImageEditPlan: Equatable, Sendable {
        public let generationWidth: Int
        public let generationHeight: Int
        public let outputWidth: Int
        public let outputHeight: Int

        public init(
            generationWidth: Int,
            generationHeight: Int,
            outputWidth: Int,
            outputHeight: Int
        ) {
            self.generationWidth = generationWidth
            self.generationHeight = generationHeight
            self.outputWidth = outputWidth
            self.outputHeight = outputHeight
        }

        /// True when the decode has to be resampled before it is written.
        public var needsResample: Bool {
            generationWidth != outputWidth || generationHeight != outputHeight
        }
    }

    /// Below `nativeCanvasArea` the conditioning grid would have to shrink with the target
    /// grid — the two must match or the centred RoPE positions only overlap over the middle
    /// of the source — and far below the trained token count the denoise degrades and then
    /// collapses into striping. So a small request is generated at the model's own area in
    /// the requested aspect and resampled down; a request at or above that area is generated
    /// as asked.
    public static func imageEditPlan(width: Int, height: Int) -> ImageEditPlan {
        let outputWidth = max(1, width)
        let outputHeight = max(1, height)
        let alignedWidth = runtimeAligned(outputWidth)
        let alignedHeight = runtimeAligned(outputHeight)
        guard alignedWidth * alignedHeight < nativeCanvasArea else {
            return ImageEditPlan(
                generationWidth: alignedWidth,
                generationHeight: alignedHeight,
                outputWidth: outputWidth,
                outputHeight: outputHeight
            )
        }
        let generation = canvas(
            area: nativeCanvasArea,
            ratio: Double(outputWidth) / Double(outputHeight)
        )
        return ImageEditPlan(
            generationWidth: generation.width,
            generationHeight: generation.height,
            outputWidth: outputWidth,
            outputHeight: outputHeight
        )
    }
}
