import Foundation
import MLX
import MiniMaxH3SwiftRuntime

/// The `--request/--format/--variant` protocol the app uses to drive the worker.
///
/// Matches the shape `GenImageLTXVideoWorker` uses: a JSON request file in, and
/// newline-delimited JSON events on stdout that the service parses for progress
/// and completion metadata.
enum MiniMaxH3RequestProtocol {
    enum Format: String {
        case gguf
    }

    enum Variant: String {
        case fl2va
        /// Multimodal reference conditioning. The runtime does not implement the
        /// reference-block layout or the Qwen3-VL vision tower yet.
        case ref2va
    }

    struct Request: Decodable {
        struct Keyframe: Decodable {
            /// Frame index in the output video; negative counts from the end.
            var frameIndex: Int
            /// Path to the image anchored at that frame.
            var imagePath: String
        }

        var modelDirectory: String
        var outputPath: String
        var prompt: String
        var width: Int
        var height: Int
        var frames: Int
        var frameRate: Int
        var seed: UInt64
        var steps: Int
        var keyframes: [Keyframe]?
        /// Optional overrides; the worker resolves the standard layout otherwise.
        var transformerPath: String?
        var videoVAEPath: String?
        var audioVAEPath: String?
        var textEncoderPath: String?
        var tokenizerDirectory: String?
    }

    struct Event: Encodable {
        let type: String
        let stage: String?
        let value: Double?
        let durationSeconds: Double?
        let sampleRate: Int?
        let numFrames: Int?
        let pixelWidth: Int?
        let pixelHeight: Int?
        let message: String?

        static func progress(stage: String, value: Double) -> Self {
            Self(type: "progress", stage: stage, value: min(1, max(0, value)),
                 durationSeconds: nil, sampleRate: nil, numFrames: nil,
                 pixelWidth: nil, pixelHeight: nil, message: nil)
        }

        static func completed(
            durationSeconds: Double,
            sampleRate: Int,
            numFrames: Int,
            pixelWidth: Int,
            pixelHeight: Int
        ) -> Self {
            Self(type: "completed", stage: nil, value: 1,
                 durationSeconds: durationSeconds, sampleRate: sampleRate,
                 numFrames: numFrames, pixelWidth: pixelWidth,
                 pixelHeight: pixelHeight, message: nil)
        }

        static func error(_ message: String) -> Self {
            Self(type: "error", stage: nil, value: nil, durationSeconds: nil,
                 sampleRate: nil, numFrames: nil, pixelWidth: nil,
                 pixelHeight: nil, message: message)
        }
    }

    enum RequestError: LocalizedError {
        case unsupportedFormat(String)
        case unsupportedVariant(String)
        case missingComponent(String)
        case invalidGeometry(width: Int, height: Int, frames: Int)

        var errorDescription: String? {
            switch self {
            case let .unsupportedFormat(value):
                "H3 worker 不支援 --format \(value)（目前只支援 gguf）。"
            case let .unsupportedVariant(value):
                """
                H3 worker 的 --variant \(value) 不支援參考條件輸入。Ref2VA 的參考\
                區塊 layout 與 Qwen3-VL 視覺塔尚未實作；純文生影片仍可使用。
                """
            case let .missingComponent(name):
                "H3 模型目錄缺少必要元件：\(name)。"
            case let .invalidGeometry(width, height, frames):
                """
                H3 輸出尺寸不合法：\(width)x\(height)、\(frames) 幀。寬高需為 \
                32 的倍數且至少 64，幀數需為 4 的倍數且至少 4。
                """
            }
        }
    }

    // MARK: - Component layout

    /// Where each component sits inside a downloaded H3 GGUF model directory.
    ///
    /// Explicit paths, not directory sniffing: the installer lays the directory
    /// out from `HuggingFaceModelInstaller.miniMaxH3GGUFPlan`, so the layout is
    /// known rather than guessed.
    struct Components {
        var transformer: URL
        var videoVAE: URL
        var audioVAE: URL?
        var textEncoder: URL?
        var tokenizerDirectory: URL?

        static func resolve(request: Request) throws -> Components {
            let root = URL(fileURLWithPath: request.modelDirectory)

            func existing(_ url: URL?) -> URL? {
                guard let url,
                      FileManager.default.fileExists(atPath: url.path) else { return nil }
                return url
            }

            // The transformer file name carries the variant and quantization, so
            // it is discovered rather than hard-coded.
            let unet = root.appendingPathComponent("unet")
            let transformerOverride = request.transformerPath
                .map { URL(fileURLWithPath: $0) }
            let transformer = try existing(transformerOverride)
                ?? existing(
                    (try? FileManager.default.contentsOfDirectory(
                        at: unet, includingPropertiesForKeys: nil
                    ))?
                    .filter { $0.pathExtension.lowercased() == "gguf" }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent }
                    .first
                )
                ?? { throw RequestError.missingComponent("unet/*.gguf") }()

            let videoVAE = try existing(
                request.videoVAEPath.map { URL(fileURLWithPath: $0) }
            ) ?? existing(
                root.appendingPathComponent("vae/minimax_h3_video_vae_fp16.safetensors")
            ) ?? { throw RequestError.missingComponent("vae/minimax_h3_video_vae_fp16.safetensors") }()

            let textEncoder = existing(
                request.textEncoderPath.map { URL(fileURLWithPath: $0) }
            ) ?? existing(
                root.appendingPathComponent(
                    "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
                )
            )
            let tokenizer = existing(
                request.tokenizerDirectory.map { URL(fileURLWithPath: $0) }
            ) ?? existing(root.appendingPathComponent("upstream/FL2VA/processor"))

            return Components(
                transformer: transformer,
                videoVAE: videoVAE,
                audioVAE: existing(
                    request.audioVAEPath.map { URL(fileURLWithPath: $0) }
                ) ?? existing(
                    root.appendingPathComponent(
                        "vae/minimax_h3_audio_vae_fp32.safetensors"
                    )
                ),
                textEncoder: textEncoder,
                tokenizerDirectory: tokenizer
            )
        }
    }

    // MARK: - Geometry

    /// Convert pixel geometry to the latent geometry the transformer works in.
    ///
    /// The video VAE downsamples 16x spatially and 4x temporally, so the pixel
    /// request has to land on those multiples.
    static func latentGeometry(
        width: Int,
        height: Int,
        frames: Int
    ) throws -> (latentWidth: Int, latentHeight: Int, latentFrames: Int) {
        let vae = MiniMaxH3VideoVAEConfiguration.default
        let spatialAlignment = vae.spatialRatio * 2
        guard width >= spatialAlignment * 2, height >= spatialAlignment * 2,
              frames >= vae.temporalRatio,
              width % spatialAlignment == 0,
              height % spatialAlignment == 0,
              frames % vae.temporalRatio == 0 else {
            throw RequestError.invalidGeometry(
                width: width, height: height, frames: frames
            )
        }
        return (
            width / vae.spatialRatio,
            height / vae.spatialRatio,
            frames / vae.temporalRatio
        )
    }

    /// Audio latent frames for a video duration, at 40 latent frames per second.
    static func audioLatentFrames(frames: Int, frameRate: Int) -> Int {
        let seconds = Double(frames) / Double(max(frameRate, 1))
        let audio = MiniMaxH3AudioVAEConfiguration.default
        let perSecond = audio.sampleRate / audio.hopLength
        return max(1, Int((seconds * Double(perSecond)).rounded()))
    }
}
