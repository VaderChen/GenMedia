import Foundation

public enum GGUFDiagnosticModel: String, CaseIterable, Codable, Sendable {
    case ltxVideo096
    case ltx23Distilled
    case miniMaxH3
}

public struct GGUFDiagnosticSpec: Codable, Sendable {
    public let model: GGUFDiagnosticModel
    public let modelID: String
    public let displayName: String
    public let directoryName: String
    public let weightRelativePath: String
    public let expectedSourceType: String
    public let requiredCompanionPaths: [String]
    public let runtimeBlocker: String

    public init(
        model: GGUFDiagnosticModel,
        modelID: String,
        displayName: String,
        directoryName: String,
        weightRelativePath: String,
        expectedSourceType: String,
        requiredCompanionPaths: [String],
        runtimeBlocker: String
    ) {
        self.model = model
        self.modelID = modelID
        self.displayName = displayName
        self.directoryName = directoryName
        self.weightRelativePath = weightRelativePath
        self.expectedSourceType = expectedSourceType
        self.requiredCompanionPaths = requiredCompanionPaths
        self.runtimeBlocker = runtimeBlocker
    }
}

public struct GGUFDiagnosticReport: Sendable {
    public let spec: GGUFDiagnosticSpec
    public let modelDirectory: URL
    public let weightURL: URL
    public let fileSize: UInt64?
    public let inspection: GGUFInspection?
    public let missingCompanionPaths: [String]
    public let errorMessage: String?

    public var structurePassed: Bool {
        guard let inspection,
              errorMessage == nil,
              inspection.unsupportedTypes.isEmpty,
              inspection.quantizationCounts[spec.expectedSourceType, default: 0] > 0 else {
            return false
        }
        return true
    }

    public var generationReady: Bool {
        structurePassed && missingCompanionPaths.isEmpty
    }
}

public enum GGUFDiagnosticPlan {
    public static let all: [GGUFDiagnosticSpec] = [
        GGUFDiagnosticSpec(
            model: .ltxVideo096,
            modelID: "city96/LTX-Video-0.9.6-distilled-gguf@Q4_K_M",
            displayName: "LTX-Video 0.9.6 GGUF Q4_K_M",
            directoryName: "ltx-video-0.9.6-distilled-gguf",
            weightRelativePath: "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf",
            expectedSourceType: "Q4_K",
            requiredCompanionPaths: [
                "LTX-Video-0.9.6-VAE-BF16.safetensors",
                "text_encoder/config.json",
                "text_encoder/t5-v1_1-xxl-encoder-Q4_K_M.gguf",
                "tokenizer/spiece.model"
            ],
            runtimeBlocker: "T5 encoder、tokenizer 與 VAE 是必要配套；LTX-Video 0.9.6 專用生成 Runtime 尚未接入。"
        ),
        GGUFDiagnosticSpec(
            model: .ltx23Distilled,
            modelID: "unsloth/LTX-2.3-GGUF@distilled-1.1-Q3_K_M",
            displayName: "LTX-2.3 22B Distilled 1.1 GGUF Q3_K_M",
            directoryName: "ltx-2.3-distilled-1.1-gguf-q3-k-m",
            weightRelativePath: "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf",
            expectedSourceType: "Q3_K",
            requiredCompanionPaths: [
                "vae/ltx-2.3-22b-distilled_video_vae.safetensors",
                "vae/ltx-2.3-22b-distilled_audio_vae.safetensors",
                "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors",
                "text_encoders/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
                "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
            ],
            runtimeBlocker: "LTX-2.3 的 VAE、connector、Gemma 與 spatial upscaler 是必要配套；現有 Swift Worker 尚未支援這份 GGUF 佈局。"
        ),
        GGUFDiagnosticSpec(
            model: .miniMaxH3,
            modelID: "unsloth/MiniMax-H3-GGUF@fl2va-pruned-Q4_K",
            displayName: "MiniMax H3 GGUF FL2VA Pruned Q4_K",
            directoryName: "minimax-h3-gguf-fl2va-pruned-q4-k",
            weightRelativePath: "minimax_h3_fl2va_pruned-Q4_K.gguf",
            expectedSourceType: "Q4_K",
            requiredCompanionPaths: [
                "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                "vae/minimax_h3_video_vae_fp16.safetensors",
                "vae/minimax_h3_audio_vae_fp32.safetensors"
            ],
            runtimeBlocker: "Qwen3-VL GGUF 與 Video/Audio VAE 是必要配套；H3 架構尚未接入 Swift Worker。"
        )
    ]

    public static func spec(for model: GGUFDiagnosticModel) -> GGUFDiagnosticSpec {
        all.first { $0.model == model }!
    }
}

public enum GGUFDiagnosticRunner {
    public static func run(
        modelRoot: URL,
        models: [GGUFDiagnosticModel] = GGUFDiagnosticModel.allCases,
        fileManager: FileManager = .default
    ) -> [GGUFDiagnosticReport] {
        models.map { model in
            let spec = GGUFDiagnosticPlan.spec(for: model)
            let directory = modelRoot.appendingPathComponent(spec.directoryName, isDirectory: true)
            let weightURL = directory.appendingPathComponent(spec.weightRelativePath)
            let missing = spec.requiredCompanionPaths.filter {
                !fileManager.fileExists(atPath: directory.appendingPathComponent($0).path)
            }

            guard fileManager.fileExists(atPath: weightURL.path) else {
                return GGUFDiagnosticReport(
                    spec: spec,
                    modelDirectory: directory,
                    weightURL: weightURL,
                    fileSize: nil,
                    inspection: nil,
                    missingCompanionPaths: missing,
                    errorMessage: "找不到主 GGUF：\(weightURL.path)"
                )
            }

            let fileSize = try? weightURL.resourceValues(forKeys: [.fileSizeKey]).fileSize.map(UInt64.init)
            do {
                let inspection = try GGUFModelLoader.inspect(fileURL: weightURL)
                return GGUFDiagnosticReport(
                    spec: spec,
                    modelDirectory: directory,
                    weightURL: weightURL,
                    fileSize: fileSize ?? nil,
                    inspection: inspection,
                    missingCompanionPaths: missing,
                    errorMessage: nil
                )
            } catch {
                return GGUFDiagnosticReport(
                    spec: spec,
                    modelDirectory: directory,
                    weightURL: weightURL,
                    fileSize: fileSize ?? nil,
                    inspection: nil,
                    missingCompanionPaths: missing,
                    errorMessage: error.localizedDescription
                )
            }
        }
    }
}
