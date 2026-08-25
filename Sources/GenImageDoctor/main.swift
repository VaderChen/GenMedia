import Foundation
import GenImageCore
import GenImageRuntime

@main
struct GenImageDoctor {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        if arguments.first == "upscale-test" {
            await runUpscaleTest(arguments: Array(arguments.dropFirst()))
            return
        }
        if arguments.first == "generate-test" {
            await runGenerateTest(arguments: Array(arguments.dropFirst()))
            return
        }
        if arguments.first == "describe-test" {
            await runDescribeTest(arguments: Array(arguments.dropFirst()))
            return
        }

        let jsonOutput = arguments.contains("--json")
        let explicitPaths = arguments.filter { !$0.hasPrefix("-") }
        let rootURL = ModelStorage.rootURL(explicitPath: explicitPaths.first)

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: rootURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            FileHandle.standardError.write(Data("找不到模型目錄：\(rootURL.path)\n".utf8))
            Foundation.exit(2)
        }

        let catalog = LocalModelDiscovery.discover(at: rootURL)
        guard !catalog.models.isEmpty else {
            FileHandle.standardError.write(Data("模型目錄存在，但沒有找到支援的模型。\n".utf8))
            Foundation.exit(3)
        }

        if jsonOutput {
            outputJSON(rootURL: rootURL, catalog: catalog)
        } else {
            outputText(rootURL: rootURL, catalog: catalog)
        }
    }

    private static func outputText(rootURL: URL, catalog: DiscoveredModelCatalog) {
        print("GenImage Model Doctor")
        print("root: \(rootURL.path)")
        print("models: \(catalog.models.count)")

        for model in catalog.models {
            let capabilities = model.capabilities.map(\.title).sorted().joined(separator: ", ")
            print("  ✓ \(model.displayName)")
            print("    capabilities: \(capabilities)")
            print("    format: \(model.quantization.rawValue)")
            print("    size: \(String(format: "%.2f", model.approximateDownloadGB)) GB")
            print("    path: \(model.localURL?.path ?? "-")")
        }

        print("loras: \(catalog.loras.count)")
        for lora in catalog.loras {
            print("  ✓ \(lora.displayName)")
            print("    size: \(String(format: "%.1f", lora.fileSizeMB)) MB")
            print("    path: \(lora.localURL.path)")
        }

        print("profiles: \(catalog.profiles.count)")
        for profile in catalog.profiles {
            print("  ✓ [\(profile.capability.title)] \(profile.name)")
            print("    engine: \(profile.architecture.title)")
            print("    revision: \(profile.modelRevision)")
        }
    }

    private static func outputJSON(rootURL: URL, catalog: DiscoveredModelCatalog) {
        struct Report: Encodable {
            let schemaVersion: Int
            let root: String
            let models: [ModelDescriptor]
            let loras: [LoRADescriptor]
            let profiles: [InferenceProfile]
        }

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            let report = Report(
                schemaVersion: 1,
                root: rootURL.path,
                models: catalog.models,
                loras: catalog.loras,
                profiles: catalog.profiles
            )
            let data = try encoder.encode(report)
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data("\n".utf8))
        } catch {
            FileHandle.standardError.write(Data("無法輸出 JSON：\(error.localizedDescription)\n".utf8))
            Foundation.exit(4)
        }
    }

    private static func printUsage() {
        print("""
        GenImageDoctor [model-root] [--json]
        GenImageDoctor upscale-test <model.mlmodel> <input-image> <output-directory>
        GenImageDoctor generate-test <model-directory> <output-directory> [prompt]
        GenImageDoctor describe-test <model-directory> <input-image> [language-code]

        驗證 GenImage 可辨識的本機模型與自動產生的 Profiles。
        未提供 model-root 時，依序使用 GENIMAGE_MODEL_ROOT 與內建預設路徑。
        """)
    }

    private static func runUpscaleTest(arguments: [String]) async {
        guard arguments.count == 3 else {
            FileHandle.standardError.write(Data("upscale-test 需要 model、input 與 output-directory。\n".utf8))
            Foundation.exit(64)
        }

        let modelURL = URL(fileURLWithPath: arguments[0])
        let inputURL = URL(fileURLWithPath: arguments[1])
        let outputDirectory = URL(fileURLWithPath: arguments[2], isDirectory: true)
        let projectID = UUID()
        let asset = MediaAsset(
            projectID: projectID,
            kind: .imported,
            title: inputURL.deletingPathExtension().lastPathComponent,
            fileURL: inputURL,
            pixelWidth: 0,
            pixelHeight: 0
        )
        let profile = InferenceProfile(
            name: "Doctor Upscale",
            capability: .upscale,
            modelID: modelURL.path,
            modelRevision: "doctor",
            architecture: .coreML,
            defaults: ProfileDefaults(upscaleScale: 4, tileSize: 512)
        )
        let service = CoreMLUpscaleService(outputDirectory: outputDirectory)

        do {
            let output = try await service.upscale(
                request: UpscaleRequest(asset: asset, profile: profile, scale: 4),
                progress: { value in
                    FileHandle.standardError.write(Data("progress \(Int(value * 100))%\n".utf8))
                }
            )
            print(output.fileURL?.path ?? "")
        } catch {
            FileHandle.standardError.write(Data("Upscale 失敗：\(error.localizedDescription)\n".utf8))
            Foundation.exit(5)
        }
    }

    private static func runGenerateTest(arguments: [String]) async {
        guard arguments.count >= 2 else {
            FileHandle.standardError.write(
                Data("generate-test 需要 model-directory 與 output-directory。\n".utf8)
            )
            Foundation.exit(64)
        }

        let modelURL = URL(fileURLWithPath: arguments[0], isDirectory: true)
        let outputDirectory = URL(fileURLWithPath: arguments[1], isDirectory: true)
        let prompt = arguments.dropFirst(2).joined(separator: " ").trimmingCharacters(in: .whitespaces)
        let projectID = UUID()
        let profile = InferenceProfile(
            name: "Doctor Z-Image",
            capability: .textToImage,
            modelID: modelURL.path,
            modelRevision: "doctor",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(width: 256, height: 256, steps: 1, outputCount: 1)
        )
        let recipe = GenerationRecipe(
            name: "Doctor Test",
            prompt: prompt.isEmpty ? "A small red fox in a quiet forest, natural light" : prompt,
            modelID: modelURL.path,
            profileID: profile.id,
            width: 256,
            height: 256,
            steps: 1,
            outputCount: 1,
            seed: 1
        )
        let service = ZImageTextToImageService(outputDirectory: outputDirectory)

        do {
            let outputs = try await service.generate(
                request: TextToImageRequest(
                    projectID: projectID,
                    recipe: recipe,
                    profile: profile
                ),
                progress: { value in
                    FileHandle.standardError.write(Data("progress \(Int(value * 100))%\n".utf8))
                }
            )
            print(outputs.first?.fileURL?.path ?? "")
        } catch {
            FileHandle.standardError.write(Data("文生圖失敗：\(error.localizedDescription)\n".utf8))
            Foundation.exit(6)
        }
    }

    private static func runDescribeTest(arguments: [String]) async {
        guard arguments.count >= 2 else {
            FileHandle.standardError.write(
                Data("describe-test 需要 model-directory 與 input-image。\n".utf8)
            )
            Foundation.exit(64)
        }

        let modelURL = URL(fileURLWithPath: arguments[0], isDirectory: true)
        let inputURL = URL(fileURLWithPath: arguments[1])
        let languageCode = arguments.count > 2 ? arguments[2] : "zh-Hant"
        let asset = MediaAsset(
            projectID: UUID(),
            kind: .imported,
            title: inputURL.deletingPathExtension().lastPathComponent,
            fileURL: inputURL,
            pixelWidth: 0,
            pixelHeight: 0
        )
        let profile = InferenceProfile(
            name: "Doctor Qwen3-VL",
            capability: .imageToText,
            modelID: modelURL.path,
            modelRevision: "doctor",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 96, languageCode: languageCode)
        )
        let service = QwenVLImageDescriptionService()

        do {
            let description = try await service.describe(
                request: ImageDescriptionRequest(
                    asset: asset,
                    profile: profile,
                    languageCode: languageCode
                ),
                progress: { value in
                    FileHandle.standardError.write(Data("progress \(Int(value * 100))%\n".utf8))
                }
            )
            print(description)
        } catch {
            FileHandle.standardError.write(Data("圖生文失敗：\(error.localizedDescription)\n".utf8))
            Foundation.exit(7)
        }
    }
}
