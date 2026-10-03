import Foundation
import GenImageCore
import MLX
import Testing
@testable import GenImageRuntime

@Suite(.serialized)
struct QwenPromptEnhancementTests {
    @Test func requiresCompleteAnswerAndKeepsOnlyRewrittenPrompt() throws {
        let text = #"<think>Plan the composition.</think>{"rewritten_prompt":"A red teapot.","wh_ratio":"1:1"}"#
        #expect(try QwenPromptEnhancementService.parse(text, kind: .textToImage) == "A red teapot.")
        let edit = #"{"rewritten_prompt":"Make the teapot blue.","wh_ratio":"","ratio_follow":"<image1>"}"#
        #expect(try QwenPromptEnhancementService.parse(edit, kind: .imageToImage) == "Make the teapot blue.")
        let panorama = #"{"rewritten_prompt":"A sign reading \"Final answer:\" in a panorama.","wh_ratio":"2:1"}"#
        #expect(try QwenPromptEnhancementService.parse(panorama, kind: .textToImage) == "A sign reading \"Final answer:\" in a panorama.")
        let missingQuote = #"{"rewritten_prompt":"Make it blue.","wh_ratio":"","ratio_follow":<image1>"}"#
        #expect(try QwenPromptEnhancementService.parse(missingQuote, kind: .imageToImage) == "Make it blue.")
        let literalTag = #"{"rewritten_prompt":"A sign reading </think>.","wh_ratio":"1:1"}"#
        #expect(try QwenPromptEnhancementService.parse(literalTag, kind: .textToImage) == "A sign reading </think>.")
        for invalid in ["<think>unfinished", #"{"rewritten_prompt":"incomplete""#,
                        String(missingQuote.dropLast()),
                        missingQuote.replacingOccurrences(of: "<image1>", with: "<image2>"),
                        #"{"rewritten_prompt":"cat","wh_ratio":"0:1"}"#,
                        #"{"rewritten_prompt":" ","wh_ratio":"1:1"}"#,
                        #"{"rewritten_prompt":"cat","wh_ratio":"","ratio_follow":"<image2>"}"#] {
            #expect(throws: QwenPromptEnhancementService.Failure.self) {
                try QwenPromptEnhancementService.parse(invalid, kind: .imageToImage)
            }
        }
        #expect(throws: QwenPromptEnhancementService.Failure.self) {
            try QwenPromptEnhancementService.parse(edit, kind: .textToImage)
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_PE_REPLAY_OUTPUT"] != nil))
    func parsesCompleteResponseFromNativeImageEnhancer() throws {
        let path = try #require(ProcessInfo.processInfo.environment["GENIMAGE_PE_REPLAY_OUTPUT"])
        let raw = try String(contentsOfFile: path, encoding: .utf8)
        let rewritten = try QwenPromptEnhancementService.parse(raw, kind: .imageToImage)
        #expect(rewritten.contains("藍色"))
        #expect(!rewritten.contains("ratio_follow"))
        #expect(!rewritten.contains("</think>"))
    }

    @Test func dependenciesSurviveCopyAndLegacyProfilesDecodeWithoutEnhancement() throws {
        for profile in QwenImage21PromptEnhancer.profiles {
            let id = try #require(profile.promptEnhancerModelID)
            let kind = try #require(QwenImage21PromptEnhancer.model(for: id))
            #expect(kind.capability == profile.capability)
            #expect(profile.requiredModelIDs.contains(kind.modelID))
            #expect(profile.duplicated().promptEnhancerModelID == kind.modelID)
            let decoded = try JSONDecoder().decode(InferenceProfile.self, from: JSONEncoder().encode(profile))
            #expect(decoded == profile)
            #expect(try Qwen21ImageService.validatedEnhancer(profile: profile, modelURL: URL(fileURLWithPath: "/pe")) == kind)
            #expect(throws: Qwen21ImageService.Failure.self) {
                try Qwen21ImageService.validatedEnhancer(profile: profile, modelURL: nil)
            }
        }
        var json = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(QwenImage21Model.profiles[0])) as? [String: Any])
        json.removeValue(forKey: "promptEnhancerModelID")
        let legacy = try JSONDecoder().decode(InferenceProfile.self, from: JSONSerialization.data(withJSONObject: json))
        #expect(legacy.promptEnhancerModelID == nil)
        #expect(legacy.requiredModelIDs == [QwenImage21Model.id])
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_PE_INSTALL_ROOT"] != nil))
    func downloadsAndRediscoversImageEnhancementModels() async throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_PE_INSTALL_ROOT"]))
        for enhancer in QwenImage21PromptEnhancer.allCases {
            let installed = try await HuggingFaceModelInstaller().install(modelID: enhancer.modelID, rootURL: root, progress: { _ in })
            #expect(installed.lastPathComponent == "4bit")
            #expect(try HuggingFaceModelInstaller.verify(modelID: enhancer.modelID, rootURL: root) == installed)
            #expect(LocalModelDiscovery.discover(at: root).models.contains { $0.id == enhancer.modelID && $0.localURL == installed })
        }
        let entry = QwenImage21Acceleration.entry
        let installed = try await HuggingFaceModelInstaller().install(modelID: entry.id, rootURL: root, progress: { _ in })
        #expect(try HuggingFaceModelInstaller.verify(modelID: entry.id, rootURL: root) == installed)
        #expect(LocalModelDiscovery.discoverLoRAs(at: root).contains {
            $0.localURL == installed.resolvingSymlinksInPath().standardizedFileURL && $0.compatibleCapabilities == [.textToImage, .imageToImage]
        })
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_PE_SMOKE_ROOT"] != nil))
    func nativePromptEnhancerUsesInstalledTextAndImageModels() async throws {
        let root = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_PE_SMOKE_ROOT"]))
        let output = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_PE_SMOKE_OUTPUT"]))
        let image = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["GENIMAGE_PE_SMOKE_IMAGE"]))
        Memory.cacheLimit = 256 * 1024 * 1024
        for kind in QwenImage21PromptEnhancer.allCases {
            if let selected = ProcessInfo.processInfo.environment["GENIMAGE_PE_SMOKE_KIND"], selected != kind.rawValue { continue }
            let prompt = kind == .textToImage ? "奶油色背景上的紅色茶壺，產品照" : "將茶壺改成藍色，保留背景與形狀"
            let result = try await QwenPromptEnhancementService.enhance(prompt: prompt,
                imageURL: kind == .imageToImage ? image : nil,
                modelURL: root.appendingPathComponent(kind.directoryName).appendingPathComponent("4bit"),
                kind: kind, seed: 42,
                responseObserver: { text in
                    try? text.write(to: output.appendingPathComponent("pe-\(kind.rawValue)-raw.txt"), atomically: true, encoding: .utf8)
                }, progress: { _ in })
            try result.write(to: output.appendingPathComponent("pe-\(kind.rawValue).txt"), atomically: true, encoding: .utf8)
            #expect(result.count > 20)
            Memory.clearCache()
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["GENIMAGE_QWEN_PE_E2E_ROOT"] != nil))
    func enhancementHandsOffToNativeTurboWorkerAndKeepsSourceLink() async throws {
        let env = ProcessInfo.processInfo.environment
        func url(_ key: String) throws -> URL { URL(fileURLWithPath: try #require(env[key])) }
        let root = try url("GENIMAGE_QWEN_PE_E2E_ROOT")
        let output = try url("GENIMAGE_QWEN_PE_E2E_OUTPUT")
        let source = MediaAsset(projectID: UUID(), kind: .imported, title: "Red teapot",
            fileURL: try url("GENIMAGE_QWEN_PE_E2E_IMAGE"), pixelWidth: 256, pixelHeight: 256)
        let profile = try #require(QwenImage21PromptEnhancer.profiles.first {
            $0.capability == .imageToImage && !$0.loras.isEmpty
        })
        let entry = QwenImage21Acceleration.entry
        let request = ImageToImageRequest(projectID: source.projectID, sourceAsset: source,
            recipe: .init(prompt: "將茶壺改成藍色，保留背景與形狀", modelID: QwenImage21Model.id,
                width: 256, height: 256, steps: 6, outputCount: 1, seed: 42),
            profile: profile, modelURL: try url("GENIMAGE_QWEN_PE_E2E_MODEL"), quantization: .fourBit,
            profileLoRAs: [.init(adapterID: entry.id,
                localURL: root.appendingPathComponent(entry.directoryName).appendingPathComponent(entry.filename), scale: 1)],
            promptEnhancerURL: root.appendingPathComponent(QwenImage21PromptEnhancer.imageToImage.directoryName).appendingPathComponent("4bit"))
        Memory.cacheLimit = 256 * 1024 * 1024
        let service = Qwen21ImageService(outputDirectory: output, workerExecutable: try url("GENIMAGE_QWEN_PE_E2E_WORKER"))
        let result = try await service.generate(request: request, progress: { _ in })
        #expect(result.parentAssetID == source.id)
        #expect(result.pixelWidth == 256 && result.pixelHeight == 256)
        #expect(FileManager.default.fileExists(atPath: try #require(result.fileURL).path))
        try JSONEncoder().encode(result).write(to: output.appendingPathComponent("result.json"))
        Memory.clearCache()
    }
}
