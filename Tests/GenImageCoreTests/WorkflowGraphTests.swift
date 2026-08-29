import Foundation
import Testing
@testable import GenImageCore

struct WorkflowGraphTests {
    @Test func lineageKeepsPipelineOrder() {
        let projectID = UUID()
        let original = MediaAsset(
            projectID: projectID,
            kind: .imported,
            title: "原圖",
            pixelWidth: 1024,
            pixelHeight: 1024
        )
        let generated = MediaAsset(
            projectID: projectID,
            parentAssetID: original.id,
            kind: .generated,
            title: "生成圖",
            pixelWidth: 1024,
            pixelHeight: 1024
        )
        let upscaled = MediaAsset(
            projectID: projectID,
            parentAssetID: generated.id,
            kind: .upscaled,
            title: "放大圖",
            pixelWidth: 4096,
            pixelHeight: 4096
        )
        let graph = WorkflowGraph(assets: [upscaled, original, generated])

        #expect(graph.lineage(of: upscaled.id).map(\.id) == [original.id, generated.id, upscaled.id])
    }

    // Exact counts rot every time the catalog grows — this pins the shape of the catalog
    // instead: each capability is covered, and the entries carry the metadata the UI gates on.
    @Test func modelCapabilitiesAreDiscoverable() {
        let generationModels = ModelCatalog.builtIn.filter {
            $0.capabilities.contains(.textToImage)
        }

        #expect(!generationModels.isEmpty)
        #expect(generationModels.allSatisfy { $0.recommendedMemoryGB > 0 })
        #expect(ModelCatalog.builtIn.contains { $0.id == "realesrgan-x2@coreml" })

        let qwenImageEditModels = ModelCatalog.builtIn.filter {
            $0.capabilities.contains(.imageToImage)
        }
        #expect(!qwenImageEditModels.isEmpty)
        // Qwen Image Edit ships in three quantizations and the Runtime picks between them.
        #expect(Set(qwenImageEditModels.map(\.quantization)).isSuperset(of: [
            ModelQuantization.fourBit,
            ModelQuantization.eightBit,
            ModelQuantization.fp16
        ]))
    }

    @Test func builtInCatalogEntriesAreWellFormed() {
        let identifiers = ModelCatalog.builtIn.map(\.id)

        #expect(Set(identifiers).count == identifiers.count)
        #expect(ModelCatalog.builtIn.allSatisfy { !$0.id.isEmpty })
        #expect(ModelCatalog.builtIn.allSatisfy { !$0.capabilities.isEmpty })
        #expect(ModelCatalog.builtIn.allSatisfy { $0.recommendedMemoryGB > 0 })
    }

    @Test func recipeSizePresetChangesBothDimensions() {
        var recipe = GenerationRecipe(modelID: "test")

        recipe.applySizePreset(width: 768, height: 1024)

        #expect(recipe.width == 768)
        #expect(recipe.height == 1024)
    }

    @Test func everyPrimaryActionHasAnIndependentProfile() {
        let capabilities = Set(ModelCatalog.builtInProfiles.map(\.capability))

        #expect(capabilities.contains(.textToImage))
        #expect(capabilities.contains(.imageToText))
        #expect(capabilities.contains(.imageToImage))
        #expect(capabilities.contains(.upscale))
    }

    @Test func operationKeepsProfileSnapshot() {
        let projectID = UUID()
        let profile = ModelCatalog.builtInProfiles[0]
        let operation = WorkflowOperation(
            projectID: projectID,
            action: .generate,
            profileSnapshot: profile
        )

        #expect(operation.profileSnapshot?.modelRevision == "main")
        #expect(operation.profileSnapshot?.architecture == .mlxSwift)
    }

    @Test func recipeRejectsDimensionsThatInferenceCannotUse() {
        var recipe = GenerationRecipe(
            prompt: "test",
            modelID: "test"
        )
        recipe.width = 1000

        #expect(throws: RecipeValidationError.self) {
            try recipe.validate()
        }
    }

    @Test func localDiscoveryCreatesIndependentProfiles() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("genimage-discovery-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let requiredFiles = [
            "z-image-turbo-q4/model_index.json",
            "z-image-turbo-q4/quantization.json",
            "z-image-turbo-q4/text_encoder/model.safetensors",
            "z-image-turbo-q4/transformer/model-00001-of-00002.safetensors",
            "z-image-turbo-q4/transformer/model-00002-of-00002.safetensors",
            "z-image-turbo-q4/vae/diffusion_pytorch_model.safetensors",
            "z-image-turbo-q4/tokenizer/tokenizer.json",
            "Qwen3-VL-4B-Instruct-4bit/config.json",
            "Qwen3-VL-4B-Instruct-4bit/model.safetensors",
            "Qwen3-VL-4B-Instruct-4bit/tokenizer.json",
            "Qwen3-VL-4B-Instruct-4bit/tokenizer_config.json",
            "Qwen3-VL-4B-Instruct-4bit/preprocessor_config.json",
            "Qwen3-VL-4B-Instruct-4bit/processor_config.json",
            "Qwen3-VL-4B-Instruct-4bit/video_preprocessor_config.json",
            "upscale/realesrgan512.mlmodel",
            "z-image-turbo-loras/example-style.safetensors"
        ]

        for relativePath in requiredFiles {
            let fileURL = root.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data("fixture".utf8).write(to: fileURL)
        }

        let result = LocalModelDiscovery.discover(at: root)
        let capabilities = Set(result.profiles.map(\.capability))

        #expect(capabilities.contains(.textToImage))
        #expect(capabilities.contains(.imageToText))
        #expect(capabilities.contains(.textToText))
        #expect(capabilities.contains(.upscale))
        #expect(result.profiles.contains {
            $0.capability == .upscale && $0.defaults.upscaleScale == 2
        })
        #expect(result.profiles.contains {
            $0.capability == .upscale && $0.defaults.upscaleScale == 4
        })
        #expect(result.models.allSatisfy { $0.localURL != nil })
        #expect(result.loras.count == 1)
        #expect(result.loras.first?.displayName == "example-style")
    }

    @Test func localDiscoveryRecognizesLTXGGUFWeight() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("genimage-ltx-gguf-\(UUID().uuidString)", isDirectory: true)
        let directory = root.appendingPathComponent(
            "ltx-video-0.9.6-distilled-gguf",
            isDirectory: true
        )
        let weightURL = directory.appendingPathComponent(
            "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M.gguf"
        )
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: weightURL.path, contents: nil)
        let handle = try FileHandle(forWritingTo: weightURL)
        try handle.seek(toOffset: 1_330_049_535)
        try handle.write(contentsOf: Data([0]))
        try handle.close()
        let companionPaths = [
            "LTX-Video-0.9.6-VAE-BF16.safetensors",
            "text_encoder/config.json",
            "text_encoder/t5-v1_1-xxl-encoder-Q4_K_M.gguf",
            "tokenizer/spiece.model"
        ]
        for relativePath in companionPaths {
            let companionURL = directory.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: companionURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data("fixture".utf8).write(to: companionURL)
        }

        let discovered = LocalModelDiscovery.discover(at: root)

        #expect(discovered.models.contains {
            $0.id == "city96/LTX-Video-0.9.6-distilled-gguf@Q4_K_M"
        })
    }

    @Test func recipeRejectsOutOfRangeLoRAScale() {
        var recipe = GenerationRecipe(prompt: "test", modelID: "test")
        recipe.lora = LoRASelection(
            adapterID: "test-lora",
            localURL: URL(fileURLWithPath: "/tmp/test-lora.safetensors"),
            scale: 1.1
        )

        #expect(throws: RecipeValidationError.self) {
            try recipe.validate()
        }
    }
}
