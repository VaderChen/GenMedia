import Foundation
import Testing

@testable import MiniMaxH3SwiftRuntime

@Suite("MiniMax H3 configuration")
struct MiniMaxH3ConfigurationTests {
    private let configuration = MiniMaxH3Configuration.fl2va

    @Test("Attention width is not the hidden size")
    func attentionInnerDimIsDistinctFromHidden() {
        // 56 * 128 = 7168 while hidden is 5376, so qkv/out projections are not
        // square. Assuming inner == hidden silently mis-shapes both.
        #expect(configuration.attentionInnerDim == 7168)
        #expect(configuration.attentionInnerDim != configuration.hiddenSize)
    }

    @Test("Video patch dim follows latent channels and patch size")
    func videoPatchDimMatchesWeights() {
        // 24 channels * 1 * 2 * 2 = 96, matching video_patch_proj/video_out.
        #expect(configuration.videoPatchDim == 96)
    }

    @Test("AdaLN widths follow expand * modalities * hidden")
    func adalnWidths() {
        #expect(configuration.adalnOutputDim == 96768)
        #expect(configuration.finalAdalnOutputDim == 10752)
    }

    @Test("Patch projections and output projections face opposite ways")
    func patchProjectionDirection() {
        let expected = configuration.expectedTensorShapes
        // Linear(patch_dim -> hidden) going in, Linear(hidden -> patch_dim)
        // coming out. These are transposes; getting one backwards still has a
        // matching element count, so only the direction distinguishes them.
        #expect(expected["video_patch_proj.weight"] == [5376, 96])
        #expect(expected["final_layer.video_out.weight"] == [96, 5376])
        #expect(expected["audio_patch_proj.weight"] == [5376, 32])
        #expect(expected["final_layer.audio_out.weight"] == [32, 5376])
        // The bias width is what pins the direction down: it equals
        // out_features.
        #expect(expected["video_patch_proj.bias"] == [5376])
        #expect(expected["final_layer.video_out.bias"] == [96])
    }

    @Test("Expected tensor set covers every block")
    func expectedTensorCoverage() {
        let expected = configuration.expectedTensorShapes
        #expect(expected["blocks.49.adaln_proj.linear.weight"] == [96768, 2688])
        #expect(expected["blocks.0.attn.qkv_proj.weight"] == [21504, 5376])
        #expect(expected["blocks.0.attn.out_proj.weight"] == [5376, 7168])
        // fc1 is SwiGLU-gated, so it is 2x the feed-forward width.
        #expect(expected["blocks.0.mlp.fc1.weight"] == [28672, 5376])
        #expect(expected["blocks.0.mlp.fc2.weight"] == [5376, 14336])
        #expect(expected["token_refiner.blocks.1.mlp.fc1.weight"] == [28672, 5376])
    }

    @Test("Pruned checkpoints use the compact AdaLN curve")
    func prunedConfigurationUsesAdalnCurve() {
        let pruned = MiniMaxH3Configuration.fl2vaPruned
        let expected = pruned.expectedTensorShapes

        #expect(pruned.usesAdalnCurves)
        #expect(pruned.timeEmbedDim == 8)
        #expect(expected["adaln_t_table"] == [1025, 8])
        #expect(expected["blocks.0.adaln_proj.linear.weight"] == [96768, 8])
        #expect(expected["final_layer.adaln_proj.linear.weight"] == [10752, 8])
        #expect(expected["time_embedder.proj_in.weight"] == nil)
    }

    @Test("Inventory detection selects the pruned configuration")
    func inventoryDetectionSelectsPrunedConfiguration() throws {
        let pruned = MiniMaxH3Configuration.fl2vaPruned
        let inventory = Self.inventory(from: pruned.expectedTensorShapes)

        #expect(try MiniMaxH3Configuration.forInventory(inventory) == pruned)
    }

    @Test("Validation accepts a matching inventory")
    func validationAcceptsMatchingInventory() throws {
        let inventory = Self.inventory(from: configuration.expectedTensorShapes)
        try configuration.validate(against: inventory)
    }

    @Test("Validation rejects a missing tensor")
    func validationRejectsMissingTensor() {
        var shapes = configuration.expectedTensorShapes
        shapes.removeValue(forKey: "blocks.7.attn.qkv_proj.weight")
        let inventory = Self.inventory(from: shapes)
        #expect(throws: MiniMaxH3WeightError.self) {
            try configuration.validate(against: inventory)
        }
    }

    @Test("Validation rejects a transposed weight")
    func validationRejectsTransposedWeight() {
        // The exact failure the ComfyUI orig_shape metadata prevents: same
        // element count, wrong shape.
        var shapes = configuration.expectedTensorShapes
        shapes["blocks.0.adaln_proj.linear.weight"] = [2688, 96768]
        let inventory = Self.inventory(from: shapes)
        #expect(throws: MiniMaxH3WeightError.self) {
            try configuration.validate(against: inventory)
        }
    }

    @Test("Override is only flagged when it changes the shape")
    func overrideFlagging() {
        let unchanged = MiniMaxH3WeightInventory.Entry(
            name: "a", storedShape: [4, 8], logicalShape: [4, 8],
            ggmlType: "Q4_0", usedOrigShape: true
        )
        let changed = MiniMaxH3WeightInventory.Entry(
            name: "b", storedShape: [32, 1], logicalShape: [4, 8],
            ggmlType: "Q4_0", usedOrigShape: true
        )
        #expect(!unchanged.overrideChangesShape)
        #expect(changed.overrideChangesShape)
    }

    private static func inventory(
        from shapes: [String: [Int]]
    ) -> MiniMaxH3WeightInventory {
        MiniMaxH3WeightInventory(
            architecture: "wan",
            entries: shapes.map { name, shape in
                MiniMaxH3WeightInventory.Entry(
                    name: name,
                    storedShape: shape,
                    logicalShape: shape,
                    ggmlType: "Q4_0",
                    usedOrigShape: false
                )
            }.sorted { $0.name < $1.name }
        )
    }
}
