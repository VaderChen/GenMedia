import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM
import Testing

@Suite(.serialized)
struct Qwen21VisionOracleTests {
    private struct Tensor: Decodable {
        let shape: [Int]
        let values: [Float]
        var array: MLXArray { MLXArray(values).reshaped(shape) }
    }

    private struct ImageCase: Decodable {
        let name: String
        let grid: [Int]
        let tokens: [Int]
        let pixels: Tensor
        let expected: Tensor
        let sensitivity: [String: Float]
    }

    private struct Fixture: Decodable {
        let configuration: String
        let weights: [String: Tensor]
        let cases: [ImageCase]
    }

    @Test func imageConditioningMatchesIndependentVisionAndTextOracle() throws {
        let url = try #require(Bundle.module.url(
            forResource: "vision-oracle", withExtension: "json", subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        let configuration = try JSONDecoder().decode(
            Qwen3VLConfiguration.self, from: Data(fixture.configuration.utf8))
        let model = Qwen3VL(configuration)
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        // Fully deterministic parameters: no randomly initialized model tensor may remain.
        #expect(Set(parameters.keys) == Set(fixture.weights.keys))
        for (key, value) in fixture.weights {
            let parameter = try #require(parameters[key])
            #expect(parameter.shape == value.shape)
        }
        try model.update(parameters: .unflattened(fixture.weights.mapValues(\.array)), verify: .all)
        model.train(false)

        for item in fixture.cases {
            let image = LMInput.ProcessedImage(
                pixels: item.pixels.array,
                frames: [THW(item.grid[0], item.grid[1], item.grid[2])])
            let input = LMInput(text: .init(tokens: MLXArray(item.tokens).reshaped([1, -1])), image: image)
            let actual = try model.encodeImageConditioning(input)
            #expect(actual.shape == item.expected.shape)
            let maximumError = max(abs(actual - item.expected.array)).item(Float.self)
            #expect(maximumError < 0.0001, "\(item.name): max absolute error \(maximumError)")

            // Keep the fixture sensitive to the activation, deepstack, and MRoPE regressions.
            for (mutation, error) in item.sensitivity {
                #expect(error > 0.001, "Oracle does not distinguish \(mutation)")
            }
        }
    }
}
