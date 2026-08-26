import Testing

@testable import GenImageCore

struct AssetKindTests {
    @Test func imageClassificationCoversEveryAssetKind() {
        let expected: [AssetKind: Bool] = [
            .imported: true,
            .importedVideo: false,
            .importedAudio: false,
            .generated: true,
            .generatedVideo: false,
            .generatedAudio: false,
            .generatedSubtitle: false,
            .edited: true,
            .upscaled: true
        ]

        #expect(Set(expected.keys) == Set(AssetKind.allCases))
        for kind in AssetKind.allCases {
            #expect(kind.isImage == expected[kind])
        }
    }

    @Test func timedMediaClassificationCoversEveryAssetKind() {
        let expected: [AssetKind: Bool] = [
            .imported: false,
            .importedVideo: true,
            .importedAudio: true,
            .generated: false,
            .generatedVideo: true,
            .generatedAudio: true,
            .generatedSubtitle: false,
            .edited: false,
            .upscaled: false
        ]

        #expect(Set(expected.keys) == Set(AssetKind.allCases))
        for kind in AssetKind.allCases {
            #expect(kind.isTimedMedia == expected[kind])
        }
    }
}
