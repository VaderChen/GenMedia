import Foundation
import Testing

@testable import GenImageRuntime

struct MediaAudioPreparerTests {
    @Test func speechRecognitionFormatResamplesAndMixesToMono() {
        let format = MediaAudioPreparer.speechRecognitionFormat

        #expect(format.sampleRate == 16_000)
        #expect(format.channelCount == 1)
        #expect(
            format.outputFrameCount(
                inputFrameCount: 44_100,
                inputSampleRate: 44_100
            ) == 16_000
        )
        #expect(
            format.outputSampleCount(
                inputSampleCount: 88_200,
                inputSampleRate: 44_100,
                inputChannelCount: 2
            ) == 16_000
        )
        #expect(
            format.outputSampleCount(
                inputSampleCount: 0,
                inputSampleRate: 48_000,
                inputChannelCount: 2
            ) == 0
        )
    }

    @Test func pathsUseStableDirectoryAndAudioFilename() {
        let root = URL(fileURLWithPath: "/tmp/genmedia-tests", isDirectory: true)
        let identifier = UUID(uuidString: "12345678-1234-1234-1234-1234567890AB")!
        let paths = MediaAudioPreparer.paths(
            temporaryRoot: root,
            identifier: identifier
        )

        #expect(
            paths.temporaryDirectory.path
                == "/tmp/genmedia-tests/genmedia-asr-12345678-1234-1234-1234-1234567890AB"
        )
        #expect(paths.audioURL == paths.temporaryDirectory.appendingPathComponent("audio.m4a"))
    }
}
