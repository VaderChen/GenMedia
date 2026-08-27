import Foundation
import GenImageCore
import Testing

@testable import GenImageRuntime

struct MediaCompatibilityServiceTests {
    @Test func configuredFFmpegOverridesBundleAndPathCandidates() {
        let bundleURL = URL(fileURLWithPath: "/Applications/GenMedia.app")
        let executableURL = bundleURL
            .appendingPathComponent("Contents/MacOS/GenMedia")
        let configured = URL(fileURLWithPath: "/custom/ffmpeg")
        let candidates = MediaCompatibilityService.executableCandidates(
            for: .ffmpeg,
            environment: [
                "GENMEDIA_FFMPEG": configured.path,
                "PATH": "/custom/bin"
            ],
            bundleURL: bundleURL,
            executableURL: executableURL
        )

        #expect(candidates.first == configured)
        #expect(candidates.contains(bundleURL
            .appendingPathComponent("Contents/Resources/bin/ffmpeg")))
        #expect(candidates.contains(URL(fileURLWithPath: "/custom/bin/ffmpeg")))
        #expect(candidates.filter { $0 == bundleURL
            .appendingPathComponent("Contents/Resources/bin/ffmpeg") }.count == 1)
    }

    @Test func ffprobeCanBeDerivedFromConfiguredFFmpegPath() {
        let candidates = MediaCompatibilityService.executableCandidates(
            for: .ffprobe,
            environment: ["GENMEDIA_FFMPEG": "/custom/tools/ffmpeg"],
            bundleURL: nil,
            executableURL: nil
        )

        #expect(candidates.first == URL(fileURLWithPath: "/custom/tools/ffprobe"))
    }

    @Test func bundleFFmpegPrecedesPathWhenNoOverrideIsConfigured() {
        let bundleURL = URL(fileURLWithPath: "/Applications/GenMedia.app")
        let bundleCandidate = bundleURL
            .appendingPathComponent("Contents/Resources/bin/ffmpeg")
        let pathCandidate = URL(fileURLWithPath: "/custom/bin/ffmpeg")
        let candidates = MediaCompatibilityService.executableCandidates(
            for: .ffmpeg,
            environment: ["PATH": "/custom/bin"],
            bundleURL: bundleURL,
            executableURL: nil
        )

        #expect(candidates.first == bundleCandidate)
        #expect(candidates.firstIndex(of: bundleCandidate)! < candidates.firstIndex(of: pathCandidate)!)
    }

    @Test func bitrateScalesWithVideoResolution() {
        #expect(MediaCompatibilityService.videoBitrateArguments(
            pixelWidth: 1_280,
            pixelHeight: 720
        ) == ["-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M"])
        #expect(MediaCompatibilityService.videoBitrateArguments(
            pixelWidth: 512,
            pixelHeight: 512
        ) == ["-b:v", "2300k", "-maxrate", "3400k", "-bufsize", "4600k"])
        #expect(MediaCompatibilityService.videoBitrateArguments(
            pixelWidth: 3_840,
            pixelHeight: 2_160
        ) == ["-b:v", "40M", "-maxrate", "60M", "-bufsize", "80M"])
    }
}

struct MediaSourceCompatibilityServiceTests {
    @Test func playbackPlanKeepsDirectlyPlayableAudio() {
        let plan = MediaSourceCompatibilityService.playbackPlan(
            sourceExtension: "mp3",
            probe: probe(
                hasAudio: true,
                audioCodec: "mp3"
            )
        )

        #expect(plan == .original)
    }

    @Test func playbackPlanConvertsUnsupportedAudio() {
        let plan = MediaSourceCompatibilityService.playbackPlan(
            sourceExtension: "ogg",
            probe: probe(
                hasAudio: true,
                audioCodec: "vorbis"
            )
        )

        #expect(plan == .audioM4A)
    }

    @Test func playbackPlanKeepsNativeH264Video() {
        let plan = MediaSourceCompatibilityService.playbackPlan(
            sourceExtension: "mp4",
            probe: probe(
                hasVideo: true,
                hasAudio: true,
                videoCodec: "h264",
                audioCodec: "aac"
            )
        )

        #expect(plan == .original)
    }

    @Test func playbackPlanRemuxesH264WhenContainerOrAudioDiffers() {
        let containerMismatch = MediaSourceCompatibilityService.playbackPlan(
            sourceExtension: "mkv",
            probe: probe(
                hasVideo: true,
                hasAudio: true,
                videoCodec: "h264",
                audioCodec: "aac"
            )
        )
        let audioMismatch = MediaSourceCompatibilityService.playbackPlan(
            sourceExtension: "mp4",
            probe: probe(
                hasVideo: true,
                hasAudio: true,
                videoCodec: "h264",
                audioCodec: "opus"
            )
        )

        #expect(containerMismatch == .videoMP4(
            videoCodec: "copy",
            audioCodec: "copy",
            videoTag: nil
        ))
        #expect(audioMismatch == .videoMP4(
            videoCodec: "copy",
            audioCodec: "aac",
            videoTag: nil
        ))
    }

    @Test func playbackPlanHandlesHEVCAndUnknownVideo() {
        let nativeHEVC = MediaSourceCompatibilityService.playbackPlan(
            sourceExtension: "mp4",
            probe: probe(
                hasVideo: true,
                hasAudio: true,
                videoCodec: "hevc",
                videoCodecTag: "hvc1",
                audioCodec: "aac"
            )
        )
        let remuxableHEVC = MediaSourceCompatibilityService.playbackPlan(
            sourceExtension: "mkv",
            probe: probe(
                hasVideo: true,
                hasAudio: true,
                videoCodec: "hevc",
                audioCodec: "aac"
            )
        )
        let unknownVideo = MediaSourceCompatibilityService.playbackPlan(
            sourceExtension: "mp4",
            probe: probe(
                hasVideo: true,
                videoCodec: "vp9"
            )
        )

        #expect(nativeHEVC == .original)
        #expect(remuxableHEVC == .videoMP4(
            videoCodec: "copy",
            audioCodec: "copy",
            videoTag: "hvc1"
        ))
        #expect(unknownVideo == .videoH264)
    }

    @Test func playbackPlanHandlesVideoWithoutAudio() {
        let plan = MediaSourceCompatibilityService.playbackPlan(
            sourceExtension: "mov",
            probe: probe(
                hasVideo: true,
                videoCodec: "h264"
            )
        )

        #expect(plan == .original)
    }

    private func probe(
        hasVideo: Bool = false,
        hasAudio: Bool = false,
        videoCodec: String? = nil,
        videoCodecTag: String? = nil,
        audioCodec: String? = nil
    ) -> MediaProbeResult {
        MediaProbeResult(
            durationSeconds: 12,
            hasAudio: hasAudio,
            hasVideo: hasVideo,
            videoCodec: videoCodec,
            videoCodecTag: videoCodecTag,
            audioCodec: audioCodec,
            pixelWidth: hasVideo ? 1_920 : 0,
            pixelHeight: hasVideo ? 1_080 : 0
        )
    }
}
