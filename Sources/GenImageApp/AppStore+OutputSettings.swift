import Foundation
import GenImageCore
import GenImageRuntime

// 圖片、影片與音樂的輸出參數。
extension AppStore {
    func applyActiveGenerationProfileDefaults() {
        guard let profile = activeProfile(for: .textToImage) else { return }
        recipe.width = profile.defaults.width ?? recipe.width
        recipe.height = profile.defaults.height ?? recipe.height
        recipe.steps = profile.defaults.steps ?? recipe.steps
        recipe.outputCount = profile.defaults.outputCount ?? recipe.outputCount
    }

    func applyActiveVideoProfileDefaults() {
        guard let profile = preferredVideoProfile else { return }
        var updated = videoOutputSettings
        updated.width = profile.defaults.width ?? updated.width
        updated.height = profile.defaults.height ?? updated.height
        updated.steps = profile.defaults.steps ?? updated.steps
        updated.outputCount = profile.defaults.outputCount ?? updated.outputCount
        updated.frameCount = normalizedVideoFrameCount(
            profile.defaults.frameCount ?? updated.frameCount,
            for: profile
        )
        updated.frameRate = profile.defaults.frameRate ?? updated.frameRate
        videoOutputSettings = updated
    }

    func randomizeSeed() {
        recipe.seed = UInt64.random(in: 0...UInt64.max)
    }

    func randomizeVideoSeed() {
        videoOutputSettings.seed = UInt64.random(in: 0...UInt64.max)
    }

    func applyActiveMusicProfileDefaults() {
        guard let profile = activeProfile(for: .textToMusic) else { return }
        var updated = musicOutputSettings
        updated.durationSeconds = profile.defaults.durationSeconds ?? updated.durationSeconds
        updated.steps = profile.defaults.steps ?? updated.steps
        musicOutputSettings = updated
    }

    func randomizeMusicSeed() {
        musicOutputSettings.seed = UInt64.random(in: 0...UInt64.max)
    }

    func updateMusicOutputSettings(
        prompt: String?,
        lyrics: String?,
        style: MusicStyle?,
        durationSeconds: Int?,
        steps: Int?,
        seed: UInt64?,
        format: AudioOutputFormat?
    ) {
        var updated = musicOutputSettings
        if let prompt { updated.prompt = prompt }
        if let lyrics { updated.lyrics = lyrics }
        if let style { updated.style = style }
        if let durationSeconds {
            updated.durationSeconds = min(
                max(durationSeconds, MusicGenerationOptions.supportedDurationSeconds.lowerBound),
                MusicGenerationOptions.supportedDurationSeconds.upperBound
            )
        }
        if let steps { updated.steps = min(max(steps, 1), 100) }
        if let seed { updated.seed = seed }
        if let format { updated.format = format }
        musicOutputSettings = updated
    }

    func updateVideoOutputSettings(
        width: Int?,
        height: Int?,
        steps: Int?,
        outputCount: Int?,
        frameCount: Int?,
        frameRate: Int?,
        seed: UInt64?
    ) {
        var updated = videoOutputSettings
        if let width { updated.width = OutputGeometry.quantize(width) }
        if let height { updated.height = OutputGeometry.quantize(height) }
        if let steps { updated.steps = min(max(steps, 1), 100) }
        if let outputCount { updated.outputCount = min(max(outputCount, 1), 8) }
        if let frameCount {
            updated.frameCount = normalizedVideoFrameCount(
                frameCount,
                for: preferredVideoProfile
            )
            if updated.frameCount != frameCount {
                statusMessage = "LTX-2.3 幀數已從 \(frameCount) 自動調整為合法值 \(updated.frameCount)（8n+1）。"
            }
        }
        if let frameRate { updated.frameRate = min(max(frameRate, 1), 120) }
        if let seed { updated.seed = seed }
        videoOutputSettings = updated
    }

    func normalizedVideoFrameCount(
        _ frameCount: Int,
        for profile: InferenceProfile?
    ) -> Int {
        let clamped = min(max(frameCount, 1), 512)
        if profile?.modelID.lowercased().contains("ltx-2.3-mlx") == true {
            return LTXVideoGenerationService.normalizedFrameCount(clamped)
        }
        if profile.map({ MiniMaxH3VideoGenerationService.isSupportedModelID($0.modelID) }) == true {
            return MiniMaxH3VideoGenerationService.normalizedFrameCount(clamped)
        }
        return clamped
    }

    func chooseSize(width: Int, height: Int) {
        recipe.applySizePreset(width: width, height: height)
    }
}
