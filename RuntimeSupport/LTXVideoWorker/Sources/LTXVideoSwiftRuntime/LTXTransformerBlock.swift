import MLX
import MLXNN

public final class LTXBasicAVTransformerBlock: Module {
    @ModuleInfo(key: "attn1") public var attn1: LTXAttention
    @ModuleInfo(key: "audio_attn1") public var audioAttn1: LTXAttention
    @ModuleInfo(key: "attn2") public var attn2: LTXAttention
    @ModuleInfo(key: "audio_attn2") public var audioAttn2: LTXAttention
    @ModuleInfo(key: "audio_to_video_attn") public var audioToVideoAttn: LTXAttention
    @ModuleInfo(key: "video_to_audio_attn") public var videoToAudioAttn: LTXAttention
    @ModuleInfo(key: "ff") public var ff: LTXFeedForward
    @ModuleInfo(key: "audio_ff") public var audioFF: LTXFeedForward

    @ParameterInfo(key: "scale_shift_table") public var scale_shift_table: MLXArray
    @ParameterInfo(key: "audio_scale_shift_table") public var audio_scale_shift_table: MLXArray
    @ParameterInfo(key: "prompt_scale_shift_table") public var prompt_scale_shift_table: MLXArray
    @ParameterInfo(key: "audio_prompt_scale_shift_table") public var audio_prompt_scale_shift_table: MLXArray
    @ParameterInfo(key: "scale_shift_table_a2v_ca_video") public var scale_shift_table_a2v_ca_video: MLXArray
    @ParameterInfo(key: "scale_shift_table_a2v_ca_audio") public var scale_shift_table_a2v_ca_audio: MLXArray
    private let normEps: Float

    public init(configuration: LTXTransformerConfiguration) {
        let videoDimension = configuration.videoDim
        let audioDimension = configuration.audioDim
        self._attn1 = ModuleInfo(wrappedValue: LTXAttention(
            queryDimension: videoDimension,
            numHeads: configuration.videoNumHeads,
            headDimension: configuration.videoHeadDim,
            useRoPE: true,
            normEps: configuration.normEps
        ), key: "attn1")
        self._audioAttn1 = ModuleInfo(wrappedValue: LTXAttention(
            queryDimension: audioDimension,
            numHeads: configuration.audioNumHeads,
            headDimension: configuration.audioHeadDim,
            useRoPE: true,
            normEps: configuration.normEps
        ), key: "audio_attn1")
        self._attn2 = ModuleInfo(wrappedValue: LTXAttention(
            queryDimension: videoDimension,
            numHeads: configuration.videoNumHeads,
            headDimension: configuration.videoHeadDim,
            useRoPE: false,
            normEps: configuration.normEps
        ), key: "attn2")
        self._audioAttn2 = ModuleInfo(wrappedValue: LTXAttention(
            queryDimension: audioDimension,
            numHeads: configuration.audioNumHeads,
            headDimension: configuration.audioHeadDim,
            useRoPE: false,
            normEps: configuration.normEps
        ), key: "audio_attn2")
        self._audioToVideoAttn = ModuleInfo(wrappedValue: LTXAttention(
            queryDimension: videoDimension,
            keyValueDimension: audioDimension,
            outputDimension: videoDimension,
            numHeads: configuration.avCrossNumHeads,
            headDimension: configuration.avCrossHeadDim,
            useRoPE: true,
            normEps: configuration.normEps
        ), key: "audio_to_video_attn")
        self._videoToAudioAttn = ModuleInfo(wrappedValue: LTXAttention(
            queryDimension: audioDimension,
            keyValueDimension: videoDimension,
            outputDimension: audioDimension,
            numHeads: configuration.avCrossNumHeads,
            headDimension: configuration.avCrossHeadDim,
            useRoPE: true,
            normEps: configuration.normEps
        ), key: "video_to_audio_attn")
        self._ff = ModuleInfo(wrappedValue: LTXFeedForward(
            dimension: videoDimension, multiplier: configuration.ffMult
        ), key: "ff")
        self._audioFF = ModuleInfo(wrappedValue: LTXFeedForward(
            dimension: audioDimension, multiplier: configuration.ffMult
        ), key: "audio_ff")
        self._scale_shift_table = ParameterInfo(
            wrappedValue: zeros([9, videoDimension]), key: "scale_shift_table"
        )
        self._audio_scale_shift_table = ParameterInfo(
            wrappedValue: zeros([9, audioDimension]), key: "audio_scale_shift_table"
        )
        self._prompt_scale_shift_table = ParameterInfo(
            wrappedValue: zeros([2, videoDimension]), key: "prompt_scale_shift_table"
        )
        self._audio_prompt_scale_shift_table = ParameterInfo(
            wrappedValue: zeros([2, audioDimension]), key: "audio_prompt_scale_shift_table"
        )
        self._scale_shift_table_a2v_ca_video = ParameterInfo(
            wrappedValue: zeros([5, videoDimension]), key: "scale_shift_table_a2v_ca_video"
        )
        self._scale_shift_table_a2v_ca_audio = ParameterInfo(
            wrappedValue: zeros([5, audioDimension]), key: "scale_shift_table_a2v_ca_audio"
        )
        self.normEps = configuration.normEps
        super.init()
    }

    private func unpack(
        _ parameters: MLXArray,
        table: MLXArray,
        count: Int,
        dimension: Int
    ) -> [MLXArray] {
        if parameters.ndim == 2 {
            let values = parameters.reshaped(-1, count, dimension)
                + table[.newAxis, ..<count, 0...]
            return (0..<count).map { values[0..., $0, .newAxis, 0...] }
        }
        let values = parameters.reshaped(parameters.shape[0], parameters.shape[1], count, dimension)
            + table[.newAxis, .newAxis, ..<count, 0...]
        return (0..<count).map { values[0..., 0..., $0, 0...] }
    }

    public func callAsFunction(
        video: MLXArray,
        audio: MLXArray,
        videoAdaLN: MLXArray,
        audioAdaLN: MLXArray,
        videoPromptAdaLN: MLXArray,
        audioPromptAdaLN: MLXArray,
        avVideo: MLXArray,
        avAudio: MLXArray,
        avA2VGate: MLXArray,
        avV2AGate: MLXArray,
        videoText: MLXArray? = nil,
        audioText: MLXArray? = nil,
        videoRoPE: LTXRoPEFrequencies? = nil,
        audioRoPE: LTXRoPEFrequencies? = nil,
        videoCrossRoPE: LTXRoPEFrequencies? = nil,
        audioCrossRoPE: LTXRoPEFrequencies? = nil,
        videoMask: MLXArray? = nil,
        audioMask: MLXArray? = nil,
        videoCrossMask: MLXArray? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        let videoValues = unpack(videoAdaLN, table: scale_shift_table, count: 9, dimension: video.shape[2])
        let audioValues = unpack(audioAdaLN, table: audio_scale_shift_table, count: 9, dimension: audio.shape[2])
        let videoAV = unpack(avVideo, table: scale_shift_table_a2v_ca_video, count: 4, dimension: video.shape[2])
        let audioAV = unpack(avAudio, table: scale_shift_table_a2v_ca_audio, count: 4, dimension: audio.shape[2])
        let videoPrompt = unpack(videoPromptAdaLN, table: prompt_scale_shift_table, count: 2, dimension: video.shape[2])
        let audioPrompt = unpack(audioPromptAdaLN, table: audio_prompt_scale_shift_table, count: 2, dimension: audio.shape[2])

        var video = video + attn1(
            LTXTransformerOps.rmsNorm(video, eps: normEps) * (1 + videoValues[1]) + videoValues[0],
            rope: videoRoPE,
            mask: videoMask
        ) * videoValues[2]
        var audio = audio + audioAttn1(
            LTXTransformerOps.rmsNorm(audio, eps: normEps) * (1 + audioValues[1]) + audioValues[0],
            rope: audioRoPE,
            mask: audioMask
        ) * audioValues[2]

        if let videoText {
            let text = videoText * (1 + videoPrompt[1]) + videoPrompt[0]
            video = video + attn2(
                LTXTransformerOps.rmsNorm(video, eps: normEps) * (1 + videoValues[7]) + videoValues[6],
                encoderHiddenStates: text,
                mask: videoCrossMask
            ) * videoValues[8]
        }
        if let audioText {
            let text = audioText * (1 + audioPrompt[1]) + audioPrompt[0]
            audio = audio + audioAttn2(
                LTXTransformerOps.rmsNorm(audio, eps: normEps) * (1 + audioValues[7]) + audioValues[6],
                encoderHiddenStates: text
            ) * audioValues[8]
        }

        let videoNorm = LTXTransformerOps.rmsNorm(video, eps: normEps)
        let audioNorm = LTXTransformerOps.rmsNorm(audio, eps: normEps)
        let videoGate = avA2VGate.ndim == 2
            ? (avA2VGate + scale_shift_table_a2v_ca_video[4, 0...])
                .reshaped(avA2VGate.shape[0], 1, avA2VGate.shape[1])
            : avA2VGate + scale_shift_table_a2v_ca_video[4, 0...]
        let audioGate = avV2AGate.ndim == 2
            ? (avV2AGate + scale_shift_table_a2v_ca_audio[4, 0...])
                .reshaped(avV2AGate.shape[0], 1, avV2AGate.shape[1])
            : avV2AGate + scale_shift_table_a2v_ca_audio[4, 0...]
        let a2v = audioToVideoAttn(
            videoNorm * (1 + videoAV[0]) + videoAV[1],
            encoderHiddenStates: audioNorm * (1 + audioAV[0]) + audioAV[1],
            rope: videoCrossRoPE,
            keyRoPE: audioCrossRoPE
        ) * videoGate
        let v2a = videoToAudioAttn(
            audioNorm * (1 + audioAV[2]) + audioAV[3],
            encoderHiddenStates: videoNorm * (1 + videoAV[2]) + videoAV[3],
            rope: audioCrossRoPE,
            keyRoPE: videoCrossRoPE
        ) * audioGate
        video = video + a2v
        audio = audio + v2a
        video = video + ff(
            LTXTransformerOps.rmsNorm(video, eps: normEps) * (1 + videoValues[4]) + videoValues[3]
        ) * videoValues[5]
        audio = audio + audioFF(
            LTXTransformerOps.rmsNorm(audio, eps: normEps) * (1 + audioValues[4]) + audioValues[3]
        ) * audioValues[5]
        return (video, audio)
    }
}
