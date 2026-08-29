import MLX
import MLXNN

public final class LTXTransformer: Module {
    @ModuleInfo(key: "patchify_proj") public var patchifyProj: Linear
    @ModuleInfo(key: "audio_patchify_proj") public var audioPatchifyProj: Linear
    @ModuleInfo(key: "proj_out") public var projOut: Linear
    @ModuleInfo(key: "audio_proj_out") public var audioProjOut: Linear
    @ModuleInfo(key: "adaln_single") public var adalnSingle: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_adaln_single") public var audioAdaLN: LTXAdaLayerNormSingle
    @ModuleInfo(key: "prompt_adaln_single") public var promptAdaLN: LTXAdaLayerNormSingle
    @ModuleInfo(key: "audio_prompt_adaln_single") public var audioPromptAdaLN: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_video_scale_shift_adaln_single") public var avVideoAdaLN: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_audio_scale_shift_adaln_single") public var avAudioAdaLN: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_a2v_gate_adaln_single") public var avA2VGateAdaLN: LTXAdaLayerNormSingle
    @ModuleInfo(key: "av_ca_v2a_gate_adaln_single") public var avV2AGateAdaLN: LTXAdaLayerNormSingle
    @ModuleInfo(key: "transformer_blocks") public var transformerBlocks: [LTXBasicAVTransformerBlock]

    @ParameterInfo(key: "scale_shift_table") public var scale_shift_table: MLXArray
    @ParameterInfo(key: "audio_scale_shift_table") public var audio_scale_shift_table: MLXArray
    public let configuration: LTXTransformerConfiguration

    public init(configuration: LTXTransformerConfiguration) {
        self.configuration = configuration
        self._patchifyProj = ModuleInfo(
            wrappedValue: Linear(configuration.videoPatchChannels, configuration.videoDim),
            key: "patchify_proj"
        )
        self._audioPatchifyProj = ModuleInfo(
            wrappedValue: Linear(configuration.audioPatchChannels, configuration.audioDim),
            key: "audio_patchify_proj"
        )
        self._projOut = ModuleInfo(
            wrappedValue: Linear(configuration.videoDim, configuration.videoPatchChannels),
            key: "proj_out"
        )
        self._audioProjOut = ModuleInfo(
            wrappedValue: Linear(configuration.audioDim, configuration.audioPatchChannels),
            key: "audio_proj_out"
        )
        self._adalnSingle = ModuleInfo(wrappedValue: LTXAdaLayerNormSingle(
            dimension: configuration.videoDim,
            parameterCount: 9,
            timestepDimension: configuration.timestepEmbeddingDim
        ), key: "adaln_single")
        self._audioAdaLN = ModuleInfo(wrappedValue: LTXAdaLayerNormSingle(
            dimension: configuration.audioDim,
            parameterCount: 9,
            timestepDimension: configuration.timestepEmbeddingDim
        ), key: "audio_adaln_single")
        self._promptAdaLN = ModuleInfo(wrappedValue: LTXAdaLayerNormSingle(
            dimension: configuration.videoDim,
            parameterCount: 2,
            timestepDimension: configuration.timestepEmbeddingDim
        ), key: "prompt_adaln_single")
        self._audioPromptAdaLN = ModuleInfo(wrappedValue: LTXAdaLayerNormSingle(
            dimension: configuration.audioDim,
            parameterCount: 2,
            timestepDimension: configuration.timestepEmbeddingDim
        ), key: "audio_prompt_adaln_single")
        self._avVideoAdaLN = ModuleInfo(wrappedValue: LTXAdaLayerNormSingle(
            dimension: configuration.videoDim,
            parameterCount: 4,
            timestepDimension: configuration.timestepEmbeddingDim
        ), key: "av_ca_video_scale_shift_adaln_single")
        self._avAudioAdaLN = ModuleInfo(wrappedValue: LTXAdaLayerNormSingle(
            dimension: configuration.audioDim,
            parameterCount: 4,
            timestepDimension: configuration.timestepEmbeddingDim
        ), key: "av_ca_audio_scale_shift_adaln_single")
        self._avA2VGateAdaLN = ModuleInfo(wrappedValue: LTXAdaLayerNormSingle(
            dimension: configuration.videoDim,
            parameterCount: 1,
            timestepDimension: configuration.timestepEmbeddingDim
        ), key: "av_ca_a2v_gate_adaln_single")
        self._avV2AGateAdaLN = ModuleInfo(wrappedValue: LTXAdaLayerNormSingle(
            dimension: configuration.audioDim,
            parameterCount: 1,
            timestepDimension: configuration.timestepEmbeddingDim
        ), key: "av_ca_v2a_gate_adaln_single")
        self._transformerBlocks = ModuleInfo(
            wrappedValue: (0..<configuration.numLayers).map { _ in
                LTXBasicAVTransformerBlock(configuration: configuration)
            },
            key: "transformer_blocks"
        )
        self._scale_shift_table = ParameterInfo(
            wrappedValue: zeros([2, configuration.videoDim]),
            key: "scale_shift_table"
        )
        self._audio_scale_shift_table = ParameterInfo(
            wrappedValue: zeros([2, configuration.audioDim]),
            key: "audio_scale_shift_table"
        )
        super.init()
    }

    private func timestepParameters(
        _ module: LTXAdaLayerNormSingle,
        timesteps: MLXArray
    ) -> (parameters: MLXArray, embedded: MLXArray) {
        if timesteps.ndim == 2 {
            let batch = timesteps.shape[0]
            let count = timesteps.shape[1]
            let embedding = LTXTransformerOps.timestepEmbedding(
                timesteps.reshaped(-1) * configuration.timestepScaleMultiplier,
                dimension: configuration.timestepEmbeddingDim
            )
            let values = module(embedding)
            return (
                values.parameters.reshaped(batch, count, -1),
                values.embedded.reshaped(batch, count, -1)
            )
        }
        let embedding = LTXTransformerOps.timestepEmbedding(
            timesteps.reshaped(-1) * configuration.timestepScaleMultiplier,
            dimension: configuration.timestepEmbeddingDim
        )
        return module(embedding)
    }

    private func output(
        _ hidden: MLXArray,
        embeddedTimestep: MLXArray,
        table: MLXArray,
        projection: Linear
    ) -> MLXArray {
        let embedded = embeddedTimestep.ndim == 2
            ? embeddedTimestep[0..., .newAxis, 0...]
            : embeddedTimestep
        let values = table[.newAxis, .newAxis, 0...] + embedded[0..., 0..., .newAxis, 0...]
        let shifted = LTXTransformerOps.layerNorm(hidden, eps: configuration.normEps)
            * (1 + values[0..., 0..., 1, 0...])
            + values[0..., 0..., 0, 0...]
        return projection(shifted)
    }

    public func callAsFunction(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        timestep: MLXArray,
        videoTextEmbeds: MLXArray? = nil,
        audioTextEmbeds: MLXArray? = nil,
        videoPositions: MLXArray? = nil,
        audioPositions: MLXArray? = nil,
        videoAttentionMask: MLXArray? = nil,
        audioAttentionMask: MLXArray? = nil,
        videoCrossAttentionMask: MLXArray? = nil,
        videoTimesteps: MLXArray? = nil,
        audioTimesteps: MLXArray? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        let videoLatent = videoLatent.asType(.bfloat16)
        let audioLatent = audioLatent.asType(.bfloat16)
        let timestep = timestep.asType(.bfloat16)
        let videoTextEmbeds = videoTextEmbeds?.asType(.bfloat16)
        let audioTextEmbeds = audioTextEmbeds?.asType(.bfloat16)
        let videoDType = videoLatent.dtype
        let audioDType = audioLatent.dtype
        let videoHidden = patchifyProj(videoLatent)
        let audioHidden = audioPatchifyProj(audioLatent)
        let globalTimestep = timestep.asType(videoDType)
        let globalEmbedding = LTXTransformerOps.timestepEmbedding(
            globalTimestep * configuration.timestepScaleMultiplier,
            dimension: configuration.timestepEmbeddingDim
        )
        let gateEmbedding = LTXTransformerOps.timestepEmbedding(
            timestep.asType(audioDType) * configuration.avCATimestepScaleMultiplier,
            dimension: configuration.timestepEmbeddingDim
        )

        let videoTime = timestepParameters(
            adalnSingle,
            timesteps: videoTimesteps ?? timestep
        )
        let audioTime = timestepParameters(
            audioAdaLN,
            timesteps: audioTimesteps ?? timestep
        )
        let videoPrompt = promptAdaLN(globalEmbedding)
        let audioPrompt = audioPromptAdaLN(globalEmbedding)
        let avVideo = timestepParameters(
            avVideoAdaLN,
            timesteps: videoTimesteps ?? timestep
        )
        let avAudio = timestepParameters(
            avAudioAdaLN,
            timesteps: audioTimesteps ?? timestep
        )
        let avA2VGate = avA2VGateAdaLN(gateEmbedding).parameters
        let avV2AGate = avV2AGateAdaLN(gateEmbedding).parameters

        let videoRoPE = videoPositions.map {
            LTXTransformerOps.precomputeRoPE(
                positions: $0,
                numHeads: configuration.videoNumHeads,
                headDimension: configuration.videoHeadDim,
                theta: configuration.ropeTheta,
                maxPositions: configuration.positionalEmbeddingMaxPos,
                type: configuration.ropeType
            )
        }
        let audioRoPE = audioPositions.map {
            LTXTransformerOps.precomputeRoPE(
                positions: $0,
                numHeads: configuration.audioNumHeads,
                headDimension: configuration.audioHeadDim,
                theta: configuration.ropeTheta,
                maxPositions: configuration.audioPositionalEmbeddingMaxPos,
                type: configuration.ropeType
            )
        }
        let crossMax = max(
            configuration.positionalEmbeddingMaxPos[0],
            configuration.audioPositionalEmbeddingMaxPos[0]
        )
        let videoCrossRoPE = videoPositions.map {
            LTXTransformerOps.precomputeRoPE(
                positions: $0[0..., 0..., ..<1],
                numHeads: configuration.avCrossNumHeads,
                headDimension: configuration.avCrossHeadDim,
                theta: configuration.ropeTheta,
                maxPositions: [crossMax],
                type: configuration.ropeType
            )
        }
        let audioCrossRoPE = audioPositions.map {
            LTXTransformerOps.precomputeRoPE(
                positions: $0[0..., 0..., ..<1],
                numHeads: configuration.avCrossNumHeads,
                headDimension: configuration.avCrossHeadDim,
                theta: configuration.ropeTheta,
                maxPositions: [crossMax],
                type: configuration.ropeType
            )
        }

        var currentVideo = videoHidden
        var currentAudio = audioHidden
        for block in transformerBlocks {
            let values = block(
                video: currentVideo,
                audio: currentAudio,
                videoAdaLN: videoTime.parameters,
                audioAdaLN: audioTime.parameters,
                videoPromptAdaLN: videoPrompt.parameters,
                audioPromptAdaLN: audioPrompt.parameters,
                avVideo: avVideo.parameters,
                avAudio: avAudio.parameters,
                avA2VGate: avA2VGate,
                avV2AGate: avV2AGate,
                videoText: videoTextEmbeds,
                audioText: audioTextEmbeds,
                videoRoPE: videoRoPE,
                audioRoPE: audioRoPE,
                videoCrossRoPE: videoCrossRoPE,
                audioCrossRoPE: audioCrossRoPE,
                videoMask: videoAttentionMask,
                audioMask: audioAttentionMask,
                videoCrossMask: videoCrossAttentionMask
            )
            currentVideo = values.video
            currentAudio = values.audio
        }
        return (
            output(currentVideo, embeddedTimestep: videoTime.embedded, table: scale_shift_table, projection: projOut),
            output(currentAudio, embeddedTimestep: audioTime.embedded, table: audio_scale_shift_table, projection: audioProjOut)
        )
    }
}

public final class LTXX0Model {
    public let transformer: LTXTransformer

    public init(transformer: LTXTransformer) {
        self.transformer = transformer
    }

    public func callAsFunction(
        videoLatent: MLXArray,
        audioLatent: MLXArray,
        sigma: MLXArray,
        videoTextEmbeds: MLXArray? = nil,
        audioTextEmbeds: MLXArray? = nil,
        videoPositions: MLXArray? = nil,
        audioPositions: MLXArray? = nil,
        videoAttentionMask: MLXArray? = nil,
        audioAttentionMask: MLXArray? = nil
    ) -> (video: MLXArray, audio: MLXArray) {
        let velocity = transformer(
            videoLatent: videoLatent,
            audioLatent: audioLatent,
            timestep: sigma,
            videoTextEmbeds: videoTextEmbeds,
            audioTextEmbeds: audioTextEmbeds,
            videoPositions: videoPositions,
            audioPositions: audioPositions,
            videoAttentionMask: videoAttentionMask,
            audioAttentionMask: audioAttentionMask
        )
        let videoSigma = sigma[0..., .newAxis, .newAxis]
        let audioSigma = sigma[0..., .newAxis, .newAxis]
        return (
            (videoLatent.asType(.float32) - videoSigma.asType(.float32) * velocity.video.asType(.float32))
                .asType(videoLatent.dtype),
            (audioLatent.asType(.float32) - audioSigma.asType(.float32) * velocity.audio.asType(.float32))
                .asType(audioLatent.dtype)
        )
    }
}
