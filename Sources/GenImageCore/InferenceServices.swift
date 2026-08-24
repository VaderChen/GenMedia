import Foundation

public struct TextToImageRequest: Sendable, Hashable {
    public var projectID: UUID
    public var recipe: GenerationRecipe
    public var profile: InferenceProfile
    public var sourceAsset: ImageAsset?

    public init(
        projectID: UUID,
        recipe: GenerationRecipe,
        profile: InferenceProfile,
        sourceAsset: ImageAsset? = nil
    ) {
        self.projectID = projectID
        self.recipe = recipe
        self.profile = profile
        self.sourceAsset = sourceAsset
    }
}

public struct ImageDescriptionRequest: Sendable, Hashable {
    public var asset: ImageAsset
    public var profile: InferenceProfile
    public var languageCode: String

    public init(
        asset: ImageAsset,
        profile: InferenceProfile,
        languageCode: String = "zh-Hant"
    ) {
        self.asset = asset
        self.profile = profile
        self.languageCode = languageCode
    }
}

public struct ImageToImageRequest: Sendable, Hashable {
    public var projectID: UUID
    public var sourceAsset: ImageAsset
    public var recipe: GenerationRecipe
    public var profile: InferenceProfile
    public var modelURL: URL
    public var quantization: ModelQuantization

    public init(
        projectID: UUID,
        sourceAsset: ImageAsset,
        recipe: GenerationRecipe,
        profile: InferenceProfile,
        modelURL: URL,
        quantization: ModelQuantization
    ) {
        self.projectID = projectID
        self.sourceAsset = sourceAsset
        self.recipe = recipe
        self.profile = profile
        self.modelURL = modelURL
        self.quantization = quantization
    }
}

public struct UpscaleRequest: Sendable, Hashable {
    public var asset: ImageAsset
    public var profile: InferenceProfile
    public var scale: Int

    public init(asset: ImageAsset, profile: InferenceProfile, scale: Int = 4) {
        self.asset = asset
        self.profile = profile
        self.scale = scale
    }
}

public struct VideoGenerationOptions: Sendable, Hashable {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var outputCount: Int
    public var frameCount: Int
    public var frameRate: Int
    public var seed: UInt64

    public init(
        prompt: String,
        width: Int,
        height: Int,
        steps: Int,
        outputCount: Int,
        frameCount: Int,
        frameRate: Int,
        seed: UInt64
    ) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.outputCount = outputCount
        self.frameCount = frameCount
        self.frameRate = frameRate
        self.seed = seed
    }

    public func validate() throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw VideoGenerationValidationError.emptyPrompt
        }
        guard (64...4096).contains(width), (64...4096).contains(height) else {
            throw VideoGenerationValidationError.invalidDimensions
        }
        guard width.isMultiple(of: 16), height.isMultiple(of: 16) else {
            throw VideoGenerationValidationError.dimensionsMustBeMultiplesOf16
        }
        guard (1...100).contains(steps) else {
            throw VideoGenerationValidationError.invalidSteps
        }
        guard (1...8).contains(outputCount) else {
            throw VideoGenerationValidationError.invalidOutputCount
        }
        guard (1...512).contains(frameCount) else {
            throw VideoGenerationValidationError.invalidFrameCount
        }
        guard (1...120).contains(frameRate) else {
            throw VideoGenerationValidationError.invalidFrameRate
        }
    }
}

public enum VideoGenerationValidationError: LocalizedError, Sendable {
    case emptyPrompt
    case invalidDimensions
    case dimensionsMustBeMultiplesOf16
    case invalidSteps
    case invalidOutputCount
    case invalidFrameCount
    case invalidFrameRate

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt: "Prompt 不可為空白。"
        case .invalidDimensions: "影片寬高必須介於 64 到 4096。"
        case .dimensionsMustBeMultiplesOf16: "影片寬高必須是 16 的倍數。"
        case .invalidSteps: "影片生成步數必須介於 1 到 100。"
        case .invalidOutputCount: "影片生成數量必須介於 1 到 8。"
        case .invalidFrameCount: "影片幀數必須介於 1 到 512。"
        case .invalidFrameRate: "影片幀率必須介於 1 到 120 FPS。"
        }
    }
}

public struct MusicGenerationOptions: Sendable, Hashable {
    public static let supportedDurationSeconds = 5...300

    public var prompt: String
    public var lyrics: String
    public var durationSeconds: Int
    public var steps: Int
    public var seed: UInt64
    public var format: AudioOutputFormat

    public init(
        prompt: String,
        lyrics: String,
        durationSeconds: Int,
        steps: Int,
        seed: UInt64,
        format: AudioOutputFormat
    ) {
        self.prompt = prompt
        self.lyrics = lyrics
        self.durationSeconds = durationSeconds
        self.steps = steps
        self.seed = seed
        self.format = format
    }

    public func validate() throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw MusicGenerationValidationError.emptyPrompt
        }
        guard Self.supportedDurationSeconds.contains(durationSeconds) else {
            throw MusicGenerationValidationError.invalidDuration
        }
        guard (1...100).contains(steps) else {
            throw MusicGenerationValidationError.invalidSteps
        }
    }
}

public enum MusicGenerationValidationError: LocalizedError, Sendable {
    case emptyPrompt
    case invalidDuration
    case invalidSteps

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt: "音樂風格 Prompt 不可為空白。"
        case .invalidDuration: "音樂長度必須介於 5 到 300 秒。"
        case .invalidSteps: "音樂生成步數必須介於 1 到 100。"
        }
    }
}

public struct MusicGenerationRequest: Sendable, Hashable {
    public var projectID: UUID
    public var recipeID: UUID
    public var options: MusicGenerationOptions
    public var profile: InferenceProfile
    public var modelURL: URL

    public init(
        projectID: UUID,
        recipeID: UUID,
        options: MusicGenerationOptions,
        profile: InferenceProfile,
        modelURL: URL
    ) {
        self.projectID = projectID
        self.recipeID = recipeID
        self.options = options
        self.profile = profile
        self.modelURL = modelURL
    }
}

public struct VideoGenerationRequest: Sendable, Hashable {
    public var projectID: UUID
    public var recipeID: UUID
    public var sourceAssets: [ImageAsset]
    public var sourceAsset: ImageAsset? { sourceAssets.first }
    public var options: VideoGenerationOptions
    public var profile: InferenceProfile
    public var modelURL: URL
    public var profileLoRAs: [VideoGenerationLoRA]

    public init(
        projectID: UUID,
        recipeID: UUID,
        sourceAsset: ImageAsset?,
        sourceAssets: [ImageAsset]? = nil,
        options: VideoGenerationOptions,
        profile: InferenceProfile,
        modelURL: URL,
        profileLoRAs: [VideoGenerationLoRA] = []
    ) {
        self.projectID = projectID
        self.recipeID = recipeID
        self.sourceAssets = sourceAssets ?? (sourceAsset.map { [$0] } ?? [])
        self.options = options
        self.profile = profile
        self.modelURL = modelURL
        self.profileLoRAs = profileLoRAs
    }
}

public struct VideoGenerationLoRA: Sendable, Hashable {
    public var modelID: String
    public var localURL: URL
    public var scale: Double
    public var conditioning: ProfileLoRAConditioning?
    public var conditioningScale: Double

    public init(configuration: ProfileLoRAConfiguration, localURL: URL) {
        modelID = configuration.modelID
        self.localURL = localURL
        scale = configuration.scale
        conditioning = configuration.conditioning
        conditioningScale = configuration.conditioningScale
    }
}

public protocol TextToImageGenerating: Sendable {
    func generate(
        request: TextToImageRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> [ImageAsset]
}

public protocol ImageDescribing: Sendable {
    func describe(
        request: ImageDescriptionRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> String
}

public protocol ImageToImageGenerating: Sendable {
    func generate(
        request: ImageToImageRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ImageAsset
}

public protocol ImageUpscaling: Sendable {
    func upscale(
        request: UpscaleRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ImageAsset
}

public protocol VideoGenerating: Sendable {
    func generate(
        request: VideoGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> [ImageAsset]
}

public protocol MusicGenerating: Sendable {
    func generate(
        request: MusicGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ImageAsset
}

public struct InferenceServices: Sendable {
    public var textToImage: any TextToImageGenerating
    public var imageToText: any ImageDescribing
    public var imageToImage: any ImageToImageGenerating
    public var upscale: any ImageUpscaling
    public var video: (any VideoGenerating)?
    public var music: (any MusicGenerating)?

    public init(
        textToImage: any TextToImageGenerating,
        imageToText: any ImageDescribing,
        imageToImage: any ImageToImageGenerating,
        upscale: any ImageUpscaling,
        video: (any VideoGenerating)? = nil,
        music: (any MusicGenerating)? = nil
    ) {
        self.textToImage = textToImage
        self.imageToText = imageToText
        self.imageToImage = imageToImage
        self.upscale = upscale
        self.video = video
        self.music = music
    }
}
