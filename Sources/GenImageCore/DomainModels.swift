import Foundation

public enum ModelCapability: String, CaseIterable, Codable, Hashable, Sendable, Identifiable {
    case textToImage
    case imageToText
    case upscale
    case imageToImage
    case imageToVideo
    case textToVideo
    case textToMusic
    case controlNet
    case lora

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .textToImage: "文生圖"
        case .imageToText: "圖生文"
        case .upscale: "Upscale"
        case .imageToImage: "圖生圖"
        case .imageToVideo: "圖生影"
        case .textToVideo: "文生影"
        case .textToMusic: "文生音樂"
        case .controlNet: "ControlNet"
        case .lora: "LoRA"
        }
    }

    public var symbolName: String {
        switch self {
        case .textToImage: "text.below.photo"
        case .imageToText: "photo.badge.magnifyingglass"
        case .upscale: "arrow.up.left.and.arrow.down.right"
        case .imageToImage: "photo.on.rectangle.angled"
        case .imageToVideo: "photo.badge.play"
        case .textToVideo: "text.badge.play"
        case .textToMusic: "music.note"
        case .controlNet: "point.3.connected.trianglepath.dotted"
        case .lora: "slider.horizontal.3"
        }
    }
}

public enum ModelQuantization: String, CaseIterable, Codable, Hashable, Sendable {
    case twoBit = "2-bit"
    case bf16 = "BF16"
    case fp16 = "FP16"
    case eightBit = "8-bit"
    case fourBit = "4-bit"
    case coreML = "Core ML"
    case lora = "LoRA"
}

public enum InferenceArchitecture: String, CaseIterable, Codable, Hashable, Sendable, Identifiable {
    case mlxSwift
    case coreML
    case localService
    case externalCLI

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .mlxSwift: "MLX Swift"
        case .coreML: "Core ML"
        case .localService: "本機服務"
        case .externalCLI: "外部 CLI"
        }
    }
}

public enum ProfileLoRAConditioning: String, CaseIterable, Codable, Hashable, Sendable {
    case sourceImageCanny
}

public struct ProfileLoRAConfiguration: Codable, Hashable, Sendable {
    public var modelID: String
    public var scale: Double
    public var conditioning: ProfileLoRAConditioning?
    public var conditioningScale: Double

    public init(
        modelID: String,
        scale: Double = 1,
        conditioning: ProfileLoRAConditioning? = nil,
        conditioningScale: Double = 1
    ) {
        self.modelID = modelID
        self.scale = scale
        self.conditioning = conditioning
        self.conditioningScale = conditioningScale
    }
}

public struct ProfileDefaults: Codable, Hashable, Sendable {
    public var width: Int?
    public var height: Int?
    public var steps: Int?
    public var outputCount: Int?
    public var maxTokens: Int?
    public var languageCode: String?
    public var upscaleScale: Int?
    public var tileSize: Int?
    public var frameCount: Int?
    public var frameRate: Int?
    public var durationSeconds: Int?

    public init(
        width: Int? = nil,
        height: Int? = nil,
        steps: Int? = nil,
        outputCount: Int? = nil,
        maxTokens: Int? = nil,
        languageCode: String? = nil,
        upscaleScale: Int? = nil,
        tileSize: Int? = nil,
        frameCount: Int? = nil,
        frameRate: Int? = nil,
        durationSeconds: Int? = nil
    ) {
        self.width = width
        self.height = height
        self.steps = steps
        self.outputCount = outputCount
        self.maxTokens = maxTokens
        self.languageCode = languageCode
        self.upscaleScale = upscaleScale
        self.tileSize = tileSize
        self.frameCount = frameCount
        self.frameRate = frameRate
        self.durationSeconds = durationSeconds
    }
}

public enum MusicDurationSemantics: String, Codable, Hashable, Sendable {
    case target
    case maximum
}

public struct ProfileMusicConfiguration: Codable, Hashable, Sendable {
    public var minimumDurationSeconds: Int
    public var maximumDurationSeconds: Int
    public var durationSemantics: MusicDurationSemantics

    public init(
        minimumDurationSeconds: Int = 5,
        maximumDurationSeconds: Int = 300,
        durationSemantics: MusicDurationSemantics = .target
    ) {
        self.minimumDurationSeconds = minimumDurationSeconds
        self.maximumDurationSeconds = maximumDurationSeconds
        self.durationSemantics = durationSemantics
    }
}

public struct InferenceProfile: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var name: String
    public var capability: ModelCapability
    public var modelID: String
    public var modelRevision: String
    public var architecture: InferenceArchitecture
    public var defaults: ProfileDefaults
    public var music: ProfileMusicConfiguration?
    public var loras: [ProfileLoRAConfiguration]
    public var profileRevision: Int
    public var notes: String
    public var isBuiltIn: Bool

    public init(
        id: UUID = UUID(),
        name: String,
        capability: ModelCapability,
        modelID: String,
        modelRevision: String = "main",
        architecture: InferenceArchitecture,
        defaults: ProfileDefaults = ProfileDefaults(),
        music: ProfileMusicConfiguration? = nil,
        loras: [ProfileLoRAConfiguration] = [],
        profileRevision: Int = 1,
        notes: String = "",
        isBuiltIn: Bool = false
    ) {
        self.id = id
        self.name = name
        self.capability = capability
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.architecture = architecture
        self.defaults = defaults
        self.music = music
        self.loras = loras
        self.profileRevision = profileRevision
        self.notes = notes
        self.isBuiltIn = isBuiltIn
    }

    public var requiredModelIDs: [String] {
        ([modelID] + loras.map(\.modelID)).reduce(into: []) { result, modelID in
            if !result.contains(modelID) { result.append(modelID) }
        }
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case capability
        case modelID
        case modelRevision
        case architecture
        case defaults
        case music
        case loras
        case profileRevision
        case notes
        case isBuiltIn
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        name = try container.decode(String.self, forKey: .name)
        capability = try container.decode(ModelCapability.self, forKey: .capability)
        modelID = try container.decode(String.self, forKey: .modelID)
        modelRevision = try container.decodeIfPresent(String.self, forKey: .modelRevision) ?? "main"
        architecture = try container.decode(InferenceArchitecture.self, forKey: .architecture)
        defaults = try container.decodeIfPresent(ProfileDefaults.self, forKey: .defaults)
            ?? ProfileDefaults()
        music = try container.decodeIfPresent(ProfileMusicConfiguration.self, forKey: .music)
        loras = try container.decodeIfPresent([ProfileLoRAConfiguration].self, forKey: .loras) ?? []
        profileRevision = try container.decodeIfPresent(Int.self, forKey: .profileRevision) ?? 1
        notes = try container.decodeIfPresent(String.self, forKey: .notes) ?? ""
        isBuiltIn = try container.decodeIfPresent(Bool.self, forKey: .isBuiltIn) ?? false
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(capability, forKey: .capability)
        try container.encode(modelID, forKey: .modelID)
        try container.encode(modelRevision, forKey: .modelRevision)
        try container.encode(architecture, forKey: .architecture)
        try container.encode(defaults, forKey: .defaults)
        try container.encodeIfPresent(music, forKey: .music)
        try container.encode(loras, forKey: .loras)
        try container.encode(profileRevision, forKey: .profileRevision)
        try container.encode(notes, forKey: .notes)
        try container.encode(isBuiltIn, forKey: .isBuiltIn)
    }
}

public struct ModelDescriptor: Identifiable, Codable, Hashable, Sendable {
    public let id: String
    public var displayName: String
    public var publisher: String
    public var summary: String
    public var capabilities: Set<ModelCapability>
    public var quantization: ModelQuantization
    public var approximateDownloadGB: Double
    public var recommendedMemoryGB: Int
    public var licenseName: String
    public var sourceURL: URL?
    public var localURL: URL?
    public var isRecommended: Bool

    public init(
        id: String,
        displayName: String,
        publisher: String,
        summary: String,
        capabilities: Set<ModelCapability>,
        quantization: ModelQuantization,
        approximateDownloadGB: Double,
        recommendedMemoryGB: Int,
        licenseName: String,
        sourceURL: URL? = nil,
        localURL: URL? = nil,
        isRecommended: Bool = false
    ) {
        self.id = id
        self.displayName = displayName
        self.publisher = publisher
        self.summary = summary
        self.capabilities = capabilities
        self.quantization = quantization
        self.approximateDownloadGB = approximateDownloadGB
        self.recommendedMemoryGB = recommendedMemoryGB
        self.licenseName = licenseName
        self.sourceURL = sourceURL
        self.localURL = localURL
        self.isRecommended = isRecommended
    }
}

public struct LoRADescriptor: Identifiable, Codable, Hashable, Sendable {
    public let id: String
    public var displayName: String
    public var localURL: URL
    public var fileSizeMB: Double
    public var compatibleCapabilities: Set<ModelCapability>

    public init(
        id: String,
        displayName: String,
        localURL: URL,
        fileSizeMB: Double,
        compatibleCapabilities: Set<ModelCapability> = [.textToImage]
    ) {
        self.id = id
        self.displayName = displayName
        self.localURL = localURL
        self.fileSizeMB = fileSizeMB
        self.compatibleCapabilities = compatibleCapabilities
    }
}

public struct LoRASelection: Codable, Hashable, Sendable {
    public var adapterID: String
    public var localURL: URL
    public var scale: Double

    public init(adapterID: String, localURL: URL, scale: Double = 1) {
        self.adapterID = adapterID
        self.localURL = localURL
        self.scale = scale
    }
}

public enum InstallationPhase: String, Codable, Hashable, Sendable {
    case notInstalled
    case queued
    case downloading
    case paused
    case verifying
    case installed
    case failed
}

public struct ModelInstallation: Codable, Hashable, Sendable {
    public var phase: InstallationPhase
    public var progress: Double
    public var downloadedGB: Double
    public var errorMessage: String?

    public init(
        phase: InstallationPhase = .notInstalled,
        progress: Double = 0,
        downloadedGB: Double = 0,
        errorMessage: String? = nil
    ) {
        self.phase = phase
        self.progress = progress
        self.downloadedGB = downloadedGB
        self.errorMessage = errorMessage
    }
}

public struct GenerationRecipe: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var name: String
    public var prompt: String
    public var negativePrompt: String
    public var modelID: String
    public var profileID: UUID?
    public var width: Int
    public var height: Int
    public var steps: Int
    public var outputCount: Int
    public var seed: UInt64
    public var lora: LoRASelection?

    public init(
        id: UUID = UUID(),
        name: String = "未命名配方",
        prompt: String = "",
        negativePrompt: String = "",
        modelID: String,
        profileID: UUID? = nil,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 9,
        outputCount: Int = 4,
        seed: UInt64 = 42,
        lora: LoRASelection? = nil
    ) {
        self.id = id
        self.name = name
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.modelID = modelID
        self.profileID = profileID
        self.width = width
        self.height = height
        self.steps = steps
        self.outputCount = outputCount
        self.seed = seed
        self.lora = lora
    }

    public mutating func applySizePreset(width: Int, height: Int) {
        self.width = width
        self.height = height
    }

    public func validate() throws {
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw RecipeValidationError.emptyPrompt
        }
        guard (64...4096).contains(width), (64...4096).contains(height) else {
            throw RecipeValidationError.invalidDimensions
        }
        guard width.isMultiple(of: 16), height.isMultiple(of: 16) else {
            throw RecipeValidationError.dimensionsMustBeMultiplesOf16
        }
        guard (1...100).contains(steps) else {
            throw RecipeValidationError.invalidSteps
        }
        guard (1...8).contains(outputCount) else {
            throw RecipeValidationError.invalidOutputCount
        }
        if let lora, !lora.scale.isFinite || !(0...1).contains(lora.scale) {
            throw RecipeValidationError.invalidLoRAScale
        }
    }
}

public enum RecipeValidationError: LocalizedError, Sendable {
    case emptyPrompt
    case invalidDimensions
    case dimensionsMustBeMultiplesOf16
    case invalidSteps
    case invalidOutputCount
    case invalidLoRAScale

    public var errorDescription: String? {
        switch self {
        case .emptyPrompt: "Prompt 不可為空白。"
        case .invalidDimensions: "寬高必須介於 64 到 4096。"
        case .dimensionsMustBeMultiplesOf16: "寬高必須是 16 的倍數。"
        case .invalidSteps: "步數必須介於 1 到 100。"
        case .invalidOutputCount: "生成數量必須介於 1 到 8。"
        case .invalidLoRAScale: "LoRA 權重必須介於 0 到 1。"
        }
    }
}

public enum AssetKind: String, Codable, Hashable, Sendable {
    case imported
    case generated
    case generatedVideo
    case generatedAudio
    case edited
    case upscaled

    public var isImage: Bool {
        switch self {
        case .imported, .generated, .edited, .upscaled:
            true
        case .generatedVideo, .generatedAudio:
            false
        }
    }
}

public enum AudioOutputFormat: String, CaseIterable, Codable, Hashable, Sendable, Identifiable {
    case mp3
    case m4a
    case aac
    case flac

    public var id: String { rawValue }

    public var fileExtension: String { rawValue }

    public var displayName: String { rawValue.uppercased() }
}

public enum MusicStyle: String, CaseIterable, Codable, Hashable, Sendable, Identifiable {
    case pop
    case rock
    case hipHop
    case rnb
    case electronic
    case jazz
    case classical
    case folk
    case country
    case blues
    case reggae
    case metal
    case ambient
    case cinematic
    case anime
    case lofi

    public var id: String { rawValue }

    public var prompt: String {
        switch self {
        case .pop: "Pop"
        case .rock: "Rock"
        case .hipHop: "Hip-hop"
        case .rnb: "R&B"
        case .electronic: "Electronic dance music"
        case .jazz: "Jazz"
        case .classical: "Classical"
        case .folk: "Folk"
        case .country: "Country"
        case .blues: "Blues"
        case .reggae: "Reggae"
        case .metal: "Metal"
        case .ambient: "Ambient"
        case .cinematic: "Cinematic soundtrack"
        case .anime: "Anime soundtrack"
        case .lofi: "Lo-fi"
        }
    }
}

public struct ImageAsset: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var projectID: UUID
    public var parentAssetID: UUID?
    public var operationID: UUID?
    public var kind: AssetKind
    public var title: String
    public var fileURL: URL?
    public var pixelWidth: Int
    public var pixelHeight: Int
    public var mediaDurationSeconds: Double?
    public var sampleRate: Int?
    public var channelCount: Int?
    public var audioFormat: AudioOutputFormat?
    public var recipeID: UUID?
    public var createdAt: Date

    public init(
        id: UUID = UUID(),
        projectID: UUID,
        parentAssetID: UUID? = nil,
        operationID: UUID? = nil,
        kind: AssetKind,
        title: String,
        fileURL: URL? = nil,
        pixelWidth: Int,
        pixelHeight: Int,
        mediaDurationSeconds: Double? = nil,
        sampleRate: Int? = nil,
        channelCount: Int? = nil,
        audioFormat: AudioOutputFormat? = nil,
        recipeID: UUID? = nil,
        createdAt: Date = .now
    ) {
        self.id = id
        self.projectID = projectID
        self.parentAssetID = parentAssetID
        self.operationID = operationID
        self.kind = kind
        self.title = title
        self.fileURL = fileURL
        self.pixelWidth = pixelWidth
        self.pixelHeight = pixelHeight
        self.mediaDurationSeconds = mediaDurationSeconds
        self.sampleRate = sampleRate
        self.channelCount = channelCount
        self.audioFormat = audioFormat
        self.recipeID = recipeID
        self.createdAt = createdAt
    }
}

public enum WorkflowAction: String, Codable, Hashable, Sendable {
    case importImage
    case describe
    case generate
    case generateVideo
    case generateMusic
    case imageToImage
    case upscale

    public var title: String {
        switch self {
        case .importImage: "匯入"
        case .describe: "圖生文"
        case .generate: "文生圖"
        case .generateVideo: "生成影片"
        case .generateMusic: "生成音樂"
        case .imageToImage: "圖生圖"
        case .upscale: "Upscale"
        }
    }

    public var symbolName: String {
        switch self {
        case .importImage: "square.and.arrow.down"
        case .describe: "text.viewfinder"
        case .generate: "sparkles"
        case .generateVideo: "play.rectangle"
        case .generateMusic: "music.note"
        case .imageToImage: "photo.on.rectangle.angled"
        case .upscale: "arrow.up.left.and.arrow.down.right"
        }
    }
}

public struct WorkflowOperation: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var projectID: UUID
    public var action: WorkflowAction
    public var inputAssetID: UUID?
    public var outputAssetIDs: [UUID]
    public var recipeID: UUID?
    public var profileSnapshot: InferenceProfile?
    public var createdAt: Date

    public init(
        id: UUID = UUID(),
        projectID: UUID,
        action: WorkflowAction,
        inputAssetID: UUID? = nil,
        outputAssetIDs: [UUID] = [],
        recipeID: UUID? = nil,
        profileSnapshot: InferenceProfile? = nil,
        createdAt: Date = .now
    ) {
        self.id = id
        self.projectID = projectID
        self.action = action
        self.inputAssetID = inputAssetID
        self.outputAssetIDs = outputAssetIDs
        self.recipeID = recipeID
        self.profileSnapshot = profileSnapshot
        self.createdAt = createdAt
    }
}

public enum JobState: String, Codable, Hashable, Sendable {
    case queued
    case running
    case cancelling
    case completed
    case cancelled
    case failed
}

public struct GenerationJob: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var action: WorkflowAction
    public var title: String
    public var state: JobState
    public var progress: Double
    public var createdAt: Date
    public var startedAt: Date?
    public var finishedAt: Date?
    public var errorMessage: String?

    public init(
        id: UUID = UUID(),
        action: WorkflowAction,
        title: String,
        state: JobState = .queued,
        progress: Double = 0,
        createdAt: Date = .now,
        startedAt: Date? = nil,
        finishedAt: Date? = nil,
        errorMessage: String? = nil
    ) {
        self.id = id
        self.action = action
        self.title = title
        self.state = state
        self.progress = progress
        self.createdAt = createdAt
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.errorMessage = errorMessage
    }
}

public struct Project: Identifiable, Codable, Hashable, Sendable {
    public let id: UUID
    public var name: String
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        name: String,
        createdAt: Date = .now,
        updatedAt: Date = .now
    ) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}
