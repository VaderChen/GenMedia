import AppKit
import Combine
import Foundation
import GenImageCore
import GenImageRuntime

enum AppSection: String, CaseIterable, Identifiable {
    case workspace
    case models
    case profiles

    var id: String { rawValue }

    var title: String {
        switch self {
        case .workspace: "工作區"
        case .models: "模型中心"
        case .profiles: "Profiles"
        }
    }

    var symbolName: String {
        switch self {
        case .workspace: "rectangle.3.group"
        case .models: "shippingbox"
        case .profiles: "switch.2"
        }
    }
}

enum PreviewMode: String, CaseIterable, Identifiable {
    case grid
    case single
    case compare

    var id: String { rawValue }

    var title: String {
        switch self {
        case .grid: "網格"
        case .single: "單張"
        case .compare: "比較"
        }
    }

    var symbolName: String {
        switch self {
        case .grid: "square.grid.2x2"
        case .single: "rectangle"
        case .compare: "rectangle.split.2x1"
        }
    }
}

enum ModelFilter: String, CaseIterable, Identifiable {
    case all
    case textToImage
    case imageToText
    case imageToVideo
    case textToVideo
    case textToMusic
    case upscale
    case lora

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: "全部"
        case .textToImage: "文生圖"
        case .imageToText: "圖生文"
        case .imageToVideo: "圖生影"
        case .textToVideo: "文生影"
        case .textToMusic: "文生音樂"
        case .upscale: "Upscale"
        case .lora: "LoRA"
        }
    }

    var capability: ModelCapability? {
        switch self {
        case .all: nil
        case .textToImage: .textToImage
        case .imageToText: .imageToText
        case .imageToVideo: .imageToVideo
        case .textToVideo: .textToVideo
        case .textToMusic: .textToMusic
        case .upscale: .upscale
        case .lora: .lora
        }
    }
}

struct VideoOutputSettings: Codable, Hashable, Sendable {
    var width: Int
    var height: Int
    var steps: Int
    var outputCount: Int
    var frameCount: Int
    var frameRate: Int
    var seed: UInt64

    init(
        width: Int = 1280,
        height: Int = 720,
        steps: Int = 8,
        outputCount: Int = 1,
        frameCount: Int = 97,
        frameRate: Int = 24,
        seed: UInt64 = 42
    ) {
        self.width = width
        self.height = height
        self.steps = steps
        self.outputCount = outputCount
        self.frameCount = frameCount
        self.frameRate = frameRate
        self.seed = seed
    }
}

struct MusicOutputSettings: Codable, Hashable, Sendable {
    var prompt: String
    var lyrics: String
    var style: MusicStyle
    var durationSeconds: Int
    var steps: Int
    var seed: UInt64
    var format: AudioOutputFormat

    private enum CodingKeys: String, CodingKey {
        case prompt
        case lyrics
        case style
        case durationSeconds
        case steps
        case seed
        case format
    }

    init(
        prompt: String = "",
        lyrics: String = "",
        style: MusicStyle = .pop,
        durationSeconds: Int = 10,
        steps: Int = 30,
        seed: UInt64 = 42,
        format: AudioOutputFormat = .mp3
    ) {
        self.prompt = prompt
        self.lyrics = lyrics
        self.style = style
        self.durationSeconds = durationSeconds
        self.steps = steps
        self.seed = seed
        self.format = format
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            prompt: try container.decodeIfPresent(String.self, forKey: .prompt) ?? "",
            lyrics: try container.decodeIfPresent(String.self, forKey: .lyrics) ?? "",
            style: try container.decodeIfPresent(MusicStyle.self, forKey: .style) ?? .pop,
            durationSeconds: try container.decodeIfPresent(Int.self, forKey: .durationSeconds) ?? 10,
            steps: try container.decodeIfPresent(Int.self, forKey: .steps) ?? 30,
            seed: try container.decodeIfPresent(UInt64.self, forKey: .seed) ?? 42,
            format: try container.decodeIfPresent(AudioOutputFormat.self, forKey: .format) ?? .mp3
        )
    }
}

@MainActor
final class AppStore: ObservableObject {
    @Published var selectedSection: AppSection = .workspace
    @Published var projects: [Project] {
        didSet { persistProjectWorkspace() }
    }
    @Published var selectedProjectID: UUID {
        didSet { persistProjectWorkspace() }
    }
    @Published var assets: [ImageAsset] {
        didSet { persistProjectWorkspace() }
    }
    @Published var operations: [WorkflowOperation] {
        didSet { persistProjectWorkspace() }
    }
    @Published var selectedAssetID: UUID? {
        didSet { persistProjectWorkspace() }
    }
    @Published var comparisonAssetID: UUID? {
        didSet { persistProjectWorkspace() }
    }
    @Published var recipe: GenerationRecipe {
        didSet {
            Self.persistRecipeSettings(recipe)
        }
    }
    @Published var videoOutputSettings: VideoOutputSettings {
        didSet {
            Self.persistVideoOutputSettings(videoOutputSettings)
        }
    }
    @Published var musicOutputSettings: MusicOutputSettings {
        didSet {
            Self.persistMusicOutputSettings(musicOutputSettings)
        }
    }
    @Published var jobs: [GenerationJob] = []
    @Published var modelRootPath: String
    @Published var outputDirectoryPath: String
    @Published var models: [ModelDescriptor]
    @Published var loras: [LoRADescriptor]
    @Published var installations: [String: ModelInstallation]
    @Published var profiles: [InferenceProfile]
    @Published var disabledProfileIDs: Set<UUID>
    @Published var activeProfileIDs: [ModelCapability: UUID]
    @Published var previewMode: PreviewMode = .grid
    @Published var previewZoom: Double = 1
    @Published var isInspectorVisible = true
    @Published var modelSearch = ""
    @Published var modelFilter: ModelFilter = .all
    @Published var statusMessage: String?
    @Published var systemMetrics: SystemMetricsSnapshot = .unavailable
    @Published var availableUpdate: AppUpdateInfo?
    @Published var isReleasingMemory = false

    private var jobTasks: [UUID: Task<Void, Never>] = [:]
    private var cancellationRequestedJobIDs: Set<UUID> = []
    private var lastJobProgressUpdate: [UUID: Date] = [:]
    private var modelTasks: [String: Task<Void, Never>] = [:]
    private var modelTaskTokens: [String: UUID] = [:]
    private var systemMetricsTask: Task<Void, Never>?
    private var updateCheckTask: Task<Void, Never>?
    private var textToImageService: ZImageTextToImageService
    private let imageToTextService: QwenVLImageDescriptionService
    private var imageToImageService: Qwen2511ImageToImageService
    private var upscaleService: CoreMLUpscaleService
    private var videoGenerationService: LTXVideoGenerationService
    private var musicGenerationService: MusicGenerationRouter
    private let modelInstaller = HuggingFaceModelInstaller()
    private let projectWorkspaceURL: URL
    private var projectWorkspacePersistenceEnabled = false

    private static let recipeSettingsKey = "GenImage.recipeSettings.v1"
    private static let videoOutputSettingsKey = "GenImage.videoOutputSettings.v1"
    private static let musicOutputSettingsKey = "GenImage.musicOutputSettings.v1"
    private static let modelRootKey = "GenImage.modelRootPath.v1"
    private static let outputDirectoryKey = "GenImage.outputDirectoryPath.v1"
    private static let disabledProfilesKey = "GenImage.disabledProfiles.v1"
    private static let activeProfilesKey = "GenImage.activeProfiles.v1"
    private static let jobProgressUpdateInterval: TimeInterval = 1
    private static let modelProgressUpdateInterval: TimeInterval = 0.5

    private struct PersistedRecipeSettings: Codable {
        var prompt: String
        var negativePrompt: String
        var width: Int
        var height: Int
        var steps: Int
        var outputCount: Int
        var seed: UInt64
        var lora: LoRASelection?

        init(recipe: GenerationRecipe) {
            prompt = recipe.prompt
            negativePrompt = recipe.negativePrompt
            width = recipe.width
            height = recipe.height
            steps = recipe.steps
            outputCount = recipe.outputCount
            seed = recipe.seed
            lora = recipe.lora
        }
    }

    init() {
        projectWorkspaceURL = ProjectWorkspacePersistence.defaultURL()
        let restoredWorkspace = try? ProjectWorkspacePersistence.load(from: projectWorkspaceURL)
        let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.temporaryDirectory
        let defaultGeneratedDirectory = applicationSupport
            .appendingPathComponent("GenImage", isDirectory: true)
            .appendingPathComponent("Generated", isDirectory: true)
        let configuredOutputDirectory = ProcessInfo.processInfo.environment["GENIMAGE_OUTPUT_DIRECTORY"]
            ?? UserDefaults.standard.string(forKey: Self.outputDirectoryKey)
        let generatedDirectory: URL
        if let configuredOutputDirectory,
           !configuredOutputDirectory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           NSString(string: configuredOutputDirectory).expandingTildeInPath.hasPrefix("/") {
            generatedDirectory = URL(
                fileURLWithPath: (configuredOutputDirectory as NSString).expandingTildeInPath,
                isDirectory: true
            ).standardizedFileURL
        } else {
            generatedDirectory = defaultGeneratedDirectory
        }
        try? FileManager.default.createDirectory(
            at: generatedDirectory,
            withIntermediateDirectories: true
        )
        textToImageService = ZImageTextToImageService(outputDirectory: generatedDirectory)
        imageToTextService = QwenVLImageDescriptionService()
        imageToImageService = Qwen2511ImageToImageService(outputDirectory: generatedDirectory)
        upscaleService = CoreMLUpscaleService(
            outputDirectory: generatedDirectory
        )
        videoGenerationService = LTXVideoGenerationService(outputDirectory: generatedDirectory)
        musicGenerationService = Self.makeMusicGenerationService(outputDirectory: generatedDirectory)
        let fallbackProject = Project(name: "示範專案")
        let initialProjects = restoredWorkspace?.projects.isEmpty == false
            ? restoredWorkspace!.projects
            : [fallbackProject]
        let initialProjectIDs = Set(initialProjects.map(\.id))
        let initialSelectedProjectID = restoredWorkspace.map(\.selectedProjectID)
            .flatMap { initialProjectIDs.contains($0) ? $0 : nil }
            ?? initialProjects[0].id
        let configuredModelRoot = ProcessInfo.processInfo.environment["GENIMAGE_MODEL_ROOT"]
            ?? UserDefaults.standard.string(forKey: Self.modelRootKey)
        let modelRootURL = ModelStorage.rootURL(explicitPath: configuredModelRoot)
        let discovered = LocalModelDiscovery.discover(
            at: modelRootURL
        )
        let catalog = Self.mergedModels(discovered: discovered)
        let profileCatalog = Self.mergedProfiles(discovered: discovered)
        let savedRecipeSettings = Self.loadRecipeSettings()
        let savedVideoOutputSettings = Self.loadVideoOutputSettings()
        let savedMusicOutputSettings = Self.loadMusicOutputSettings()
        let disabledProfileIDs = Self.disabledProfileIDs(in: profileCatalog)
        let initialActiveProfileIDs = Self.persistedActiveProfileIDs(
            in: profileCatalog,
            models: catalog
        )
        let generationProfile = initialActiveProfileIDs[.textToImage]
            .flatMap { activeID in profileCatalog.first { $0.id == activeID } }
            ?? profileCatalog.first { $0.capability == .textToImage }!
        let videoProfile = [ModelCapability.textToVideo, .imageToVideo]
            .compactMap { capability in
                initialActiveProfileIDs[capability].flatMap { activeID in
                    profileCatalog.first { $0.id == activeID }
                }
            }
            .first
            ?? profileCatalog.first { $0.capability == .textToVideo }
            ?? profileCatalog.first { $0.capability == .imageToVideo }
        let musicProfile = initialActiveProfileIDs[.textToMusic]
            .flatMap { activeID in profileCatalog.first { $0.id == activeID } }
            ?? profileCatalog.first { $0.capability == .textToMusic }

        projects = initialProjects
        selectedProjectID = initialSelectedProjectID
        modelRootPath = modelRootURL.path
        outputDirectoryPath = generatedDirectory.path
        models = catalog
        loras = discovered.loras
        profiles = profileCatalog
        self.disabledProfileIDs = disabledProfileIDs
        activeProfileIDs = initialActiveProfileIDs
        installations = Self.installations(for: catalog)
        let initialRecipe = GenerationRecipe(
            name: "暖色城市夜景",
            prompt: savedRecipeSettings?.prompt ?? "",
            negativePrompt: savedRecipeSettings?.negativePrompt ?? "模糊、低解析度、過度銳化",
            modelID: generationProfile.modelID,
            profileID: initialActiveProfileIDs[.textToImage],
            width: Self.persistedDimension(
                savedRecipeSettings?.width,
                fallback: generationProfile.defaults.width ?? 1024
            ),
            height: Self.persistedDimension(
                savedRecipeSettings?.height,
                fallback: generationProfile.defaults.height ?? 1024
            ),
            steps: Self.persistedValue(
                savedRecipeSettings?.steps,
                range: 1...100,
                fallback: generationProfile.defaults.steps ?? 9
            ),
            outputCount: Self.persistedValue(
                savedRecipeSettings?.outputCount,
                range: 1...8,
                fallback: generationProfile.defaults.outputCount ?? 4
            ),
            seed: savedRecipeSettings?.seed ?? UInt64.random(in: 0...UInt64.max),
            lora: Self.validatedPersistedLoRA(savedRecipeSettings?.lora, available: discovered.loras)
        )
        recipe = initialRecipe
        let initialVideoOutputSettings = Self.validatedVideoOutputSettings(
            savedVideoOutputSettings,
            defaults: videoProfile?.defaults ?? ProfileDefaults()
        )
        videoOutputSettings = initialVideoOutputSettings
        let initialMusicOutputSettings = Self.validatedMusicOutputSettings(
            savedMusicOutputSettings,
            defaults: musicProfile?.defaults ?? ProfileDefaults()
        )
        musicOutputSettings = initialMusicOutputSettings
        Self.persistRecipeSettings(initialRecipe)
        Self.persistVideoOutputSettings(initialVideoOutputSettings)
        Self.persistMusicOutputSettings(initialMusicOutputSettings)
        UserDefaults.standard.set(generatedDirectory.path, forKey: Self.outputDirectoryKey)

        let restoredAssets = (restoredWorkspace?.assets ?? []).filter {
            initialProjectIDs.contains($0.projectID)
        }
        let restoredAssetIDs = Set(restoredAssets.map(\.id))
        assets = restoredAssets
        selectedAssetID = restoredWorkspace?.selectedAssetID.flatMap {
            restoredAssetIDs.contains($0) ? $0 : nil
        }
        comparisonAssetID = restoredWorkspace?.comparisonAssetID.flatMap {
            restoredAssetIDs.contains($0) ? $0 : nil
        }
        operations = (restoredWorkspace?.operations ?? []).filter {
            initialProjectIDs.contains($0.projectID)
        }
        projectWorkspacePersistenceEnabled = true
        persistProjectWorkspace()
        startSystemMetricsUpdates()
        checkForUpdates()
    }

    private func persistProjectWorkspace() {
        guard projectWorkspacePersistenceEnabled else { return }
        let snapshot = ProjectWorkspaceSnapshot(
            projects: projects,
            selectedProjectID: selectedProjectID,
            assets: assets,
            operations: operations,
            selectedAssetID: selectedAssetID,
            comparisonAssetID: comparisonAssetID
        )
        do {
            try ProjectWorkspacePersistence.save(snapshot, to: projectWorkspaceURL)
        } catch {
            statusMessage = "無法保存開啟中的生成專案：\(error.localizedDescription)"
        }
    }

    private static func makeMusicGenerationService(
        outputDirectory: URL
    ) -> MusicGenerationRouter {
        MusicGenerationRouter(
            adapters: [
                MiniMaxMusic3GenerationService(outputDirectory: outputDirectory),
                ACEStepMusicGenerationService(outputDirectory: outputDirectory)
            ]
        )
    }

    func checkForUpdates() {
        updateCheckTask?.cancel()
        updateCheckTask = Task { [weak self] in
            let update = await AppUpdateChecker.availableUpdate()
            guard !Task.isCancelled else { return }
            self?.availableUpdate = update
        }
    }

    func dismissAvailableUpdate() {
        availableUpdate = nil
    }

    private static func loadRecipeSettings() -> PersistedRecipeSettings? {
        guard let data = UserDefaults.standard.data(forKey: recipeSettingsKey) else { return nil }
        return try? JSONDecoder().decode(PersistedRecipeSettings.self, from: data)
    }

    private static func persistRecipeSettings(_ recipe: GenerationRecipe) {
        guard let data = try? JSONEncoder().encode(PersistedRecipeSettings(recipe: recipe)) else { return }
        UserDefaults.standard.set(data, forKey: recipeSettingsKey)
    }

    private static func loadVideoOutputSettings() -> VideoOutputSettings? {
        guard let data = UserDefaults.standard.data(forKey: videoOutputSettingsKey) else { return nil }
        return try? JSONDecoder().decode(VideoOutputSettings.self, from: data)
    }

    private static func persistVideoOutputSettings(_ settings: VideoOutputSettings) {
        guard let data = try? JSONEncoder().encode(settings) else { return }
        UserDefaults.standard.set(data, forKey: videoOutputSettingsKey)
    }

    private static func loadMusicOutputSettings() -> MusicOutputSettings? {
        guard let data = UserDefaults.standard.data(forKey: musicOutputSettingsKey) else { return nil }
        return try? JSONDecoder().decode(MusicOutputSettings.self, from: data)
    }

    private static func persistMusicOutputSettings(_ settings: MusicOutputSettings) {
        guard let data = try? JSONEncoder().encode(settings) else { return }
        UserDefaults.standard.set(data, forKey: musicOutputSettingsKey)
    }

    private static func mergedModels(discovered: DiscoveredModelCatalog) -> [ModelDescriptor] {
        let builtInModels = ModelCatalog.builtIn.filter { model in
            if discovered.models.contains(where: { $0.id == model.id }) {
                return false
            }
            if model.id == "local-captioner-3b@q4" {
                return !discovered.models.contains { $0.capabilities.contains(.imageToText) }
            }
            if model.id == "qwen3-vl-8b-nsfw-caption-v45@mxfp4" {
                return !discovered.models.contains { $0.id == model.id }
            }
            if model.id == "realesrgan-x4@coreml" {
                return !discovered.models.contains {
                    $0.capabilities.contains(.upscale) && $0.displayName.contains("Real-ESRGAN 4×")
                }
            }
            if model.id == "realesrgan-x2@coreml" {
                return !discovered.models.contains {
                    $0.capabilities.contains(.upscale) && $0.displayName.contains("Real-ESRGAN 2×")
                }
            }
            return true
        }
        return discovered.models + builtInModels
    }

    private static func mergedProfiles(discovered: DiscoveredModelCatalog) -> [InferenceProfile] {
        let builtInProfiles = ModelCatalog.builtInProfiles.filter { profile in
            if discovered.profiles.contains(where: {
                $0.modelID == profile.modelID && $0.capability == profile.capability
            }) {
                return false
            }
            if profile.modelID == "local-captioner-3b@q4" {
                return !discovered.profiles.contains { $0.capability == .imageToText }
            }
            if profile.modelID == "qwen3-vl-8b-nsfw-caption-v45@mxfp4" {
                return !discovered.profiles.contains { $0.modelID == profile.modelID }
            }
            if profile.modelID == "realesrgan-x4@coreml" {
                return !discovered.profiles.contains {
                    $0.capability == .upscale && $0.defaults.upscaleScale == 4
                }
            }
            if profile.modelID == "realesrgan-x2@coreml" {
                return !discovered.profiles.contains {
                    $0.capability == .upscale && $0.defaults.upscaleScale == 2
                }
            }
            return true
        }
        return discovered.profiles + builtInProfiles
    }

    private static func installations(
        for models: [ModelDescriptor],
        preserving previous: [String: ModelInstallation] = [:]
    ) -> [String: ModelInstallation] {
        Dictionary(
            uniqueKeysWithValues: models.map { model in
                if model.localURL != nil {
                    return (
                        model.id,
                        ModelInstallation(
                            phase: .installed,
                            progress: 1,
                            downloadedGB: model.approximateDownloadGB
                        )
                    )
                }
                if previous[model.id]?.phase == .installed {
                    return (model.id, ModelInstallation())
                }
                return (model.id, previous[model.id] ?? ModelInstallation())
            }
        )
    }

    private static func profileSignature(_ profile: InferenceProfile) -> String {
        [
            profile.capability.rawValue,
            profile.modelID,
            profile.modelRevision,
            profile.name
        ].joined(separator: "\u{1F}")
    }

    private static func disabledProfileSignatures() -> Set<String> {
        Set(UserDefaults.standard.stringArray(forKey: disabledProfilesKey) ?? [])
    }

    private static func disabledProfileIDs(in profiles: [InferenceProfile]) -> Set<UUID> {
        let signatures = disabledProfileSignatures()
        return Set(profiles.filter { signatures.contains(profileSignature($0)) }.map(\.id))
    }

    private static func persistDisabledProfiles(
        _ disabledProfileIDs: Set<UUID>,
        in profiles: [InferenceProfile]
    ) {
        let signatures = profiles
            .filter { disabledProfileIDs.contains($0.id) }
            .map(profileSignature)
            .sorted()
        UserDefaults.standard.set(
            signatures,
            forKey: disabledProfilesKey
        )
    }

    private static func persistedActiveProfileIDs(
        in profiles: [InferenceProfile],
        models: [ModelDescriptor]
    ) -> [ModelCapability: UUID] {
        guard let signatures = UserDefaults.standard.dictionary(forKey: activeProfilesKey)
            as? [String: String] else { return [:] }

        return Dictionary(
            uniqueKeysWithValues: ModelCapability.allCases.compactMap { capability in
                guard let signature = signatures[capability.rawValue],
                      let profile = profiles.first(where: {
                          $0.capability == capability && profileSignature($0) == signature
                      }),
                      profile.requiredModelIDs.allSatisfy({ requiredModelID in
                          models.contains(where: {
                              $0.id == requiredModelID && $0.localURL != nil
                          })
                      }) else { return nil }
                return (capability, profile.id)
            }
        )
    }

    private static func persistActiveProfiles(
        _ activeProfileIDs: [ModelCapability: UUID],
        in profiles: [InferenceProfile]
    ) {
        let signaturePairs: [(String, String)] = activeProfileIDs.compactMap { entry in
            let (capability, profileID) = entry
            guard let profile = profiles.first(where: {
                $0.id == profileID && $0.capability == capability
            }) else { return nil }
            return (capability.rawValue, profileSignature(profile))
        }
        let signatures = Dictionary(uniqueKeysWithValues: signaturePairs)
        UserDefaults.standard.set(signatures, forKey: activeProfilesKey)
    }

    private static func persistedDimension(_ value: Int?, fallback: Int) -> Int {
        guard let value, (64...4096).contains(value), value.isMultiple(of: 16) else {
            return fallback
        }
        return value
    }

    private static func validatedVideoOutputSettings(
        _ saved: VideoOutputSettings?,
        defaults: ProfileDefaults
    ) -> VideoOutputSettings {
        VideoOutputSettings(
            width: persistedDimension(saved?.width, fallback: defaults.width ?? 1280),
            height: persistedDimension(saved?.height, fallback: defaults.height ?? 720),
            steps: persistedValue(saved?.steps, range: 1...100, fallback: defaults.steps ?? 8),
            outputCount: persistedValue(
                saved?.outputCount,
                range: 1...8,
                fallback: defaults.outputCount ?? 1
            ),
            frameCount: persistedValue(
                saved?.frameCount,
                range: 1...512,
                fallback: defaults.frameCount ?? 97
            ),
            frameRate: persistedValue(
                saved?.frameRate,
                range: 1...120,
                fallback: defaults.frameRate ?? 24
            ),
            seed: saved?.seed ?? UInt64.random(in: 0...UInt64.max)
        )
    }

    private static func validatedMusicOutputSettings(
        _ saved: MusicOutputSettings?,
        defaults: ProfileDefaults
    ) -> MusicOutputSettings {
        MusicOutputSettings(
            prompt: saved?.prompt ?? "",
            lyrics: saved?.lyrics ?? "",
            style: saved?.style ?? .pop,
            durationSeconds: persistedValue(
                saved?.durationSeconds,
                range: MusicGenerationOptions.supportedDurationSeconds,
                fallback: defaults.durationSeconds ?? 10
            ),
            steps: persistedValue(
                saved?.steps,
                range: 1...100,
                fallback: defaults.steps ?? 30
            ),
            seed: saved?.seed ?? UInt64.random(in: 0...UInt64.max),
            format: saved?.format ?? .mp3
        )
    }

    private static func validatedPersistedLoRA(
        _ selection: LoRASelection?,
        available: [LoRADescriptor]
    ) -> LoRASelection? {
        guard let selection,
              selection.scale.isFinite,
              (0...1).contains(selection.scale),
              let descriptor = available.first(where: { $0.id == selection.adapterID }) else {
            return nil
        }
        return LoRASelection(
            adapterID: descriptor.id,
            localURL: descriptor.localURL,
            scale: selection.scale
        )
    }

    private static func persistedValue(
        _ value: Int?,
        range: ClosedRange<Int>,
        fallback: Int
    ) -> Int {
        guard let value else { return fallback }
        return min(max(value, range.lowerBound), range.upperBound)
    }

    @discardableResult
    func setOutputDirectory(_ rawPath: String) -> Bool {
        let trimmedPath = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let expandedPath = (trimmedPath as NSString).expandingTildeInPath
        guard !expandedPath.isEmpty, NSString(string: expandedPath).isAbsolutePath else {
            statusMessage = "輸出目錄必須是絕對路徑。"
            return false
        }

        let outputURL = URL(fileURLWithPath: expandedPath, isDirectory: true).standardizedFileURL
        do {
            try FileManager.default.createDirectory(
                at: outputURL,
                withIntermediateDirectories: true
            )
        } catch {
            statusMessage = "無法建立輸出目錄：\(error.localizedDescription)"
            return false
        }

        outputDirectoryPath = outputURL.path
        UserDefaults.standard.set(outputURL.path, forKey: Self.outputDirectoryKey)
        let textToImageService = textToImageService
        let imageToImageService = imageToImageService
        let upscaleService = upscaleService
        Task {
            await textToImageService.setOutputDirectory(outputURL)
            await imageToImageService.setOutputDirectory(outputURL)
            await upscaleService.setOutputDirectory(outputURL)
        }
        videoGenerationService = LTXVideoGenerationService(outputDirectory: outputURL)
        musicGenerationService = Self.makeMusicGenerationService(outputDirectory: outputURL)
        statusMessage = "輸出目錄已更新：\(outputURL.path)"
        return true
    }

    func revealOutputDirectory() {
        let outputURL = URL(fileURLWithPath: outputDirectoryPath, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: outputURL, withIntermediateDirectories: true)
            guard NSWorkspace.shared.open(outputURL) else {
                throw CocoaError(.fileNoSuchFile)
            }
        } catch {
            statusMessage = "無法開啟輸出目錄：\(error.localizedDescription)"
        }
    }

    @discardableResult
    func setModelRoot(_ rawPath: String) -> Bool {
        let trimmedPath = rawPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let expandedPath = (trimmedPath as NSString).expandingTildeInPath
        guard !expandedPath.isEmpty, NSString(string: expandedPath).isAbsolutePath else {
            statusMessage = "模型路徑必須是絕對路徑。"
            return false
        }

        let rootURL = URL(fileURLWithPath: expandedPath, isDirectory: true).standardizedFileURL
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: rootURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            statusMessage = "找不到模型目錄：\(rootURL.path)"
            return false
        }

        let discovered = LocalModelDiscovery.discover(at: rootURL)
        let refreshedModels = Self.mergedModels(discovered: discovered)
        let customProfiles = profiles.filter { !$0.isBuiltIn }
        let refreshedProfiles = Self.mergedProfiles(discovered: discovered) + customProfiles
        let previousInstallations = installations
        let refreshedDisabledProfileIDs = Self.disabledProfileIDs(in: refreshedProfiles)

        modelTasks.values.forEach { $0.cancel() }
        modelTasks.removeAll()
        modelTaskTokens.removeAll()
        modelRootPath = rootURL.path
        UserDefaults.standard.set(rootURL.path, forKey: Self.modelRootKey)
        models = refreshedModels
        loras = discovered.loras
        profiles = refreshedProfiles
        disabledProfileIDs = refreshedDisabledProfileIDs
        installations = Self.installations(for: refreshedModels, preserving: previousInstallations)
        activeProfileIDs = Self.persistedActiveProfileIDs(
            in: refreshedProfiles,
            models: refreshedModels
        )

        recipe.lora = Self.validatedPersistedLoRA(recipe.lora, available: discovered.loras)
        if let generationProfile = activeProfile(for: .textToImage) {
            recipe.profileID = generationProfile.id
            recipe.modelID = generationProfile.modelID
        } else {
            recipe.profileID = nil
        }
        statusMessage = "已切換模型路徑，偵測到 \(discovered.models.count) 個本機模型與 \(discovered.loras.count) 個 LoRA。"
        return true
    }

    private func startSystemMetricsUpdates() {
        systemMetrics = SystemMetricsReader.read()
        systemMetricsTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                if Task.isCancelled { return }
                self?.systemMetrics = SystemMetricsReader.read()
            }
        }
    }

    var selectedProject: Project? {
        projects.first { $0.id == selectedProjectID }
    }

    var projectAssets: [ImageAsset] {
        assets
            .filter { $0.projectID == selectedProjectID }
            .sorted { $0.createdAt < $1.createdAt }
    }

    var selectedAsset: ImageAsset? {
        guard let selectedAssetID else { return nil }
        return assets.first { $0.id == selectedAssetID }
    }

    var selectedSourceImage: ImageAsset? {
        guard let asset = selectedAsset,
              asset.kind.isImage,
              let fileURL = asset.fileURL,
              FileManager.default.fileExists(atPath: fileURL.path) else {
            return nil
        }
        return asset
    }

    private func sourceImages(for assetIDs: [UUID]) -> [ImageAsset] {
        var seen = Set<UUID>()
        return assetIDs.compactMap { assetID in
            guard seen.insert(assetID).inserted,
                  let asset = assets.first(where: { $0.id == assetID }),
                  asset.kind.isImage,
                  let fileURL = asset.fileURL,
                  FileManager.default.fileExists(atPath: fileURL.path) else {
                return nil
            }
            return asset
        }
    }

    var comparisonAsset: ImageAsset? {
        guard let comparisonAssetID else { return nil }
        return assets.first { $0.id == comparisonAssetID }
    }

    var filteredModels: [ModelDescriptor] {
        models.filter { model in
            let matchesFilter = modelFilter.capability.map { model.capabilities.contains($0) } ?? true
            let search = modelSearch.trimmingCharacters(in: .whitespacesAndNewlines)
            let matchesSearch = search.isEmpty
                || model.displayName.localizedCaseInsensitiveContains(search)
                || model.publisher.localizedCaseInsensitiveContains(search)
            return matchesFilter && matchesSearch
        }
    }

    func profiles(for capability: ModelCapability) -> [InferenceProfile] {
        profiles.filter { $0.capability == capability }
    }

    func activeProfile(for capability: ModelCapability) -> InferenceProfile? {
        guard let id = activeProfileIDs[capability] else { return nil }
        return profiles.first { $0.id == id }
    }

    private var preferredVideoProfile: InferenceProfile? {
        if selectedSourceImage != nil, let profile = activeProfile(for: .imageToVideo) {
            return profile
        }
        return activeProfile(for: .textToVideo) ?? activeProfile(for: .imageToVideo)
    }

    private func ensureInferenceIdle() -> Bool {
        guard !jobs.contains(where: { [.queued, .running, .cancelling].contains($0.state) }) else {
            statusMessage = "已有生成或辨識任務正在執行，請完成或取消後再開始新任務。"
            return false
        }
        return true
    }

    func selectProfile(_ profileID: UUID, for capability: ModelCapability) {
        guard let profile = profiles.first(where: { $0.id == profileID && $0.capability == capability }) else {
            return
        }
        let missingModels = missingProfileModels(profile)
        guard missingModels.isEmpty else {
            statusMessage = "請先下載並驗證「\(profile.name)」的相關模型：\(missingModels.map(\.displayName).joined(separator: "、"))。"
            return
        }
        if let compatibilityError = profileCompatibilityError(profile) {
            statusMessage = compatibilityError
            return
        }
        let previousProfileID = activeProfileIDs[capability]
        activeProfileIDs[capability] = nil
        disabledProfileIDs.remove(profileID)
        Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)
        activeProfileIDs[capability] = profileID
        Self.persistActiveProfiles(activeProfileIDs, in: profiles)

        if capability == .textToImage {
            recipe.profileID = profileID
            recipe.modelID = profile.modelID
        }
        if previousProfileID != profileID {
            statusMessage = "已啟用「\(profile.name)」；相容性檢查通過，同類型 Profile 維持互斥。"
            releaseNonFocusedModelsIfNeeded(focusing: capability)
        }
    }

    /// 切換 Profile 時避免多個大型 Runtime 同時常駐。記憶體讀值超過
    /// 90% 才執行釋放，而且保留目前切換到的能力所需 Runtime。
    private func releaseNonFocusedModelsIfNeeded(focusing capability: ModelCapability) {
        guard let ramUsage = SystemMetricsReader.read().ramUsage,
              ramUsage > 0.90 else {
            return
        }

        let usagePercent = Int((ramUsage * 100).rounded())
        let textToImage = textToImageService
        let imageToText = imageToTextService
        let upscale = upscaleService

        Task { @MainActor [weak self, textToImage, imageToText, upscale] in
            var released: [String] = []
            if capability != .textToImage {
                await textToImage.unload()
                released.append("文生圖")
            }
            if capability != .imageToText {
                await imageToText.unload()
                released.append("圖生文")
            }
            if capability != .upscale {
                await upscale.unload()
                released.append("Upscale")
            }
            guard !released.isEmpty, let self else { return }
            self.statusMessage = "記憶體使用率約 \(usagePercent)%；已釋放非焦點模型：\(released.joined(separator: "、"))。"
        }
    }

    private func profileCompatibilityError(
        _ profile: InferenceProfile,
        modelOverride: ModelDescriptor? = nil
    ) -> String? {
        guard let model = modelOverride ?? models.first(where: { $0.id == profile.modelID }) else {
            return "Profile「\(profile.name)」找不到指定模型。"
        }
        guard let localURL = model.localURL,
              FileManager.default.fileExists(atPath: localURL.path) else {
            return "Profile「\(profile.name)」的模型路徑不存在。"
        }
        guard model.capabilities.contains(profile.capability) else {
            return "Profile「\(profile.name)」的模型不支援「\(profile.capability.title)」。"
        }

        let compatibleArchitectures: Set<InferenceArchitecture>?
        switch profile.capability {
        case .textToImage, .imageToText:
            compatibleArchitectures = [.mlxSwift]
        case .upscale:
            compatibleArchitectures = [.coreML]
        case .imageToImage, .imageToVideo, .textToVideo:
            compatibleArchitectures = [.externalCLI]
        case .textToMusic:
            compatibleArchitectures = [.mlxSwift, .externalCLI]
        case .controlNet, .lora:
            compatibleArchitectures = nil
        }
        if let compatibleArchitectures,
           !compatibleArchitectures.contains(profile.architecture) {
            let architectureNames = compatibleArchitectures
                .map(\.title)
                .sorted()
                .joined(separator: "、")
            return "Profile「\(profile.name)」的架構「\(profile.architecture.title)」與目前 Runtime 不相容；需要「\(architectureNames)」。"
        }

        for configuration in profile.loras {
            guard let loraModel = models.first(where: { $0.id == configuration.modelID }),
                  let loraURL = loraModel.localURL else {
                return "Profile「\(profile.name)」找不到 LoRA「\(configuration.modelID)」。"
            }
            if let error = loraCompatibilityError(at: loraURL) {
                return "Profile「\(profile.name)」的 LoRA 不相容：\(error)"
            }
        }
        return nil
    }

    func loraCompatibilityError(at url: URL) -> String? {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return "找不到權重路徑：\(url.path)"
        }

        let candidates: [URL]
        if isDirectory.boolValue {
            candidates = (FileManager.default.enumerator(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )?.compactMap { $0 as? URL }.filter { candidate in
                (try? candidate.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
            }) ?? []
        } else {
            candidates = [url]
        }

        var sawReadableHeader = false
        for candidate in candidates {
            guard let keys = safetensorHeaderKeys(at: candidate) else { continue }
            sawReadableHeader = true
            if hasLoRAPairs(keys) { return nil }
        }
        return sawReadableHeader
            ? "找不到可套用的 LoRA 權重配對（A/B 或 LoKr）。"
            : "無法讀取可辨識的權重標頭。"
    }

    private func safetensorHeaderKeys(at url: URL) -> [String]? {
        guard let data = try? Data(contentsOf: url, options: [.mappedIfSafe]), data.count >= 8 else {
            return nil
        }
        var headerLength: UInt64 = 0
        for (index, byte) in data.prefix(8).enumerated() {
            headerLength |= UInt64(byte) << UInt64(index * 8)
        }
        guard headerLength <= UInt64(data.count - 8), headerLength <= UInt64(Int.max) else {
            return nil
        }
        let start = data.index(data.startIndex, offsetBy: 8)
        let end = data.index(start, offsetBy: Int(headerLength))
        guard let header = try? JSONSerialization.jsonObject(with: data[start..<end]) as? [String: Any] else {
            return nil
        }
        return header.keys.filter { $0 != "__metadata__" }
    }

    private func hasLoRAPairs(_ keys: [String]) -> Bool {
        let suffixPairs = [(".lora_down", ".lora_up"), (".lora_A", ".lora_B")]
        for (downSuffix, upSuffix) in suffixPairs {
            let downBases = Set(keys.compactMap { loraBase($0, suffix: downSuffix) })
            let upBases = Set(keys.compactMap { loraBase($0, suffix: upSuffix) })
            if !downBases.isDisjoint(with: upBases) { return true }
        }
        let w1Bases = Set(keys.compactMap { loraBase($0, suffix: ".lokr_w1") })
        let w2Bases = Set(keys.compactMap { loraBase($0, suffix: ".lokr_w2") })
        return !w1Bases.isDisjoint(with: w2Bases)
    }

    private func loraBase(_ key: String, suffix: String) -> String? {
        let rawKey = key.hasSuffix(".weight") ? String(key.dropLast(".weight".count)) : key
        guard rawKey.hasSuffix(suffix) else { return nil }
        return String(rawKey.dropLast(suffix.count))
    }

    private func deactivateProfiles(usingModelID modelID: String) {
        var changed = false
        for profile in profiles where profile.requiredModelIDs.contains(modelID) {
            guard activeProfileIDs[profile.capability] == profile.id else { continue }
            activeProfileIDs[profile.capability] = nil
            disabledProfileIDs.insert(profile.id)
            if profile.capability == .textToImage {
                recipe.profileID = nil
            }
            changed = true
        }
        if changed {
            Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)
            Self.persistActiveProfiles(activeProfileIDs, in: profiles)
        }
    }

    func deactivateProfile(_ profileID: UUID, for capability: ModelCapability) {
        guard activeProfileIDs[capability] == profileID,
              let profile = profiles.first(where: { $0.id == profileID && $0.capability == capability }) else {
            return
        }
        disabledProfileIDs.insert(profileID)
        Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)
        activeProfileIDs[capability] = nil
        Self.persistActiveProfiles(activeProfileIDs, in: profiles)
        if capability == .textToImage {
            recipe.profileID = nil
        }
        let hasAvailableProfile = profiles.contains {
            $0.capability == capability && !disabledProfileIDs.contains($0.id)
        }
        statusMessage = hasAvailableProfile
            ? "已停用「\(profile.name)」；左側工具已隱藏，可到 Profiles 頁啟用其他 Profile。"
            : "已停用「\(profile.name)」；此類型已全部停用，左側工具已隱藏。"
    }

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
        if let width { updated.width = min(max((width / 16) * 16, 64), 4096) }
        if let height { updated.height = min(max((height / 16) * 16, 64), 4096) }
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

    private func normalizedVideoFrameCount(
        _ frameCount: Int,
        for profile: InferenceProfile?
    ) -> Int {
        let clamped = min(max(frameCount, 1), 512)
        guard profile?.modelID.lowercased().contains("ltx-2.3-mlx") == true else {
            return clamped
        }
        return LTXVideoGenerationService.normalizedFrameCount(clamped)
    }

    func requestVideoGeneration(sourceAssetIDs: [UUID] = []) {
        guard ensureInferenceIdle() else { return }
        let profile: InferenceProfile
        let sourceAssets: [ImageAsset]
        if sourceAssetIDs.isEmpty {
            guard let preferredVideoProfile else {
                statusMessage = "請先啟用文生影或圖生影 Profile。"
                return
            }
            profile = preferredVideoProfile
            if profile.capability == .imageToVideo {
                guard let selectedSourceImage else {
                    statusMessage = "圖生影需要先匯入或選取至少一張圖片。"
                    return
                }
                sourceAssets = [selectedSourceImage]
            } else {
                sourceAssets = []
            }
        } else {
            let resolvedSourceAssets = sourceImages(for: sourceAssetIDs)
            guard resolvedSourceAssets.count == Set(sourceAssetIDs).count else {
                statusMessage = "部分圖生影錨點已不存在或不是可用圖片，請重新選取。"
                return
            }
            guard let imageToVideoProfile = activeProfile(for: .imageToVideo) else {
                statusMessage = "已選取圖片錨點，請先啟用圖生影 Profile。"
                return
            }
            profile = imageToVideoProfile
            sourceAssets = resolvedSourceAssets
        }
        guard isProfileReady(profile) else { return }
        guard let model = models.first(where: { $0.id == profile.modelID }) else {
            statusMessage = "找不到影片 Profile 指定的模型。"
            return
        }
        guard let modelURL = model.localURL else {
            statusMessage = "找不到影片模型的本機安裝路徑。"
            return
        }
        guard let profileLoRAs = resolvedVideoLoRAs(for: profile) else { return }

        let normalizedFrameCount = normalizedVideoFrameCount(
            videoOutputSettings.frameCount,
            for: profile
        )
        if normalizedFrameCount != videoOutputSettings.frameCount {
            videoOutputSettings.frameCount = normalizedFrameCount
        }

        let options = VideoGenerationOptions(
            prompt: recipe.prompt,
            width: videoOutputSettings.width,
            height: videoOutputSettings.height,
            steps: videoOutputSettings.steps,
            outputCount: videoOutputSettings.outputCount,
            frameCount: normalizedFrameCount,
            frameRate: videoOutputSettings.frameRate,
            seed: videoOutputSettings.seed
        )
        do {
            try options.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }

        let projectID = selectedProjectID
        let recipeID = recipe.id
        let jobTitle: String
        if profile.capability == .imageToVideo {
            jobTitle = sourceAssets.count == 1
                ? "從「\(sourceAssets[0].title)」生成影片"
                : "使用 \(sourceAssets.count) 張圖片錨點生成影片"
        } else {
            jobTitle = "生成 \(options.outputCount) 部影片"
        }
        let job = GenerationJob(
            action: .generateVideo,
            title: jobTitle
        )
        jobs.append(job)
        updateJob(job.id) {
            $0.state = .running
            $0.progress = 0.001
        }
        let service = videoGenerationService
        let request = VideoGenerationRequest(
            projectID: projectID,
            recipeID: recipeID,
            sourceAsset: sourceAssets.first,
            sourceAssets: sourceAssets,
            options: options,
            profile: profile,
            modelURL: modelURL,
            profileLoRAs: profileLoRAs
        )

        let jobID = job.id
        let videoTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                var newAssets = try await service.generate(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(jobID, value: value)
                        }
                    }
                )
                try Task.checkCancellation()

                let operationID = UUID()
                for index in newAssets.indices {
                    newAssets[index].operationID = operationID
                }
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    assets.append(contentsOf: newAssets)
                    operations.append(
                        WorkflowOperation(
                            id: operationID,
                            projectID: projectID,
                            action: .generateVideo,
                            inputAssetID: sourceAssets.first?.id,
                            outputAssetIDs: newAssets.map(\.id),
                            recipeID: recipeID,
                            profileSnapshot: profile
                        )
                    )
                    comparisonAssetID = nil
                    selectedAssetID = newAssets.first?.id
                    previewMode = .single
                    updateJob(jobID) {
                        $0.progress = 1
                        $0.state = .completed
                    }
                    statusMessage = profile.capability == .imageToVideo
                        ? "圖生影完成。"
                        : "文生影完成。"
                }
            } catch is CancellationError {
                await MainActor.run { [weak self] in
                    guard let self,
                          jobs.first(where: { [.running, .cancelling].contains($0.state) }) != nil else {
                        return
                    }
                    updateJob(jobID) { $0.state = .cancelled }
                }
            } catch {
                let message = error.localizedDescription
                await MainActor.run { [weak self] in
                    guard let self,
                          jobs.first(where: { [.running, .cancelling].contains($0.state) }) != nil else {
                        return
                    }
                    updateJob(jobID) {
                        $0.state = .failed
                        $0.errorMessage = message
                    }
                    statusMessage = "影片生成失敗：\(message)"
                }
            }
            await MainActor.run { [weak self] in
                self?.finishJobTask(jobID)
            }
        }
        jobTasks[jobID] = videoTask

        Task.detached(priority: .utility) { [weak self] in
            try? await Task.sleep(for: .seconds(15))
            guard !Task.isCancelled else { return }
            await self?.failVideoJobIfStartupStalled(jobID)
        }
    }

    private func failVideoJobIfStartupStalled(_ jobID: UUID) {
        guard let job = jobs.first(where: { $0.id == jobID }),
              job.state == .running,
              job.progress < 0.01 else {
            return
        }
        jobTasks[jobID]?.cancel()
        jobTasks[jobID] = nil
        let message = "影片生成 Runtime 在 15 秒內未啟動，任務已自動停止；請重新執行。"
        updateJob(jobID) {
            $0.state = .failed
            $0.errorMessage = message
        }
        statusMessage = message
    }

    private static func formattedMusicDuration(_ durationSeconds: Double) -> String {
        let totalSeconds = max(0, Int(durationSeconds.rounded()))
        let minutes = totalSeconds / 60
        let seconds = totalSeconds % 60
        if minutes > 0, seconds > 0 {
            return "\(minutes) 分 \(seconds) 秒"
        }
        if minutes > 0 {
            return "\(minutes) 分"
        }
        return "\(seconds) 秒"
    }

    func requestMusicGeneration() {
        guard ensureInferenceIdle() else { return }
        guard let profile = activeProfile(for: .textToMusic) else {
            statusMessage = "請先啟用文生音樂 Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let model = models.first(where: { $0.id == profile.modelID }),
              let modelURL = model.localURL else {
            statusMessage = "找不到音樂模型的本機安裝路徑。"
            return
        }
        let musicPrompt = musicOutputSettings.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let options = MusicGenerationOptions(
            prompt: musicPrompt.isEmpty ? musicOutputSettings.style.prompt : musicPrompt,
            lyrics: musicOutputSettings.lyrics,
            durationSeconds: musicOutputSettings.durationSeconds,
            steps: musicOutputSettings.steps,
            seed: musicOutputSettings.seed,
            format: musicOutputSettings.format
        )
        do {
            try options.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }

        let projectID = selectedProjectID
        let recipeID = recipe.id
        let requestedDuration = "\(options.durationSeconds) 秒"
        let durationTitle = profile.music?.durationSemantics == .maximum
            ? "最長 \(requestedDuration)"
            : requestedDuration
        let job = GenerationJob(
            action: .generateMusic,
            title: "生成 \(durationTitle) \(options.format.displayName) 音樂"
        )
        jobs.append(job)
        updateJob(job.id) {
            $0.state = .running
            $0.progress = 0.001
        }
        let service = musicGenerationService
        let request = MusicGenerationRequest(
            projectID: projectID,
            recipeID: recipeID,
            options: options,
            profile: profile,
            modelURL: modelURL
        )
        let jobID = job.id
        let musicTask = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                var asset = try await service.generate(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJob(jobID) {
                                $0.progress = max($0.progress, min(1, max(0, value)))
                            }
                        }
                    }
                )
                try Task.checkCancellation()
                let operationID = UUID()
                asset.operationID = operationID
                await MainActor.run { [weak self] in
                    guard let self else { return }
                    assets.append(asset)
                    operations.append(
                        WorkflowOperation(
                            id: operationID,
                            projectID: projectID,
                            action: .generateMusic,
                            outputAssetIDs: [asset.id],
                            recipeID: recipeID,
                            profileSnapshot: profile
                        )
                    )
                    comparisonAssetID = nil
                    selectedAssetID = asset.id
                    previewMode = .single
                    updateJob(jobID) {
                        $0.progress = 1
                        $0.state = .completed
                    }
                    let actualDuration = asset.mediaDurationSeconds.map(Self.formattedMusicDuration)
                        ?? "未知"
                    let requestedLabel = profile.music?.durationSemantics == .maximum
                        ? "設定最長 \(requestedDuration)"
                        : "設定 \(requestedDuration)"
                    statusMessage = "文生音樂完成，\(requestedLabel)，實際生成 \(actualDuration)，已輸出 \(options.format.displayName)。"
                }
            } catch is CancellationError {
                await MainActor.run { [weak self] in
                    guard let self,
                          jobs.first(where: { $0.id == jobID })?.state == .running else {
                        return
                    }
                    updateJob(jobID) { $0.state = .cancelled }
                }
            } catch {
                let message = error.localizedDescription
                await MainActor.run { [weak self] in
                    guard let self,
                          jobs.first(where: { $0.id == jobID })?.state == .running else {
                        return
                    }
                    updateJob(jobID) {
                        $0.state = .failed
                        $0.errorMessage = message
                    }
                    statusMessage = "音樂生成失敗：\(message)"
                }
            }
            await MainActor.run { [weak self] in
                self?.jobTasks[jobID] = nil
            }
        }
        jobTasks[jobID] = musicTask
    }

    func chooseSize(width: Int, height: Int) {
        recipe.applySizePreset(width: width, height: height)
    }

    func selectAsset(_ id: UUID) {
        if selectedAssetID != id {
            comparisonAssetID = selectedAssetID
            selectedAssetID = id
        }
    }

    func removeAsset(_ id: UUID, selecting replacementID: UUID?) {
        guard let removedAsset = assets.first(where: { $0.id == id }) else { return }

        operations = operations.compactMap { operation in
            let referencedInput = operation.inputAssetID == id
            let referencedOutput = operation.outputAssetIDs.contains(id)
            guard referencedInput || referencedOutput else { return operation }

            var updated = operation
            if referencedInput { updated.inputAssetID = nil }
            updated.outputAssetIDs.removeAll { $0 == id }

            if updated.outputAssetIDs.isEmpty,
               operation.action != .describe || referencedInput {
                return nil
            }
            return updated
        }

        assets.removeAll { $0.id == id }
        for index in assets.indices where assets[index].parentAssetID == id {
            assets[index].parentAssetID = nil
        }

        if selectedAssetID == id {
            selectedAssetID = replacementID.flatMap { replacement in
                assets.contains(where: { $0.id == replacement }) ? replacement : nil
            }
        }
        if comparisonAssetID == id || comparisonAssetID == selectedAssetID {
            comparisonAssetID = nil
        }

        let fileRemovalError = removeManagedAssetFile(at: removedAsset.fileURL)
        if let fileRemovalError {
            statusMessage = "已從工作區移除「\(removedAsset.title)」，但無法刪除應用程式副本：\(fileRemovalError.localizedDescription)"
        } else {
            statusMessage = "已移除「\(removedAsset.title)」。"
        }
    }

    func closeWorkspaceProject(assetIDs: [UUID]) {
        let closedAssetIDs = Set(assetIDs).intersection(Set(assets.map(\.id)))
        guard !closedAssetIDs.isEmpty else { return }

        operations.removeAll { operation in
            operation.inputAssetID.map(closedAssetIDs.contains) == true
                || !closedAssetIDs.isDisjoint(with: operation.outputAssetIDs)
        }
        assets.removeAll { closedAssetIDs.contains($0.id) }
        for index in assets.indices where assets[index].parentAssetID.map(closedAssetIDs.contains) == true {
            assets[index].parentAssetID = nil
        }
        if selectedAssetID.map(closedAssetIDs.contains) == true {
            selectedAssetID = nil
        }
        if comparisonAssetID.map(closedAssetIDs.contains) == true {
            comparisonAssetID = nil
        }
        statusMessage = "已關閉生成專案分頁；\(closedAssetIDs.count) 個結果已從工作區移除，輸出檔案仍保留於磁碟。"
    }

    @discardableResult
    func importImage(url: URL, pixelWidth: Int, pixelHeight: Int) -> UUID {
        let asset = ImageAsset(
            projectID: selectedProjectID,
            kind: .imported,
            title: url.deletingPathExtension().lastPathComponent,
            fileURL: url,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight
        )
        assets.append(asset)
        operations.append(
            WorkflowOperation(
                projectID: selectedProjectID,
                action: .importImage,
                outputAssetIDs: [asset.id]
            )
        )
        selectAsset(asset.id)
        statusMessage = "已匯入「\(asset.title)」；可以執行圖生文、圖生圖、圖生影或 Upscale。"
        return asset.id
    }

    func describeSelected() {
        guard ensureInferenceIdle() else { return }
        guard let input = selectedSourceImage else {
            statusMessage = "請先匯入或選取一張圖片。"
            return
        }
        guard let profile = activeProfile(for: .imageToText) else {
            statusMessage = "請先設定圖生文 Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let executionProfile = runtimeProfile(from: profile) else { return }

        let job = GenerationJob(action: .describe, title: "描述「\(input.title)」")
        jobs.append(job)
        updateJob(job.id) { $0.state = .running }
        let service = imageToTextService
        let request = ImageDescriptionRequest(
            asset: input,
            profile: executionProfile,
            languageCode: executionProfile.defaults.languageCode ?? "zh-Hant"
        )

        jobTasks[job.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let description = try await service.describe(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(job.id, value: value)
                        }
                    }
                )
                try Task.checkCancellation()
                recipe.prompt = description
                operations.append(
                    WorkflowOperation(
                        projectID: selectedProjectID,
                        action: .describe,
                        inputAssetID: input.id,
                        recipeID: recipe.id,
                        profileSnapshot: profile
                    )
                )
                updateJob(job.id) {
                    $0.progress = 1
                    $0.state = .completed
                }
                statusMessage = "圖生文完成，描述已放入 Prompt，可直接修改或生成。"
            } catch is CancellationError {
                updateJob(job.id) { $0.state = .cancelled }
            } catch {
                let message = error.localizedDescription
                updateJob(job.id) {
                    $0.state = .failed
                    $0.errorMessage = message
                }
                statusMessage = "圖生文失敗：\(message)"
            }
            finishJobTask(job.id)
        }
    }

    func generate(linkToSelectedAsset: Bool = false) {
        guard ensureInferenceIdle() else { return }
        do {
            try recipe.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }
        guard let profile = activeProfile(for: .textToImage) else {
            statusMessage = "請先設定文生圖 Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let executionProfile = runtimeProfile(from: profile) else { return }

        let input = linkToSelectedAsset ? selectedAsset : nil
        let recipeSnapshot = recipe
        let projectID = selectedProjectID
        let job = GenerationJob(action: .generate, title: "生成 \(recipeSnapshot.outputCount) 張圖片")
        jobs.append(job)
        updateJob(job.id) { $0.state = .running }
        let service = textToImageService
        let request = TextToImageRequest(
            projectID: projectID,
            recipe: recipeSnapshot,
            profile: executionProfile,
            sourceAsset: input
        )

        jobTasks[job.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                var newAssets = try await service.generate(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(job.id, value: value)
                        }
                    }
                )
                try Task.checkCancellation()

                let operationID = UUID()
                for index in newAssets.indices {
                    newAssets[index].operationID = operationID
                }
                assets.append(contentsOf: newAssets)
                operations.append(
                    WorkflowOperation(
                        id: operationID,
                        projectID: projectID,
                        action: .generate,
                        inputAssetID: input?.id,
                        outputAssetIDs: newAssets.map(\.id),
                        recipeID: recipeSnapshot.id,
                        profileSnapshot: profile
                    )
                )
                comparisonAssetID = newAssets.dropFirst().first?.id
                selectedAssetID = newAssets.first?.id
                previewMode = .grid
                updateJob(job.id) {
                    $0.progress = 1
                    $0.state = .completed
                }
                statusMessage = input == nil ? "獨立文生圖完成。" : "已從選取圖片建立新的生成分支。"
            } catch is CancellationError {
                updateJob(job.id) { $0.state = .cancelled }
            } catch {
                let message = error.localizedDescription
                updateJob(job.id) {
                    $0.state = .failed
                    $0.errorMessage = message
                }
                statusMessage = "文生圖失敗：\(message)"
            }
            finishJobTask(job.id)
        }
    }

    func imageToImageSelected() {
        guard ensureInferenceIdle() else { return }
        guard let input = selectedSourceImage else {
            statusMessage = "請先匯入或選取一張圖片。"
            return
        }
        do {
            try recipe.validate()
        } catch {
            statusMessage = error.localizedDescription
            return
        }
        guard let profile = activeProfile(for: .imageToImage) else {
            statusMessage = "請先啟用圖生圖 Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let model = models.first(where: { $0.id == profile.modelID }) else {
            statusMessage = "找不到圖生圖 Profile 指定的模型。"
            return
        }
        guard let modelURL = model.localURL else {
            statusMessage = "找不到圖生圖模型的本機安裝路徑。"
            return
        }

        let recipeSnapshot = recipe
        let projectID = selectedProjectID
        let job = GenerationJob(action: .imageToImage, title: "編輯「\(input.title)」")
        jobs.append(job)
        updateJob(job.id) { $0.state = .running }
        let service = imageToImageService
        let request = ImageToImageRequest(
            projectID: projectID,
            sourceAsset: input,
            recipe: recipeSnapshot,
            profile: profile,
            modelURL: modelURL,
            quantization: model.quantization
        )

        jobTasks[job.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                var output = try await service.generate(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(job.id, value: value)
                        }
                    }
                )
                try Task.checkCancellation()
                let operationID = UUID()
                output.operationID = operationID
                assets.append(output)
                operations.append(
                    WorkflowOperation(
                        id: operationID,
                        projectID: projectID,
                        action: .imageToImage,
                        inputAssetID: input.id,
                        outputAssetIDs: [output.id],
                        recipeID: recipeSnapshot.id,
                        profileSnapshot: profile
                    )
                )
                comparisonAssetID = input.id
                selectedAssetID = output.id
                previewMode = .compare
                updateJob(job.id) {
                    $0.progress = 1
                    $0.state = .completed
                }
                statusMessage = "圖生圖完成，已切換至前後比較。"
            } catch is CancellationError {
                updateJob(job.id) { $0.state = .cancelled }
            } catch {
                let message = error.localizedDescription
                updateJob(job.id) {
                    $0.state = .failed
                    $0.errorMessage = message
                }
                statusMessage = "圖生圖失敗：\(message)"
            }
            finishJobTask(job.id)
        }
    }

    private func removeManagedAssetFile(at fileURL: URL?) -> Error? {
        guard let fileURL else { return nil }
        let fileManager = FileManager.default
        let applicationSupport = fileManager.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? fileManager.temporaryDirectory
        let managedRoot = applicationSupport
            .appendingPathComponent("GenImage", isDirectory: true)
            .resolvingSymlinksInPath()
            .standardizedFileURL
        let candidate = fileURL.resolvingSymlinksInPath().standardizedFileURL
        let rootPath = managedRoot.path.hasSuffix("/") ? managedRoot.path : managedRoot.path + "/"

        guard candidate.path.hasPrefix(rootPath), fileManager.fileExists(atPath: candidate.path) else {
            return nil
        }
        do {
            try fileManager.removeItem(at: candidate)
            return nil
        } catch {
            return error
        }
    }

    func upscaleSelected() {
        guard ensureInferenceIdle() else { return }
        guard let input = selectedSourceImage else {
            statusMessage = "請先匯入或選取一張圖片。"
            return
        }
        guard let profile = activeProfile(for: .upscale) else {
            statusMessage = "請先設定 Upscale Profile。"
            return
        }
        guard isProfileReady(profile) else { return }
        guard let executionProfile = runtimeProfile(from: profile) else { return }
        let scale = executionProfile.defaults.upscaleScale ?? 4
        let job = GenerationJob(action: .upscale, title: "放大「\(input.title)」\(scale)×")
        jobs.append(job)
        updateJob(job.id) { $0.state = .running }
        let service = upscaleService
        let request = UpscaleRequest(asset: input, profile: executionProfile, scale: scale)

        jobTasks[job.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                var output = try await service.upscale(
                    request: request,
                    progress: { [weak self] value in
                        Task { @MainActor [weak self] in
                            self?.updateJobProgress(job.id, value: value)
                        }
                    }
                )
                try Task.checkCancellation()

                let operationID = UUID()
                output.operationID = operationID
                assets.append(output)
                operations.append(
                    WorkflowOperation(
                        id: operationID,
                        projectID: selectedProjectID,
                        action: .upscale,
                        inputAssetID: input.id,
                        outputAssetIDs: [output.id],
                        recipeID: input.recipeID,
                        profileSnapshot: profile
                    )
                )
                comparisonAssetID = input.id
                selectedAssetID = output.id
                previewMode = .compare
                updateJob(job.id) {
                    $0.progress = 1
                    $0.state = .completed
                }
                statusMessage = "Upscale 完成，已切換至前後比較。"
            } catch is CancellationError {
                updateJob(job.id) { $0.state = .cancelled }
            } catch {
                let message = error.localizedDescription
                updateJob(job.id) {
                    $0.state = .failed
                    $0.errorMessage = message
                }
                statusMessage = "Upscale 失敗：\(message)"
            }
            finishJobTask(job.id)
        }
    }

    func cancelJob(_ id: UUID) {
        guard let job = jobs.first(where: { $0.id == id }) else { return }
        guard [.queued, .running].contains(job.state) else { return }
        guard let task = jobTasks[id] else {
            updateJob(id) { $0.state = .cancelled }
            return
        }
        cancellationRequestedJobIDs.insert(id)
        updateJob(id) { $0.state = .cancelling }
        task.cancel()
        Task { @MainActor [weak self] in
            await task.value
            self?.finishJobTask(id)
        }
        statusMessage = "正在取消任務，Runtime 停止後即可開始下一個任務。"
    }

    private func finishJobTask(_ id: UUID) {
        jobTasks[id] = nil
        let cancellationRequested = cancellationRequestedJobIDs.remove(id) != nil
        guard cancellationRequested
            || jobs.first(where: { $0.id == id })?.state == .cancelling else { return }
        updateJob(id) { $0.state = .cancelled }
        statusMessage = "任務已取消，可以繼續操作。"
    }

    func releaseMemory() {
        guard !jobs.contains(where: { [.queued, .running, .cancelling].contains($0.state) }) else {
            statusMessage = "任務執行或取消中，完成後才能釋放記憶體。"
            return
        }
        guard !isReleasingMemory else { return }
        isReleasingMemory = true
        statusMessage = "正在釋放已載入的模型記憶體…"
        let textToImage = textToImageService
        let imageToText = imageToTextService
        let upscale = upscaleService
        Task { @MainActor [weak self] in
            await textToImage.unload()
            await imageToText.unload()
            await upscale.unload()
            guard let self else { return }
            isReleasingMemory = false
            statusMessage = "模型記憶體已釋放。"
        }
    }

    func clearFinishedJobs() {
        jobs.removeAll { [.completed, .cancelled, .failed].contains($0.state) }
    }

    func installation(for modelID: String) -> ModelInstallation {
        installations[modelID] ?? ModelInstallation()
    }

    func installProfileModels(_ profileID: UUID) {
        guard let profile = profiles.first(where: { $0.id == profileID }) else { return }
        let requiredModels = profile.requiredModelIDs.compactMap { requiredModelID in
            models.first { $0.id == requiredModelID }
        }
        guard requiredModels.count == profile.requiredModelIDs.count else {
            let knownIDs = Set(requiredModels.map(\.id))
            let missingIDs = profile.requiredModelIDs.filter { !knownIDs.contains($0) }
            statusMessage = "找不到 Profile 的相關模型：\(missingIDs.joined(separator: "、"))"
            return
        }

        var startedCount = 0
        for model in requiredModels where installation(for: model.id).phase != .installed {
            installModel(model)
            startedCount += 1
        }
        statusMessage = startedCount > 0
            ? "已開始下載「\(profile.name)」所需的 \(startedCount) 個相關模型；已安裝項目會自動去重。"
            : "「\(profile.name)」的相關模型皆已安裝。"
    }

    func installModel(_ model: ModelDescriptor) {
        if model.localURL != nil {
            installations[model.id] = ModelInstallation(
                phase: .installed,
                progress: 1,
                downloadedGB: model.approximateDownloadGB
            )
            return
        }
        guard HuggingFaceModelInstaller.supports(modelID: model.id) else {
            let message = "此模型尚未提供可執行的自動下載方案。"
            installations[model.id] = ModelInstallation(phase: .failed, errorMessage: message)
            statusMessage = message
            return
        }
        guard modelTasks[model.id] == nil else {
            statusMessage = "「\(model.displayName)」已有下載任務正在執行。"
            return
        }
        let taskToken = UUID()
        modelTaskTokens[model.id] = taskToken
        let startProgress = min(1, max(0, installations[model.id]?.progress ?? 0))
        installations[model.id] = ModelInstallation(
            phase: .downloading,
            progress: startProgress,
            downloadedGB: model.approximateDownloadGB * startProgress
        )
        let progressGate = ModelProgressGate(interval: Self.modelProgressUpdateInterval)

        modelTasks[model.id] = Task { @MainActor [weak self] in
            guard let self else { return }
            defer {
                if modelTaskTokens[model.id] == taskToken {
                    modelTasks[model.id] = nil
                    modelTaskTokens[model.id] = nil
                }
            }
            do {
                let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
                let localURL = try await modelInstaller.install(
                    modelID: model.id,
                    rootURL: rootURL,
                    progress: { [weak self] update in
                        guard progressGate.shouldEmit(update) else { return }
                        Task { @MainActor [weak self] in
                            guard let self, modelTaskTokens[model.id] == taskToken else { return }
                            // The resolved Hugging Face file list is authoritative;
                            // keep the catalog estimate in sync so repositories that
                            // add or remove shards do not show e.g. 9.6 GB / 7.6 GB.
                            let resolvedTotalGB = Double(update.totalBytes) / 1_073_741_824
                            if resolvedTotalGB > 0,
                               abs(models.first(where: { $0.id == model.id })?.approximateDownloadGB ?? 0
                                   - resolvedTotalGB) > 0.01 {
                                if let index = models.firstIndex(where: { $0.id == model.id }) {
                                    models[index].approximateDownloadGB = resolvedTotalGB
                                }
                            }
                            let downloadedGB = Double(update.downloadedBytes) / 1_073_741_824
                            installations[model.id] = ModelInstallation(
                                phase: .downloading,
                                progress: update.fractionCompleted,
                                downloadedGB: downloadedGB
                            )
                        }
                    }
                )
                try Task.checkCancellation()
                guard modelTaskTokens[model.id] == taskToken else { return }
                let resolvedDownloadGB = models.first(where: { $0.id == model.id })?.approximateDownloadGB
                    ?? model.approximateDownloadGB
                installations[model.id] = ModelInstallation(
                    phase: .verifying,
                    progress: 1,
                    downloadedGB: resolvedDownloadGB
                )
                _ = try HuggingFaceModelInstaller.verify(modelID: model.id, rootURL: rootURL)
                if let index = models.firstIndex(where: { $0.id == model.id }) {
                    models[index].localURL = localURL
                }
                loras = LocalModelDiscovery.discover(at: rootURL).loras
                installations[model.id] = ModelInstallation(
                    phase: .installed,
                    progress: 1,
                    downloadedGB: resolvedDownloadGB
                )
                statusMessage = "「\(model.displayName)」已下載並驗證完成。"
            } catch is CancellationError {
                guard modelTaskTokens[model.id] == taskToken else { return }
                var installation = installation(for: model.id)
                installation.phase = .paused
                installations[model.id] = installation
            } catch {
                guard modelTaskTokens[model.id] == taskToken else { return }
                var installation = installation(for: model.id)
                installation.phase = .failed
                installation.errorMessage = error.localizedDescription
                installations[model.id] = installation
                if let installerError = error as? ModelInstallerError,
                   case let .authenticationRequired(downloadURL, _) = installerError {
                    statusMessage = "Civitai 需要登入；已準備開啟下載網址。"
                    presentCivitaiAuthenticationDialog(
                        for: model,
                        downloadURL: downloadURL
                    )
                } else {
                    statusMessage = "模型下載失敗：\(error.localizedDescription)"
                }
            }
        }
    }

    private func presentCivitaiAuthenticationDialog(
        for model: ModelDescriptor,
        downloadURL: URL
    ) {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Civitai 需要登入才能下載 LoRA"
        alert.informativeText = "「\(model.displayName)」的下載網址需要 Civitai 登入或 API Token。按下「開啟下載網址」後，可在瀏覽器登入並手動下載；若要回到模型中心自動下載，請設定 CIVITAI_TOKEN 後再重試。"
        alert.addButton(withTitle: "開啟下載網址")
        alert.addButton(withTitle: "取消")
        if alert.runModal() == .alertFirstButtonReturn {
            guard NSWorkspace.shared.open(downloadURL) else {
                statusMessage = "無法開啟 Civitai 下載網址。"
                return
            }
            statusMessage = "已在預設瀏覽器開啟 Civitai 下載網址。"
        }
    }

    func pauseModel(_ model: ModelDescriptor) {
        modelTaskTokens[model.id] = nil
        modelTasks[model.id]?.cancel()
        modelTasks[model.id] = nil
        var installation = installation(for: model.id)
        installation.phase = .paused
        installations[model.id] = installation
    }

    func removeModel(_ model: ModelDescriptor) {
        guard confirmModelRemoval(model) else { return }
        modelTaskTokens[model.id] = nil
        modelTasks[model.id]?.cancel()
        modelTasks[model.id] = nil
        do {
            if HuggingFaceModelInstaller.supports(modelID: model.id) {
                let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
                try HuggingFaceModelInstaller.remove(modelID: model.id, rootURL: rootURL)
            } else if let localURL = model.localURL {
                let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
                    .resolvingSymlinksInPath()
                    .standardizedFileURL
                let targetURL = localURL
                    .resolvingSymlinksInPath()
                    .standardizedFileURL
                let rootPath = rootURL.path.hasSuffix("/") ? rootURL.path : "\(rootURL.path)/"
                guard targetURL.path != rootURL.path,
                      targetURL.path.hasPrefix(rootPath) else {
                    throw NSError(
                        domain: "GenImage.ModelRemoval",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "模型檔案不在目前的模型目錄內，為避免誤刪除已取消操作。"]
                    )
                }
                try FileManager.default.removeItem(at: targetURL)
            }
            if let index = models.firstIndex(where: { $0.id == model.id }) {
                models[index].localURL = nil
            }
            deactivateProfiles(usingModelID: model.id)
            let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
            loras = LocalModelDiscovery.discover(at: rootURL).loras
            installations[model.id] = ModelInstallation()
            statusMessage = "已移除「\(model.displayName)」。"
        } catch {
            installations[model.id] = ModelInstallation(
                phase: .failed,
                errorMessage: error.localizedDescription
            )
            statusMessage = "移除模型失敗：\(error.localizedDescription)"
        }
    }

    private func confirmModelRemoval(_ model: ModelDescriptor) -> Bool {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "確定要移除模型嗎？"
        let location = model.localURL?.path ?? model.id
        alert.informativeText = "將移除「\(model.displayName)」及其本機檔案。\n\n路徑：\(location)\n\n此操作無法復原。"
        alert.addButton(withTitle: "移除")
        alert.addButton(withTitle: "取消")
        return alert.runModal() == .alertFirstButtonReturn
    }

    func repairModel(_ model: ModelDescriptor) {
        if HuggingFaceModelInstaller.supports(modelID: model.id) {
            let rootURL = URL(fileURLWithPath: modelRootPath, isDirectory: true)
            do {
                let localURL = try HuggingFaceModelInstaller.verify(
                    modelID: model.id,
                    rootURL: rootURL
                )
                if let index = models.firstIndex(where: { $0.id == model.id }) {
                    models[index].localURL = localURL
                }
                installations[model.id] = ModelInstallation(
                    phase: .installed,
                    progress: 1,
                    downloadedGB: model.approximateDownloadGB
                )
                statusMessage = "「\(model.displayName)」驗證完成。"
            } catch {
                installations[model.id] = ModelInstallation(phase: .queued)
                deactivateProfiles(usingModelID: model.id)
                statusMessage = "偵測到缺少檔案，開始續傳修復。"
                if let index = models.firstIndex(where: { $0.id == model.id }) {
                    models[index].localURL = nil
                }
                var downloadableModel = model
                downloadableModel.localURL = nil
                installModel(downloadableModel)
            }
            return
        }
        if let localURL = model.localURL {
            if FileManager.default.fileExists(atPath: localURL.path) {
                installations[model.id] = ModelInstallation(
                    phase: .installed,
                    progress: 1,
                    downloadedGB: model.approximateDownloadGB
                )
                statusMessage = "本機模型檔案仍然存在。"
            } else {
                installations[model.id] = ModelInstallation(
                    phase: .failed,
                    errorMessage: "找不到本機模型路徑。"
                )
                statusMessage = "找不到本機模型路徑：\(localURL.path)"
            }
            return
        }
        installations[model.id] = ModelInstallation(
            phase: .failed,
            errorMessage: "此模型沒有可驗證的下載方案。"
        )
    }

    func duplicateProfile(_ profile: InferenceProfile) {
        let copy = InferenceProfile(
            name: "\(profile.name) 副本",
            capability: profile.capability,
            modelID: profile.modelID,
            modelRevision: profile.modelRevision,
            architecture: profile.architecture,
            defaults: profile.defaults,
            loras: profile.loras,
            profileRevision: 1,
            notes: profile.notes,
            isBuiltIn: false
        )
        profiles.append(copy)
        selectProfile(copy.id, for: copy.capability)
    }

    func createProfile(for capability: ModelCapability) {
        let isVideoCapability = capability == .imageToVideo || capability == .textToVideo
        let isMusicCapability = capability == .textToMusic
        let model = models.first(where: { $0.capabilities.contains(capability) })
            ?? (capability == .imageToImage
                ? models.first(where: { $0.capabilities.contains(.textToImage) })
                : nil)
        let templateProfile = profiles.first { $0.capability == capability }
        guard let modelID = model?.id ?? templateProfile?.modelID else {
            statusMessage = "找不到可作為 \(capability.title) Profile 初值的模型。"
            return
        }
        let defaults: ProfileDefaults
        if capability == .imageToImage {
            defaults = ProfileDefaults(
                width: recipe.width,
                height: recipe.height,
                steps: recipe.steps,
                outputCount: recipe.outputCount
            )
        } else if (isVideoCapability || isMusicCapability), let templateProfile {
            defaults = templateProfile.defaults
        } else {
            defaults = ProfileDefaults()
        }
        let architecture: InferenceArchitecture
        if (isVideoCapability || isMusicCapability), let templateProfile {
            architecture = templateProfile.architecture
        } else if capability == .imageToImage {
            architecture = .externalCLI
        } else {
            architecture = model?.quantization == .coreML ? .coreML : .mlxSwift
        }
        let profile = InferenceProfile(
            name: "新的 \(capability.title) Profile",
            capability: capability,
            modelID: modelID,
            architecture: architecture,
            defaults: defaults,
            loras: isVideoCapability ? (templateProfile?.loras ?? []) : [],
            notes: capability == .imageToImage || isVideoCapability || isMusicCapability
                ? "請設定支援此生成能力的模型版本與推論架構。"
                : "",
            isBuiltIn: false
        )
        profiles.append(profile)
        selectProfile(profile.id, for: capability)
    }

    func updateProfile(
        id: UUID,
        name: String,
        modelID: String,
        modelRevision: String,
        architecture: InferenceArchitecture,
        loras: [ProfileLoRAConfiguration]
    ) {
        guard let index = profiles.firstIndex(where: { $0.id == id }), !profiles[index].isBuiltIn else {
            return
        }

        guard loras.allSatisfy({ configuration in
            !configuration.modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && configuration.scale.isFinite
                && (0...1).contains(configuration.scale)
                && configuration.conditioningScale.isFinite
                && (0...1).contains(configuration.conditioningScale)
        }) else {
            statusMessage = "LoRA 模型 ID 不可空白，權重與控制強度必須介於 0 到 1。"
            return
        }

        var candidate = profiles[index]
        candidate.name = name
        candidate.modelID = modelID
        candidate.modelRevision = modelRevision
        candidate.architecture = architecture
        candidate.loras = loras
        candidate.profileRevision += 1

        // 編輯中的 Profile 尚未寫回陣列，將候選模型明確傳入檢查，避免
        // 使用中的 Profile 被更新成不存在或不相容的模型／架構。
        let candidateModel = models.first(where: { $0.id == modelID })
        if let compatibilityError = profileCompatibilityError(candidate, modelOverride: candidateModel) {
            statusMessage = compatibilityError
            return
        }

        profiles[index] = candidate
        Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)

        let capability = profiles[index].capability
        if activeProfileIDs[capability] == id {
            let isInstalled = missingProfileModels(profiles[index]).isEmpty
            if isInstalled {
                if capability == .textToImage {
                    recipe.modelID = modelID
                }
            } else {
                activeProfileIDs[capability] = nil
                if capability == .textToImage {
                    recipe.profileID = nil
                }
                statusMessage = "Profile 已更新；新模型尚未安裝，因此未設為使用中。"
            }
        }
        Self.persistActiveProfiles(activeProfileIDs, in: profiles)
    }

    func deleteProfile(_ id: UUID) {
        guard let profile = profiles.first(where: { $0.id == id }), !profile.isBuiltIn else { return }
        disabledProfileIDs.remove(id)
        profiles.removeAll { $0.id == id }
        Self.persistDisabledProfiles(disabledProfileIDs, in: profiles)

        if activeProfileIDs[profile.capability] == id {
            let replacement = profiles.first { candidate in
                candidate.capability == profile.capability
                    && !disabledProfileIDs.contains(candidate.id)
                    && missingProfileModels(candidate).isEmpty
            }
            activeProfileIDs[profile.capability] = replacement?.id
            if profile.capability == .textToImage {
                recipe.profileID = replacement?.id
                if let replacement {
                    recipe.modelID = replacement.modelID
                }
            }
        }
        Self.persistActiveProfiles(activeProfileIDs, in: profiles)
    }

    private func updateJobProgress(_ id: UUID, value: Double) {
        let normalizedValue = min(1, max(0, value))
        let now = Date()
        let lastUpdate = lastJobProgressUpdate[id] ?? .distantPast
        guard normalizedValue >= 1
            || now.timeIntervalSince(lastUpdate) >= Self.jobProgressUpdateInterval else {
            return
        }
        lastJobProgressUpdate[id] = now
        updateJob(id) { job in
            job.progress = max(job.progress, normalizedValue)
        }
        if normalizedValue >= 1 {
            lastJobProgressUpdate[id] = nil
        }
    }

    private func updateJob(_ id: UUID, change: (inout GenerationJob) -> Void) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        change(&jobs[index])
        let now = Date()
        switch jobs[index].state {
        case .running, .cancelling:
            if jobs[index].startedAt == nil {
                jobs[index].startedAt = now
            }
        case .completed, .cancelled, .failed:
            lastJobProgressUpdate[id] = nil
            if jobs[index].startedAt == nil {
                jobs[index].startedAt = jobs[index].createdAt
            }
            if jobs[index].finishedAt == nil {
                jobs[index].finishedAt = now
            }
        case .queued:
            break
        }
    }

    private func isProfileReady(_ profile: InferenceProfile) -> Bool {
        let missingModels = missingProfileModels(profile)
        guard missingModels.isEmpty else {
            statusMessage = "請先安裝 Profile 的相關模型：\(missingModels.map(\.displayName).joined(separator: "、"))。"
            return false
        }
        return true
    }

    private func missingProfileModels(_ profile: InferenceProfile) -> [ModelDescriptor] {
        profile.requiredModelIDs.compactMap { requiredModelID in
            guard let model = models.first(where: { $0.id == requiredModelID }) else {
                return ModelDescriptor(
                    id: requiredModelID,
                    displayName: requiredModelID,
                    publisher: "",
                    summary: "",
                    capabilities: [],
                    quantization: .lora,
                    approximateDownloadGB: 0,
                    recommendedMemoryGB: 0,
                    licenseName: ""
                )
            }
            guard model.localURL != nil,
                  installation(for: requiredModelID).phase == .installed else {
                return model
            }
            return nil
        }
    }

    private func resolvedVideoLoRAs(for profile: InferenceProfile) -> [VideoGenerationLoRA]? {
        var result: [VideoGenerationLoRA] = []
        for configuration in profile.loras {
            guard let model = models.first(where: { $0.id == configuration.modelID }),
                  let localURL = model.localURL,
                  FileManager.default.fileExists(atPath: localURL.path) else {
                statusMessage = "找不到 Profile LoRA 的本機檔案：\(configuration.modelID)"
                return nil
            }
            result.append(VideoGenerationLoRA(configuration: configuration, localURL: localURL))
        }
        return result
    }

    private func runtimeProfile(from profile: InferenceProfile) -> InferenceProfile? {
        guard let model = models.first(where: { $0.id == profile.modelID }) else {
            statusMessage = "找不到 Profile 指定的模型：\(profile.modelID)"
            return nil
        }
        guard let localURL = model.localURL,
              FileManager.default.fileExists(atPath: localURL.path) else {
            statusMessage = "找不到「\(model.displayName)」的本機 Runtime 路徑。"
            return nil
        }
        var executionProfile = profile
        executionProfile.modelID = localURL.path
        return executionProfile
    }
}

/// 節流下載回呼，避免大檔案下載時每個網路區塊都排入主執行緒，
/// 造成整個 WebUI 頻繁重繪而無法操作。
private final class ModelProgressGate: @unchecked Sendable {
    private let interval: TimeInterval
    private let lock = NSLock()
    private var lastReportedAt = Date.distantPast
    private var lastFraction = -1.0

    init(interval: TimeInterval) {
        self.interval = interval
    }

    func shouldEmit(_ update: ModelInstallProgress) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        let now = Date()
        let shouldEmit = update.fractionCompleted >= 1
            || update.fractionCompleted - lastFraction >= 0.01
            || now.timeIntervalSince(lastReportedAt) >= interval
        guard shouldEmit else { return false }
        lastReportedAt = now
        lastFraction = update.fractionCompleted
        return true
    }
}
