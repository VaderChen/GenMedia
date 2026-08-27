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
        steps: Int = 20,
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
            steps: try container.decodeIfPresent(Int.self, forKey: .steps) ?? 20,
            seed: try container.decodeIfPresent(UInt64.self, forKey: .seed) ?? 42,
            format: try container.decodeIfPresent(AudioOutputFormat.self, forKey: .format) ?? .mp3
        )
    }
}

/// WebUI 背後的單一狀態容器。
///
/// 這個檔案只保留型別宣告、儲存屬性與 init；其餘行為依職責拆進 `AppStore+*.swift`：
///
///   Persistence        設定的讀寫與還原驗證
///   Workspaces         工作區的建立與切換
///   Paths              模型根目錄、輸出目錄與系統資源監看
///   Selection          由選取狀態衍生的唯讀資料
///   Profiles           Profile 的選用、相容性檢查與增刪改
///   OutputSettings     圖片、影片與音樂的輸出參數
///   Assets             工作區資產的匯入、選取與移除
///   ImageGeneration    圖生文、文生圖、圖生圖與 Upscale
///   MediaGeneration    影片與音樂生成
///   SubtitleGeneration 多媒體語音辨識與字幕輸出
///   Jobs               生成工作的進度、取消與記憶體釋放
///   ModelInstallation  模型的安裝、暫停、移除與修復
///
/// 沒有標記 `private` 的成員，代表它被上述某個擴充檔案使用；Swift 的 `private` 只到檔案範圍，
/// 所以跨檔案共用的部分必須是 internal（仍侷限於 GenImageApp 這個 target）。
@MainActor
final class AppStore: ObservableObject {
    @Published var selectedSection: AppSection = .workspace
    @Published var projects: [Project] {
        didSet { persistProjectWorkspace() }
    }
    @Published var selectedProjectID: UUID {
        didSet { persistProjectWorkspace() }
    }
    @Published var assets: [MediaAsset] {
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

    var jobTasks: [UUID: Task<Void, Never>] = [:]
    var cancellationRequestedJobIDs: Set<UUID> = []
    var lastJobProgressUpdate: [UUID: Date] = [:]
    var modelTasks: [String: Task<Void, Never>] = [:]
    var modelTaskTokens: [String: UUID] = [:]
    var systemMetricsTask: Task<Void, Never>?
    private var updateCheckTask: Task<Void, Never>?
    var textToImageService: ZImageTextToImageService
    let imageToTextService: QwenVLImageDescriptionService
    var imageToImageService: Qwen2511ImageToImageService
    var upscaleService: CoreMLUpscaleService
    var videoGenerationService: LTXVideoGenerationService
    var musicGenerationService: MusicGenerationRouter
    var subtitleGenerationService: SubtitleGenerationRouter
    var mediaCompositionService: MediaCompositionService
    let modelInstaller = HuggingFaceModelInstaller()
    private let projectWorkspaceURL: URL
    private var projectWorkspacePersistenceEnabled = false

    static let recipeSettingsKey = "GenImage.recipeSettings.v1"
    static let videoOutputSettingsKey = "GenImage.videoOutputSettings.v1"
    static let musicOutputSettingsKey = "GenImage.musicOutputSettings.v1"
    static let modelRootKey = "GenImage.modelRootPath.v1"
    static let outputDirectoryKey = "GenImage.outputDirectoryPath.v1"
    static let disabledProfilesKey = "GenImage.disabledProfiles.v1"
    static let activeProfilesKey = "GenImage.activeProfiles.v1"
    static let jobProgressUpdateInterval: TimeInterval = 1
    static let modelProgressUpdateInterval: TimeInterval = 0.5

    struct PersistedRecipeSettings: Codable {
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
        // 工作區索引曾經寫在另一個根目錄下，先接回來再讀取，否則升級後會看不到既有的專案。
        ApplicationSupport.adoptLegacyDirectories()
        projectWorkspaceURL = ProjectWorkspacePersistence.defaultURL()
        let restoredWorkspace = try? ProjectWorkspacePersistence.load(from: projectWorkspaceURL)
        let defaultGeneratedDirectory = ApplicationSupport.directory(.generated)
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
        subtitleGenerationService = SubtitleGenerationRouter(outputDirectory: generatedDirectory)
        mediaCompositionService = MediaCompositionService(outputDirectory: generatedDirectory)
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
        let referencedAssetIDs = Set(restoredAssets.map(\.id))
        for orphanURL in ApplicationSupport.orphanMediaCacheFiles(
            in: ApplicationSupport.directory(.mediaCache),
            referencedAssetIDs: referencedAssetIDs
        ) {
            try? FileManager.default.removeItem(at: orphanURL)
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

    static func makeMusicGenerationService(
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
















}
