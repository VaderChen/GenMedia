import Foundation
import GenImageCore

struct WebWorkspace: Encodable {
    let id: UUID
    let name: String
    let isDefault: Bool
}

struct WebAsset: Encodable {
    let id: UUID
    let parentAssetID: UUID?
    let kind: AssetKind
    let title: String
    let fileName: String?
    let pixelWidth: Int
    let pixelHeight: Int
    let mediaDurationSeconds: Double?
    let sampleRate: Int?
    let channelCount: Int?
    let audioFormat: AudioOutputFormat?
    let subtitleFormat: SubtitleFormat?
    let languageCode: String?
    let textContent: String?
    let createdAt: Date
    let previewURL: String?
    let subtitleURL: String?
    let sidecarSubtitleFormat: SubtitleFormat?

    init(asset: MediaAsset, subtitleAssets: [MediaAsset] = []) {
        id = asset.id
        parentAssetID = asset.parentAssetID
        kind = asset.kind
        title = asset.title
        fileName = asset.fileURL?.lastPathComponent
        pixelWidth = asset.pixelWidth
        pixelHeight = asset.pixelHeight
        mediaDurationSeconds = asset.mediaDurationSeconds
        sampleRate = asset.sampleRate
        channelCount = asset.channelCount
        audioFormat = asset.audioFormat
        subtitleFormat = asset.subtitleFormat
        languageCode = asset.languageCode
        textContent = Self.subtitlePreview(for: asset)
        createdAt = asset.createdAt
        previewURL = asset.playbackURL == nil && asset.fileURL == nil
            ? nil
            : "genimage-asset://\(asset.id.uuidString)"
        let sidecar = Self.sidecar(for: asset, subtitleAssets: subtitleAssets)
        subtitleURL = sidecar == nil
            ? nil
            : "genimage-asset://\((sidecar?.assetID ?? asset.id).uuidString)/subtitle"
        sidecarSubtitleFormat = sidecar?.format
    }

    private static func sidecar(
        for asset: MediaAsset,
        subtitleAssets: [MediaAsset]
    ) -> SubtitleSidecar? {
        guard asset.kind == .importedVideo || asset.kind == .generatedVideo else {
            return nil
        }
        let mediaURLs = [asset.fileURL, asset.playbackURL].compactMap { $0 }
        let didAccess = mediaURLs.first?.startAccessingSecurityScopedResource() == true
        defer {
            if didAccess, let mediaURL = mediaURLs.first {
                mediaURL.stopAccessingSecurityScopedResource()
            }
        }
        return SubtitleSidecarResolver.locate(for: asset, among: subtitleAssets)
    }

    private static func subtitlePreview(for asset: MediaAsset) -> String? {
        guard asset.kind == .generatedSubtitle,
              let contentURL = asset.fileURL,
              let format = asset.subtitleFormat else {
            return asset.textContent
        }
        let didAccess = contentURL.startAccessingSecurityScopedResource()
        defer {
            if didAccess { contentURL.stopAccessingSecurityScopedResource() }
        }
        guard let data = try? Data(contentsOf: contentURL),
              let content = String(data: data, encoding: .utf8),
              let preview = SubtitleDocument.previewText(format: format, content: content) else {
            return asset.textContent
        }
        return preview
    }
}

struct WebModel: Encodable {
    let descriptor: ModelDescriptor
    let installation: ModelInstallation
}

struct WebOperation: Encodable {
    let id: UUID
    let action: WorkflowAction
    let inputAssetID: UUID?
    let inputAssetIDs: [UUID]?
    let outputAssetIDs: [UUID]
    let profileName: String?
    let profileRevision: Int?
}

struct WebRecipe: Encodable {
    let id: UUID
    let name: String
    let prompt: String
    let negativePrompt: String
    let modelID: String
    let profileID: UUID?
    let width: Int
    let height: Int
    let steps: Int
    let outputCount: Int
    let seed: String
    let loraID: String?
    let loraScale: Double

    init(recipe: GenerationRecipe) {
        id = recipe.id
        name = recipe.name
        prompt = recipe.prompt
        negativePrompt = recipe.negativePrompt
        modelID = recipe.modelID
        profileID = recipe.profileID
        width = recipe.width
        height = recipe.height
        steps = recipe.steps
        outputCount = recipe.outputCount
        seed = String(recipe.seed)
        loraID = recipe.lora?.adapterID
        loraScale = recipe.lora?.scale ?? 1
    }
}

struct WebVideoOutputSettings: Encodable {
    let width: Int
    let height: Int
    let steps: Int
    let outputCount: Int
    let frameCount: Int
    let frameRate: Int
    let seed: String

    init(settings: VideoOutputSettings) {
        width = settings.width
        height = settings.height
        steps = settings.steps
        outputCount = settings.outputCount
        frameCount = settings.frameCount
        frameRate = settings.frameRate
        seed = String(settings.seed)
    }
}

struct WebMusicOutputSettings: Encodable {
    let prompt: String
    let lyrics: String
    let style: MusicStyle
    let durationSeconds: Int
    let steps: Int
    let seed: String
    let format: AudioOutputFormat

    init(settings: MusicOutputSettings) {
        prompt = settings.prompt
        lyrics = settings.lyrics
        style = settings.style
        durationSeconds = settings.durationSeconds
        steps = settings.steps
        seed = String(settings.seed)
        format = settings.format
    }
}

struct WebMCPServiceState: Encodable {
    let isEnabled: Bool
    let isRunning: Bool
    let endpointURL: String?
    let errorMessage: String?

    @MainActor
    init(service: LocalMCPServiceController) {
        isEnabled = service.isEnabled
        isRunning = service.isRunning
        endpointURL = service.endpointURL
        errorMessage = service.errorMessage
    }
}

struct WebAppState: Encodable {
    let schemaVersion: Int
    let projectName: String
    let workspaces: [WebWorkspace]
    let selectedWorkspaceID: UUID
    let modelRootPath: String
    let outputDirectoryPath: String
    let civitaiTokenConfigured: Bool
    let huggingFaceTokenConfigured: Bool
    let assets: [WebAsset]
    let selectedAssetID: UUID?
    let comparisonAssetID: UUID?
    let recipe: WebRecipe
    let videoOutputSettings: WebVideoOutputSettings
    let musicOutputSettings: WebMusicOutputSettings
    let jobs: [GenerationJob]
    let models: [WebModel]
    let loras: [LoRADescriptor]
    let profiles: [InferenceProfile]
    let disabledProfileIDs: [UUID]
    let activeProfileIDs: [String: UUID]
    let operations: [WebOperation]
    let statusMessage: String?
    let availableUpdate: AppUpdateInfo?
    let systemMetrics: SystemMetricsSnapshot
    let isReleasingMemory: Bool
    let mcpService: WebMCPServiceState

    @MainActor
    init(store: AppStore, mcpService: LocalMCPServiceController) {
        schemaVersion = 1
        projectName = store.selectedProject?.name ?? "工作區"
        workspaces = store.projects.enumerated().map { index, project in
            WebWorkspace(id: project.id, name: project.name, isDefault: index == 0)
        }
        selectedWorkspaceID = store.selectedProjectID
        modelRootPath = store.modelRootPath
        outputDirectoryPath = store.outputDirectoryPath
        civitaiTokenConfigured = CivitaiTokenStore.isConfigured()
        huggingFaceTokenConfigured = HuggingFaceTokenStore.isConfigured()
        let projectAssets = store.projectAssets
        assets = projectAssets.map { asset in
            WebAsset(asset: asset, subtitleAssets: projectAssets)
        }
        selectedAssetID = store.selectedAssetID
        comparisonAssetID = store.comparisonAssetID
        recipe = WebRecipe(recipe: store.recipe)
        videoOutputSettings = WebVideoOutputSettings(settings: store.videoOutputSettings)
        musicOutputSettings = WebMusicOutputSettings(settings: store.musicOutputSettings)
        jobs = store.jobs
        models = store.models.map {
            WebModel(descriptor: $0, installation: store.installation(for: $0.id))
        }
        loras = store.loras
        profiles = store.profiles
        disabledProfileIDs = store.disabledProfileIDs.sorted {
            $0.uuidString < $1.uuidString
        }
        activeProfileIDs = Dictionary(
            uniqueKeysWithValues: store.activeProfileIDs.map { ($0.key.rawValue, $0.value) }
        )
        operations = store.projectOperations.map {
            WebOperation(
                id: $0.id,
                action: $0.action,
                inputAssetID: $0.inputAssetID,
                inputAssetIDs: $0.inputAssetIDs,
                outputAssetIDs: $0.outputAssetIDs,
                profileName: $0.profileSnapshot?.name,
                profileRevision: $0.profileSnapshot?.profileRevision
            )
        }
        statusMessage = store.statusMessage
        availableUpdate = store.availableUpdate
        systemMetrics = store.systemMetrics
        isReleasingMemory = store.isReleasingMemory
        self.mcpService = WebMCPServiceState(service: mcpService)
    }
}
