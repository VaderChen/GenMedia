import Foundation
import GenImageCore

struct WebAsset: Encodable {
    let id: UUID
    let parentAssetID: UUID?
    let kind: AssetKind
    let title: String
    let pixelWidth: Int
    let pixelHeight: Int
    let createdAt: Date
    let previewURL: String?

    init(asset: ImageAsset) {
        id = asset.id
        parentAssetID = asset.parentAssetID
        kind = asset.kind
        title = asset.title
        pixelWidth = asset.pixelWidth
        pixelHeight = asset.pixelHeight
        createdAt = asset.createdAt
        previewURL = asset.fileURL == nil ? nil : "genimage-asset://\(asset.id.uuidString)"
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

struct WebAppState: Encodable {
    let schemaVersion: Int
    let projectName: String
    let modelRootPath: String
    let outputDirectoryPath: String
    let assets: [WebAsset]
    let selectedAssetID: UUID?
    let comparisonAssetID: UUID?
    let recipe: WebRecipe
    let videoOutputSettings: WebVideoOutputSettings
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

    @MainActor
    init(store: AppStore) {
        schemaVersion = 1
        projectName = store.selectedProject?.name ?? "工作區"
        modelRootPath = store.modelRootPath
        outputDirectoryPath = store.outputDirectoryPath
        assets = store.projectAssets.map(WebAsset.init)
        selectedAssetID = store.selectedAssetID
        comparisonAssetID = store.comparisonAssetID
        recipe = WebRecipe(recipe: store.recipe)
        videoOutputSettings = WebVideoOutputSettings(settings: store.videoOutputSettings)
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
        operations = store.operations.map {
            WebOperation(
                id: $0.id,
                action: $0.action,
                inputAssetID: $0.inputAssetID,
                outputAssetIDs: $0.outputAssetIDs,
                profileName: $0.profileSnapshot?.name,
                profileRevision: $0.profileSnapshot?.profileRevision
            )
        }
        statusMessage = store.statusMessage
        availableUpdate = store.availableUpdate
        systemMetrics = store.systemMetrics
        isReleasingMemory = store.isReleasingMemory
    }
}
