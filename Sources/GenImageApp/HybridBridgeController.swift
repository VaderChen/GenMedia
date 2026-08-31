import AppKit
import Combine
import Foundation
import GenImageCore
import GenImageRuntime
import ImageIO
import UniformTypeIdentifiers
import WebKit

private enum SourceImageAction {
    case describe
    case imageToImage
    case imageToVideo
    case upscale
}

@MainActor
final class HybridBridgeController: NSObject, ObservableObject {
    let store = AppStore()
    let mcpService = LocalMCPServiceController()
    let assetSchemeHandler = AssetSchemeHandler()
    let webUISchemeHandler = WebUISchemeHandler()

    private weak var webView: WKWebView?
    private var storeCancellable: AnyCancellable?
    private var mcpServiceCancellable: AnyCancellable?
    private var pendingPush: Task<Void, Never>?
    private var pasteKeyMonitor: Any?
    private var sharingPicker: NSSharingServicePicker?
    private var pageReady = false

    override init() {
        super.init()
        storeCancellable = store.objectWillChange.sink { [weak self] _ in
            Task { @MainActor in
                self?.scheduleStatePush()
            }
        }
        mcpServiceCancellable = mcpService.objectWillChange.sink { [weak self] _ in
            Task { @MainActor in
                self?.scheduleStatePush()
            }
        }
    }

    func attach(webView: WKWebView) {
        self.webView = webView
        if let pasteKeyMonitor {
            NSEvent.removeMonitor(pasteKeyMonitor)
        }
        pasteKeyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) {
            [weak self, weak webView] event in
            guard let webView,
                  event.window === webView.window,
                  Self.isPasteShortcut(event),
                  self?.pasteImageFromSystemClipboard() == true else {
                return event
            }
            return nil
        }
    }

    func pasteImageFromSystemClipboard() -> Bool {
        guard pageReady,
              let webView,
              let payload = Self.clipboardImagePayload(from: .general),
              let data = try? JSONEncoder().encode(payload),
              let json = String(data: data, encoding: .utf8) else {
            return false
        }
        webView.evaluateJavaScript("window.GenImageNative?.receiveClipboardImage(\(json));")
        return true
    }

    private static func isPasteShortcut(_ event: NSEvent) -> Bool {
        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        return event.charactersIgnoringModifiers?.lowercased() == "v"
            && (modifiers.contains(.command) || modifiers.contains(.control))
    }

    func pushState() {
        guard pageReady, let webView else { return }
        assetSchemeHandler.updateAssets(store.assets)

        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(WebAppState(store: store, mcpService: mcpService))
            guard let json = String(data: data, encoding: .utf8) else { return }
            webView.evaluateJavaScript("window.GenImageNative?.receiveState(\(json));")
        } catch {
            sendError(id: nil, message: "無法同步應用程式狀態：\(error.localizedDescription)")
        }
    }

    private func scheduleStatePush() {
        pendingPush?.cancel()
        pendingPush = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(35))
            if Task.isCancelled { return }
            self?.pushState()
        }
    }

    private func handle(method: String, params: [String: Any]) throws {
        switch method {
        case "bootstrap":
            pushState()

        case "selectAsset":
            guard let id = uuid(params["assetID"]) else { throw BridgeError.invalidParameters }
            store.selectAsset(id)

        case "removeAsset":
            guard let id = uuid(params["assetID"]) else { throw BridgeError.invalidParameters }
            store.removeAsset(
                id,
                selecting: uuid(params["replacementAssetID"]),
                deleteFile: params["deleteFile"] as? Bool ?? false
            )

        case "closeWorkspaceProject":
            guard let rawAssetIDs = params["assetIDs"] as? [String] else {
                throw BridgeError.invalidParameters
            }
            store.closeWorkspaceProject(
                assetIDs: rawAssetIDs.compactMap(UUID.init(uuidString:))
            )

        case "createWorkspace":
            guard let name = params["name"] as? String else {
                throw BridgeError.invalidParameters
            }
            store.createWorkspace(name: name)

        case "deleteWorkspace":
            guard let id = uuid(params["workspaceID"]) else {
                throw BridgeError.invalidParameters
            }
            store.deleteWorkspace(id)

        case "selectWorkspace":
            guard let id = uuid(params["workspaceID"]) else {
                throw BridgeError.invalidParameters
            }
            store.selectWorkspace(id)

        case "openAsset":
            try openAsset(params)

        case "revealAsset":
            try revealAsset(params)

        case "downloadAsset":
            try downloadAsset(params)

        case "renameAsset":
            try renameAsset(params)

        case "copyAsset":
            try copyAsset(params)

        case "shareAsset":
            try shareAsset(params)

        case "selectProfile":
            guard let id = uuid(params["profileID"]),
                  let capabilityRaw = params["capability"] as? String,
                  let capability = ModelCapability(rawValue: capabilityRaw) else {
                throw BridgeError.invalidParameters
            }
            store.selectProfile(id, for: capability)

        case "deactivateProfile":
            guard let id = uuid(params["profileID"]),
                  let capabilityRaw = params["capability"] as? String,
                  let capability = ModelCapability(rawValue: capabilityRaw) else {
                throw BridgeError.invalidParameters
            }
            store.deactivateProfile(id, for: capability)

        case "applyProfileDefaults":
            if params["outputKind"] as? String == "music" {
                store.applyActiveMusicProfileDefaults()
            } else if params["outputKind"] as? String == "video" {
                store.applyActiveVideoProfileDefaults()
            } else {
                store.applyActiveGenerationProfileDefaults()
            }

        case "updateRecipe":
            try updateRecipe(params)

        case "updateVideoOutputSettings":
            updateVideoOutputSettings(params)

        case "updateMusicOutputSettings":
            try updateMusicOutputSettings(params)

        case "applyWorkspaceTabDraft":
            try applyWorkspaceTabDraft(params)

        case "randomizeSeed":
            store.randomizeSeed()

        case "randomizeVideoSeed":
            store.randomizeVideoSeed()

        case "randomizeMusicSeed":
            store.randomizeMusicSeed()

        case "generate":
            store.generate(linkToSelectedAsset: params["linkToSelectedAsset"] as? Bool ?? false)

        case "generateVideo":
            let sourceAssetIDs = (params["sourceAssetIDs"] as? [String] ?? [])
                .compactMap { UUID(uuidString: $0) }
            let hasImageToVideo = store.activeProfile(for: .imageToVideo) != nil
            let hasTextToVideo = store.activeProfile(for: .textToVideo) != nil
            if sourceAssetIDs.isEmpty,
               store.selectedSourceImage == nil,
               hasImageToVideo,
               !hasTextToVideo {
                performSourceImageAction(.imageToVideo)
            } else {
                store.requestVideoGeneration(sourceAssetIDs: sourceAssetIDs)
            }

        case "generateMusic":
            store.requestMusicGeneration()

        case "generateSubtitles":
            guard let rawFormat = params["format"] as? String,
                  let format = SubtitleFormat(rawValue: rawFormat) else {
                throw BridgeError.invalidParameters
            }
            let targetLanguage: SubtitleTranslationLanguage?
            if let rawTargetLanguage = params["targetLanguageCode"] as? String,
               !rawTargetLanguage.isEmpty {
                guard let parsedTargetLanguage = SubtitleTranslationLanguage(
                    rawValue: rawTargetLanguage
                ) else {
                    throw BridgeError.invalidParameters
                }
                targetLanguage = parsedTargetLanguage
            } else {
                targetLanguage = nil
            }
            performSubtitleGeneration(
                sourceAssetID: uuid(params["sourceAssetID"]),
                format: format,
                targetLanguage: targetLanguage
            )

        case "createImageLoop":
            guard let rawAssetIDs = params["sourceAssetIDs"] as? [String],
                  let fitModeRaw = params["fitMode"] as? String,
                  let fitMode = ImageLoopFitMode(rawValue: fitModeRaw),
                  let width = integer(params["width"]),
                  let height = integer(params["height"]),
                  let frameRate = integer(params["frameRate"]),
                  let imageDurationSeconds = double(params["imageDurationSeconds"]),
                  let totalDurationSeconds = double(params["totalDurationSeconds"]) else {
                throw BridgeError.invalidParameters
            }
            store.requestImageLoop(
                sourceAssetIDs: rawAssetIDs.compactMap(UUID.init(uuidString:)),
                options: ImageLoopOptions(
                    width: width,
                    height: height,
                    frameRate: frameRate,
                    imageDurationSeconds: imageDurationSeconds,
                    totalDurationSeconds: totalDurationSeconds,
                    fitMode: fitMode
                )
            )

        case "mergeMedia":
            guard let videoAssetID = uuid(params["videoAssetID"]),
                  let audioAssetID = uuid(params["audioAssetID"]),
                  let audioModeRaw = params["audioMode"] as? String,
                  let audioMode = MediaMergeAudioMode(rawValue: audioModeRaw),
                  let durationModeRaw = params["durationMode"] as? String,
                  let durationMode = MediaMergeDurationMode(rawValue: durationModeRaw),
                  let audioVolume = double(params["audioVolume"]) else {
                throw BridgeError.invalidParameters
            }
            store.requestMediaMerge(
                videoAssetID: videoAssetID,
                audioAssetID: audioAssetID,
                options: MediaMergeOptions(
                    audioMode: audioMode,
                    durationMode: durationMode,
                    audioVolume: audioVolume
                )
            )

        case "describe":
            performSourceImageAction(.describe)

        case "upscale":
            performSourceImageAction(.upscale)

        case "imageToImage":
            performSourceImageAction(.imageToImage)

        case "importImage":
            importImage(then: nil)

        case "importSubtitleMedia":
            importMedia(requiresAudio: true, then: nil)

        case "importMedia":
            importMedia(requiresAudio: false, then: nil)

        case "pasteImage":
            try importPastedImage(params)

        case "cancelJob":
            guard let id = uuid(params["jobID"]) else { throw BridgeError.invalidParameters }
            store.cancelJob(id)

        case "releaseMemory":
            store.releaseMemory()

        case "clearJobs":
            store.clearFinishedJobs()

        case "setModelRoot":
            guard let path = params["path"] as? String else {
                throw BridgeError.invalidParameters
            }
            store.setModelRoot(path)

        case "chooseModelRoot":
            chooseModelRoot()

        case "setOutputDirectory":
            guard let path = params["path"] as? String else {
                throw BridgeError.invalidParameters
            }
            store.setOutputDirectory(path)

        case "chooseOutputDirectory":
            chooseOutputDirectory()

        case "setCivitaiToken":
            guard let token = params["token"] as? String,
                  store.setCivitaiToken(token) else {
                throw BridgeError.assetActionFailed("無法儲存 Civitai API Token。")
            }

        case "clearCivitaiToken":
            guard store.clearCivitaiToken() else {
                throw BridgeError.assetActionFailed("無法清除 Civitai API Token。")
            }

        case "setHuggingFaceToken":
            guard let token = params["token"] as? String,
                  store.setHuggingFaceToken(token) else {
                throw BridgeError.assetActionFailed("無法儲存 Hugging Face API Token。")
            }

        case "clearHuggingFaceToken":
            guard store.clearHuggingFaceToken() else {
                throw BridgeError.assetActionFailed("無法清除 Hugging Face API Token。")
            }

        case "setMCPServiceEnabled":
            guard let enabled = params["enabled"] as? Bool else {
                throw BridgeError.invalidParameters
            }
            mcpService.setEnabled(enabled)

        case "revealOutputDirectory":
            store.revealOutputDirectory()

        case "installModel":
            guard let model = model(from: params) else { throw BridgeError.invalidParameters }
            store.installModel(
                model,
                civitaiToken: params["civitaiToken"] as? String,
                huggingFaceToken: params["huggingFaceToken"] as? String
            )

        case "installProfileModels":
            guard let id = uuid(params["profileID"]) else {
                throw BridgeError.invalidParameters
            }
            store.installProfileModels(
                id,
                civitaiToken: params["civitaiToken"] as? String,
                huggingFaceToken: params["huggingFaceToken"] as? String
            )

        case "pauseModel":
            guard let model = model(from: params) else { throw BridgeError.invalidParameters }
            store.pauseModel(model)

        case "removeModel":
            guard let model = model(from: params) else { throw BridgeError.invalidParameters }
            store.removeModel(model)

        case "repairModel":
            guard let model = model(from: params) else { throw BridgeError.invalidParameters }
            store.repairModel(
                model,
                civitaiToken: params["civitaiToken"] as? String,
                huggingFaceToken: params["huggingFaceToken"] as? String
            )

        case "duplicateProfile":
            guard let id = uuid(params["profileID"]),
                  let profile = store.profiles.first(where: { $0.id == id }) else {
                throw BridgeError.invalidParameters
            }
            store.duplicateProfile(profile)

        case "createProfile":
            guard let rawValue = params["capability"] as? String,
                  let capability = ModelCapability(rawValue: rawValue) else {
                throw BridgeError.invalidParameters
            }
            store.createProfile(for: capability)

        case "updateProfile":
            guard let id = uuid(params["profileID"]),
                  let name = params["name"] as? String,
                  let modelID = params["modelID"] as? String,
                  let revision = params["modelRevision"] as? String,
                  let architectureRaw = params["architecture"] as? String,
                  let architecture = InferenceArchitecture(rawValue: architectureRaw),
                  let loraValues = params["loras"] as? [[String: Any]] else {
                throw BridgeError.invalidParameters
            }
            let loras = try loraValues.map { value -> ProfileLoRAConfiguration in
                guard let modelID = value["modelID"] as? String,
                      let scale = double(value["scale"]),
                      let conditioningScale = double(value["conditioningScale"]) else {
                    throw BridgeError.invalidParameters
                }
                let conditioning: ProfileLoRAConditioning?
                if let rawConditioning = value["conditioning"] as? String,
                   !rawConditioning.isEmpty {
                    guard let parsed = ProfileLoRAConditioning(rawValue: rawConditioning) else {
                        throw BridgeError.invalidParameters
                    }
                    conditioning = parsed
                } else {
                    conditioning = nil
                }
                return ProfileLoRAConfiguration(
                    modelID: modelID,
                    scale: scale,
                    conditioning: conditioning,
                    conditioningScale: conditioningScale
                )
            }
            store.updateProfile(
                id: id,
                name: name,
                modelID: modelID,
                modelRevision: revision,
                architecture: architecture,
                loras: loras
            )

        case "deleteProfile":
            guard let id = uuid(params["profileID"]) else { throw BridgeError.invalidParameters }
            store.deleteProfile(id)

        case "openAvailableUpdate":
            guard let update = store.availableUpdate,
                  NSWorkspace.shared.open(update.releaseURL) else {
                throw BridgeError.assetActionFailed("無法開啟 GitHub Release 頁面。")
            }

        case "dismissAvailableUpdate":
            store.dismissAvailableUpdate()

        case "clearStatus":
            store.statusMessage = nil

        default:
            throw BridgeError.unknownMethod(method)
        }
    }

    private func updateRecipe(_ params: [String: Any]) throws {
        if let prompt = params["prompt"] as? String { store.recipe.prompt = prompt }
        if let negative = params["negativePrompt"] as? String { store.recipe.negativePrompt = negative }
        if let value = integer(params["width"]) { store.recipe.width = min(max(value, 64), 4096) }
        if let value = integer(params["height"]) { store.recipe.height = min(max(value, 64), 4096) }
        if let value = integer(params["steps"]) { store.recipe.steps = min(max(value, 1), 100) }
        if let value = integer(params["outputCount"]) { store.recipe.outputCount = min(max(value, 1), 8) }
        if let seed = params["seed"] as? String, let value = UInt64(seed) { store.recipe.seed = value }
        if params.keys.contains("loraID") {
            let loraID = (params["loraID"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if loraID.isEmpty {
                store.recipe.lora = nil
            } else {
                guard let descriptor = store.loras.first(where: { $0.id == loraID }) else {
                    throw BridgeError.invalidParameters
                }
                if let compatibilityError = store.loraCompatibilityError(at: descriptor.localURL) {
                    throw BridgeError.assetActionFailed("LoRA 相容性檢查失敗：\(compatibilityError)")
                }
                let requestedScale = double(params["loraScale"]) ?? store.recipe.lora?.scale ?? 1
                guard requestedScale.isFinite, (0...1).contains(requestedScale) else {
                    throw BridgeError.invalidParameters
                }
                store.recipe.lora = LoRASelection(
                    adapterID: descriptor.id,
                    localURL: descriptor.localURL,
                    scale: requestedScale
                )
                store.statusMessage = "LoRA 已通過基本相容性檢查。"
            }
        } else if let requestedScale = double(params["loraScale"]), var selection = store.recipe.lora {
            guard requestedScale.isFinite, (0...1).contains(requestedScale) else {
                throw BridgeError.invalidParameters
            }
            selection.scale = requestedScale
            store.recipe.lora = selection
        }
    }

    private func updateVideoOutputSettings(_ params: [String: Any]) {
        let seed = (params["seed"] as? String).flatMap(UInt64.init)
        store.updateVideoOutputSettings(
            width: integer(params["width"]),
            height: integer(params["height"]),
            steps: integer(params["steps"]),
            outputCount: integer(params["outputCount"]),
            frameCount: integer(params["frameCount"]),
            frameRate: integer(params["frameRate"]),
            seed: seed
        )
    }

    private func updateMusicOutputSettings(_ params: [String: Any]) throws {
        let seed = (params["seed"] as? String).flatMap(UInt64.init)
        let style: MusicStyle?
        if let rawStyle = params["style"] as? String {
            guard let parsedStyle = MusicStyle(rawValue: rawStyle) else {
                throw BridgeError.invalidParameters
            }
            style = parsedStyle
        } else {
            style = nil
        }
        let format: AudioOutputFormat?
        if let rawFormat = params["format"] as? String {
            guard let parsedFormat = AudioOutputFormat(rawValue: rawFormat) else {
                throw BridgeError.invalidParameters
            }
            format = parsedFormat
        } else {
            format = nil
        }
        store.updateMusicOutputSettings(
            prompt: params["prompt"] as? String,
            lyrics: params["lyrics"] as? String,
            style: style,
            durationSeconds: integer(params["durationSeconds"]),
            steps: integer(params["steps"]),
            seed: seed,
            format: format
        )
    }

    private func applyWorkspaceTabDraft(_ params: [String: Any]) throws {
        if let assignments = params["profileAssignments"] as? [[String: Any]] {
            for assignment in assignments {
                guard let capabilityRaw = assignment["capability"] as? String,
                      let capability = ModelCapability(rawValue: capabilityRaw) else { continue }
                let profileID = uuid(assignment["profileID"])
                let modelID = assignment["modelID"] as? String
                let modelRevision = assignment["modelRevision"] as? String
                let architecture = (assignment["architecture"] as? String)
                    .flatMap(InferenceArchitecture.init(rawValue:))
                let profile = store.profiles.first { profile in
                    if let profileID, profile.id == profileID, profile.capability == capability {
                        return true
                    }
                    return profile.capability == capability
                        && profile.modelID == modelID
                        && profile.modelRevision == modelRevision
                        && (architecture == nil || profile.architecture == architecture)
                }
                if let profile {
                    store.selectProfile(profile.id, for: capability)
                }
            }
        }
        if let recipe = params["recipe"] as? [String: Any] {
            try updateRecipe(recipe)
        }
        if let video = params["videoOutputSettings"] as? [String: Any] {
            updateVideoOutputSettings(video)
        }
        if let music = params["musicOutputSettings"] as? [String: Any] {
            try updateMusicOutputSettings(music)
        }
    }

    private func performSourceImageAction(_ action: SourceImageAction) {
        guard store.selectedSourceImage != nil else {
            importImage(then: action)
            return
        }

        switch action {
        case .describe: store.describeSelected()
        case .imageToImage: store.imageToImageSelected()
        case .imageToVideo: store.requestVideoGeneration()
        case .upscale: store.upscaleSelected()
        }
    }

    private func performSubtitleGeneration(
        sourceAssetID: UUID?,
        format: SubtitleFormat,
        targetLanguage: SubtitleTranslationLanguage?
    ) {
        if let sourceAssetID,
           store.subtitleSourceAsset(id: sourceAssetID) != nil {
            store.requestSubtitleGeneration(
                sourceAssetID: sourceAssetID,
                format: format,
                targetLanguage: targetLanguage
            )
            return
        }
        if let selectedSource = store.selectedSubtitleSource {
            store.requestSubtitleGeneration(
                sourceAssetID: selectedSource.id,
                format: format,
                targetLanguage: targetLanguage
            )
            return
        }
        importMedia(requiresAudio: true) { [weak self] assetID in
            self?.store.requestSubtitleGeneration(
                sourceAssetID: assetID,
                format: format,
                targetLanguage: targetLanguage
            )
        }
    }

    private func importImage(then action: SourceImageAction?) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        switch action {
        case nil, .imageToVideo:
            panel.allowsMultipleSelection = true
        default:
            panel.allowsMultipleSelection = false
        }
        panel.canChooseDirectories = false
        panel.message = panel.allowsMultipleSelection
            ? "選擇一張或多張要加入 GenImage 工作區的圖片"
            : "選擇要加入 GenImage 工作區的圖片"

        guard panel.runModal() == .OK else {
            return
        }
        var importedAssetIDs: [UUID] = []
        for url in panel.urls {
            guard let image = NSImage(contentsOf: url) else { continue }
            let representation = image.representations.max {
                $0.pixelsWide * $0.pixelsHigh < $1.pixelsWide * $1.pixelsHigh
            }
            let assetID = store.importImage(
                url: url,
                pixelWidth: representation?.pixelsWide ?? Int(image.size.width),
                pixelHeight: representation?.pixelsHigh ?? Int(image.size.height)
            )
            importedAssetIDs.append(assetID)
        }
        guard !importedAssetIDs.isEmpty else { return }

        switch action {
        case .describe: store.describeSelected()
        case .imageToImage: store.imageToImageSelected()
        case .imageToVideo:
            store.requestVideoGeneration(sourceAssetIDs: importedAssetIDs)
        case .upscale: store.upscaleSelected()
        default: break
        }
    }

    private func importMedia(
        requiresAudio: Bool,
        then completion: (@MainActor @Sendable (UUID) -> Void)?
    ) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = Self.mediaImportContentTypes
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = requiresAudio
            ? "選擇包含聲音的影片或音訊檔案"
            : "選擇要加入工作區的影片或音訊檔案"
        panel.prompt = "匯入"

        guard panel.runModal() == .OK, let url = panel.url else { return }
        store.requestMediaImport(
            sourceURL: url,
            requiresAudio: requiresAudio,
            completion: completion
        )
    }

    private static let mediaImportContentTypes: [UTType] = {
        let extensions = [
            "mp4", "m4v", "mov", "mkv", "webm", "avi", "wmv", "flv",
            "mpeg", "mpg", "m2ts", "mts", "ts", "3gp", "ogv", "rm", "rmvb", "vob", "mxf",
            "mp3", "m4a", "aac", "wav", "wave", "aif", "aiff", "flac", "ogg", "oga", "opus",
            "wma", "ape", "wv", "alac", "ac3", "eac3", "amr", "mka", "dsf", "dff"
        ]
        return ([UTType.movie, .audio] + extensions.compactMap {
            UTType(filenameExtension: $0)
        }).reduce(into: [UTType]()) { result, contentType in
            if !result.contains(contentType) {
                result.append(contentType)
            }
        }
    }()

    private func chooseModelRoot() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.directoryURL = URL(fileURLWithPath: store.modelRootPath, isDirectory: true)
        panel.message = "選擇 GenImage 預設模型目錄"
        panel.prompt = "選擇目錄"

        guard panel.runModal() == .OK, let url = panel.url else { return }
        store.setModelRoot(url.path)
    }

    private func chooseOutputDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.directoryURL = URL(fileURLWithPath: store.outputDirectoryPath, isDirectory: true)
        panel.message = "選擇圖片與影片輸出目錄"
        panel.prompt = "選擇目錄"

        guard panel.runModal() == .OK, let url = panel.url else { return }
        store.setOutputDirectory(url.path)
    }

    private func assetURL(from params: [String: Any]) throws -> URL {
        guard let id = uuid(params["assetID"]),
              let asset = store.assets.first(where: { $0.id == id }),
              let url = asset.fileURL,
              FileManager.default.fileExists(atPath: url.path) else {
            throw BridgeError.invalidParameters
        }
        return url
    }

    private func openAsset(_ params: [String: Any]) throws {
        let url = try assetURL(from: params)
        guard NSWorkspace.shared.open(url) else {
            throw BridgeError.assetActionFailed("無法開啟媒體檔案。")
        }
    }

    private func revealAsset(_ params: [String: Any]) throws {
        let url = try assetURL(from: params)
        NSWorkspace.shared.activateFileViewerSelecting([url])
        store.statusMessage = "已在 Finder 中開啟檔案所在目錄。"
    }

    private func downloadAsset(_ params: [String: Any]) throws {
        let sourceURL = try assetURL(from: params)
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [
            UTType(filenameExtension: sourceURL.pathExtension) ?? .png
        ]
        panel.nameFieldStringValue = sourceURL.lastPathComponent
        panel.message = "將媒體檔案儲存到"
        guard panel.runModal() == .OK, let destinationURL = panel.url else { return }
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            try FileManager.default.removeItem(at: destinationURL)
        }
        try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
        store.statusMessage = "媒體檔案已儲存至：\(destinationURL.path)"
    }

    private func renameAsset(_ params: [String: Any]) throws {
        guard let id = uuid(params["assetID"]),
              let fileName = params["fileName"] as? String else {
            throw BridgeError.invalidParameters
        }
        try store.renameAsset(id, toFileName: fileName)
    }

    private func copyAsset(_ params: [String: Any]) throws {
        let sourceURL = try assetURL(from: params)
        let data: Data
        if sourceURL.pathExtension.lowercased() == "png" {
            data = try Data(contentsOf: sourceURL)
        } else {
            guard let image = NSImage(contentsOf: sourceURL), let pngData = Self.pngData(from: image) else {
                throw BridgeError.assetActionFailed("無法轉換圖片格式。")
            }
            data = pngData
        }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        guard pasteboard.setData(data, forType: .png) else {
            throw BridgeError.assetActionFailed("無法寫入系統剪貼簿。")
        }
        store.statusMessage = "圖片已複製到剪貼簿。"
    }

    private func shareAsset(_ params: [String: Any]) throws {
        let url = try assetURL(from: params)
        guard let webView else { throw BridgeError.assetActionFailed("分享介面尚未準備完成。") }
        sharingPicker = NSSharingServicePicker(items: [url])
        sharingPicker?.show(relativeTo: webView.bounds, of: webView, preferredEdge: .minY)
    }

    private func importPastedImage(_ params: [String: Any]) throws {
        guard let dataURL = params["dataURL"] as? String,
              let comma = dataURL.firstIndex(of: ","),
              dataURL[..<comma].contains(";base64"),
              let data = Data(base64Encoded: String(dataURL[dataURL.index(after: comma)...])),
              !data.isEmpty,
              data.count <= 128 * 1_024 * 1_024 else {
            throw BridgeError.invalidPastedImage
        }

        let header = String(dataURL[..<comma]).lowercased()
        let originalFileExtension: String
        if header.contains("image/jpeg") || header.contains("image/jpg") {
            originalFileExtension = "jpg"
        } else if header.contains("image/webp") {
            originalFileExtension = "webp"
        } else if header.contains("image/tiff") {
            originalFileExtension = "tiff"
        } else {
            originalFileExtension = "png"
        }
        let normalized = try Self.normalizedClipboardImage(
            data: data,
            originalFileExtension: originalFileExtension,
            maxLongEdge: 2_048
        )

        let directory = ApplicationSupport.directory(.pasted)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(
            "Pasted-\(UUID().uuidString.prefix(8)).\(normalized.fileExtension)"
        )
        try normalized.data.write(to: url, options: .atomic)
        store.importImage(
            url: url,
            pixelWidth: normalized.pixelWidth,
            pixelHeight: normalized.pixelHeight
        )
        if normalized.wasResized {
            store.statusMessage = "剪貼簿圖片已等比例縮小至 \(normalized.pixelWidth) × \(normalized.pixelHeight) px。"
        }
        if params["describe"] as? Bool == true {
            store.describeSelected()
        }
    }

    private struct ClipboardImagePayload: Encodable {
        let dataURL: String
        let name: String?
    }

    private static func clipboardImagePayload(from pasteboard: NSPasteboard) -> ClipboardImagePayload? {
        if let data = pasteboard.data(forType: .png), NSImage(data: data) != nil {
            return clipboardImagePayload(pngData: data, name: nil)
        }

        if let data = pasteboard.data(forType: .tiff),
           let image = NSImage(data: data),
           let pngData = pngData(from: image) {
            return clipboardImagePayload(pngData: pngData, name: nil)
        }

        let fileOptions: [NSPasteboard.ReadingOptionKey: Any] = [
            .urlReadingFileURLsOnly: true
        ]
        if let fileURL = (pasteboard.readObjects(
            forClasses: [NSURL.self],
            options: fileOptions
        ) as? [NSURL])?.first as URL?,
           let image = NSImage(contentsOf: fileURL),
           let pngData = pngData(from: image) {
            return clipboardImagePayload(pngData: pngData, name: fileURL.lastPathComponent)
        }

        if let image = (pasteboard.readObjects(
            forClasses: [NSImage.self],
            options: nil
        ) as? [NSImage])?.first,
           let pngData = pngData(from: image) {
            return clipboardImagePayload(pngData: pngData, name: nil)
        }

        return nil
    }

    private static func pngData(from image: NSImage) -> Data? {
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData) else {
            return nil
        }
        return bitmap.representation(using: .png, properties: [:])
    }

    private static func clipboardImagePayload(
        pngData: Data,
        name: String?
    ) -> ClipboardImagePayload? {
        guard !pngData.isEmpty,
              pngData.count <= 128 * 1_024 * 1_024,
              let normalized = try? normalizedClipboardImage(
                data: pngData,
                originalFileExtension: "png",
                maxLongEdge: 2_048
              ) else { return nil }
        return ClipboardImagePayload(
            dataURL: "data:image/png;base64,\(normalized.data.base64EncodedString())",
            name: name
        )
    }

    private struct NormalizedClipboardImage {
        var data: Data
        var fileExtension: String
        var pixelWidth: Int
        var pixelHeight: Int
        var wasResized: Bool
    }

    private static func normalizedClipboardImage(
        data: Data,
        originalFileExtension: String,
        maxLongEdge: Int
    ) throws -> NormalizedClipboardImage {
        guard maxLongEdge > 0,
              let source = CGImageSourceCreateWithData(data as CFData, [
                kCGImageSourceShouldCache: false
              ] as CFDictionary),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                as? [CFString: Any],
              let rawWidth = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let rawHeight = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              rawWidth > 0,
              rawHeight > 0 else {
            throw BridgeError.invalidPastedImage
        }

        let orientation = (properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        let swapsDimensions = (5...8).contains(orientation)
        let displayWidth = swapsDimensions ? rawHeight : rawWidth
        let displayHeight = swapsDimensions ? rawWidth : rawHeight
        guard max(displayWidth, displayHeight) > maxLongEdge else {
            return NormalizedClipboardImage(
                data: data,
                fileExtension: originalFileExtension,
                pixelWidth: displayWidth,
                pixelHeight: displayHeight,
                wasResized: false
            )
        }

        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxLongEdge,
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(
            source,
            0,
            options as CFDictionary
        ) else {
            throw BridgeError.invalidPastedImage
        }

        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw BridgeError.invalidPastedImage
        }
        CGImageDestinationAddImage(destination, thumbnail, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw BridgeError.invalidPastedImage
        }
        return NormalizedClipboardImage(
            data: output as Data,
            fileExtension: "png",
            pixelWidth: thumbnail.width,
            pixelHeight: thumbnail.height,
            wasResized: true
        )
    }

    private func model(from params: [String: Any]) -> ModelDescriptor? {
        guard let modelID = params["modelID"] as? String else { return nil }
        return store.models.first { $0.id == modelID }
    }

    private func uuid(_ value: Any?) -> UUID? {
        guard let string = value as? String else { return nil }
        return UUID(uuidString: string)
    }

    private func integer(_ value: Any?) -> Int? {
        if let number = value as? NSNumber { return number.intValue }
        if let string = value as? String { return Int(string) }
        return nil
    }

    private func double(_ value: Any?) -> Double? {
        if let number = value as? NSNumber { return number.doubleValue }
        if let string = value as? String { return Double(string) }
        return nil
    }

    private func handleRequest(method: String, params: [String: Any]) throws -> [String: Any]? {
        if method == "createAutomaticFlowWorkspace" {
            guard let name = params["name"] as? String,
                  let workspaceID = store.createWorkspace(name: name) else {
                throw BridgeError.assetActionFailed("目前無法建立自動流程工作區。")
            }
            return ["workspaceID": workspaceID.uuidString]
        }
        try handle(method: method, params: params)
        return nil
    }

    private func sendResponse(id: String, payload: [String: Any]? = nil) {
        var response: [String: Any] = ["kind": "response", "id": id, "ok": true]
        if let payload { response["payload"] = payload }
        sendJavaScriptObject(response)
    }

    private func sendError(id: String?, message: String) {
        var object: [String: Any] = ["kind": "response", "ok": false, "error": message]
        if let id { object["id"] = id }
        sendJavaScriptObject(object)
    }

    private func sendJavaScriptObject(_ object: [String: Any]) {
        guard let webView,
              JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object),
              let json = String(data: data, encoding: .utf8) else { return }
        webView.evaluateJavaScript("window.GenImageNative?.receive(\(json));")
    }
}

extension HybridBridgeController: WKScriptMessageHandler {
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "genimage",
              let body = message.body as? [String: Any],
              let method = body["method"] as? String else { return }

        let id = body["id"] as? String
        let params = body["params"] as? [String: Any] ?? [:]

        do {
            let payload = try handleRequest(method: method, params: params)
            if let id { sendResponse(id: id, payload: payload) }
        } catch {
            sendError(id: id, message: error.localizedDescription)
        }
    }
}

extension HybridBridgeController: WKNavigationDelegate {
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        pageReady = true
        pushState()
    }
}

extension HybridBridgeController: WKUIDelegate {
    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        guard let url = navigationAction.request.url,
              let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            return nil
        }
        if !NSWorkspace.shared.open(url) {
            store.statusMessage = "無法在預設瀏覽器開啟連結。"
        }
        return nil
    }
}

private enum BridgeError: LocalizedError {
    case invalidParameters
    case invalidPastedImage
    case assetActionFailed(String)
    case unknownMethod(String)

    var errorDescription: String? {
        switch self {
        case .invalidParameters: "Bridge 參數不完整。"
        case .invalidPastedImage: "剪貼簿圖片格式無法讀取。"
        case let .assetActionFailed(message): message
        case let .unknownMethod(method): "未知的 Bridge 方法：\(method)"
        }
    }
}
