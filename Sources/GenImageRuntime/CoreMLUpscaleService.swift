import CoreImage
import CoreML
import Foundation
import GenImageCore
import Vision

public actor CoreMLUpscaleService: ImageUpscaling {
    private var outputDirectory: URL
    private let context = CIContext(options: [.cacheIntermediates: false])
    private var models: [String: MLModel] = [:]

    public init(outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    public func setOutputDirectory(_ outputDirectory: URL) {
        self.outputDirectory = outputDirectory
    }

    /// 釋放已載入的 Core ML 模型，供記憶體壓力保護使用。
    public func unload() {
        models.removeAll(keepingCapacity: false)
    }

    public func upscale(
        request: UpscaleRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ImageAsset {
        guard request.profile.capability == .upscale else {
            throw CoreMLUpscaleError.incompatibleProfile
        }
        guard request.profile.architecture == .coreML else {
            throw CoreMLUpscaleError.unsupportedArchitecture(request.profile.architecture)
        }
        guard request.scale == 2 || request.scale == 4 else {
            throw CoreMLUpscaleError.unsupportedScale(request.scale)
        }
        guard let inputURL = request.asset.fileURL else {
            throw CoreMLUpscaleError.missingInputFile
        }

        let modelURL = URL(fileURLWithPath: request.profile.modelID)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw CoreMLUpscaleError.modelNotFound(modelURL)
        }
        guard var sourceImage = CIImage(
            contentsOf: inputURL,
            options: [.applyOrientationProperty: true]
        ) else {
            throw CoreMLUpscaleError.invalidInputImage(inputURL)
        }

        let sourceExtent = sourceImage.extent.integral
        sourceImage = sourceImage.transformed(
            by: CGAffineTransform(
                translationX: -sourceExtent.minX,
                y: -sourceExtent.minY
            )
        )

        let sourceWidth = Int(sourceExtent.width)
        let sourceHeight = Int(sourceExtent.height)
        guard sourceWidth > 0, sourceHeight > 0 else {
            throw CoreMLUpscaleError.invalidInputImage(inputURL)
        }

        progress(0.02)
        let model = try loadModel(at: modelURL)
        progress(0.08)
        let tileSize = request.profile.defaults.tileSize ?? 512
        guard tileSize == 512 else {
            throw CoreMLUpscaleError.unsupportedTileSize(tileSize)
        }

        let xTiles = Int(ceil(Double(sourceWidth) / Double(tileSize)))
        let yTiles = Int(ceil(Double(sourceHeight) / Double(tileSize)))
        let totalTiles = xTiles * yTiles
        let outputExtent = CGRect(
            x: 0,
            y: 0,
            width: sourceWidth * request.scale,
            height: sourceHeight * request.scale
        )
        var composed = CIImage(color: .clear).cropped(to: outputExtent)
        var completedTiles = 0

        for yIndex in 0..<yTiles {
            for xIndex in 0..<xTiles {
                try Task.checkCancellation()

                let x = xIndex * tileSize
                let y = yIndex * tileSize
                let tileWidth = min(tileSize, sourceWidth - x)
                let tileHeight = min(tileSize, sourceHeight - y)
                let sourceRect = CGRect(x: x, y: y, width: tileWidth, height: tileHeight)
                let tile = sourceImage
                    .cropped(to: sourceRect)
                    .transformed(by: CGAffineTransform(translationX: -CGFloat(x), y: -CGFloat(y)))
                let paddedExtent = CGRect(x: 0, y: 0, width: tileSize, height: tileSize)
                let padded = tile.composited(
                    over: CIImage(color: .black).cropped(to: paddedExtent)
                )

                guard let tileCGImage = context.createCGImage(padded, from: paddedExtent) else {
                    throw CoreMLUpscaleError.cannotCreateTile
                }
                let outputTile = try predict(cgImage: tileCGImage, model: model)
                let scaledOutput: CIImage
                if request.scale == 2 {
                    scaledOutput = outputTile.applyingFilter(
                        "CILanczosScaleTransform",
                        parameters: [
                            kCIInputScaleKey: 0.5,
                            kCIInputAspectRatioKey: 1.0
                        ]
                    )
                } else {
                    scaledOutput = outputTile
                }
                let croppedOutput = scaledOutput
                    .cropped(
                        to: CGRect(
                            x: 0,
                            y: 0,
                            width: tileWidth * request.scale,
                            height: tileHeight * request.scale
                        )
                    )
                    .transformed(
                        by: CGAffineTransform(
                            translationX: CGFloat(x * request.scale),
                            y: CGFloat(y * request.scale)
                        )
                    )
                composed = croppedOutput.composited(over: composed)

                completedTiles += 1
                let tileProgress = Double(completedTiles) / Double(totalTiles)
                progress(0.08 + tileProgress * 0.84)
            }
        }

        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
        let outputURL = OutputFileNaming.imageURL(in: outputDirectory, pathExtension: "png")
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        progress(0.94)
        try context.writePNGRepresentation(
            of: composed,
            to: outputURL,
            format: .RGBA8,
            colorSpace: colorSpace
        )
        progress(1)

        return ImageAsset(
            projectID: request.asset.projectID,
            parentAssetID: request.asset.id,
            kind: .upscaled,
            title: "\(request.asset.title) · \(request.scale)×",
            fileURL: outputURL,
            pixelWidth: sourceWidth * request.scale,
            pixelHeight: sourceHeight * request.scale,
            recipeID: request.asset.recipeID
        )
    }

    private func loadModel(at sourceURL: URL) throws -> MLModel {
        if let model = models[sourceURL.path] {
            return model
        }

        let compiledURL: URL
        if sourceURL.pathExtension == "mlmodelc" {
            compiledURL = sourceURL
        } else {
            compiledURL = try MLModel.compileModel(at: sourceURL)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        models[sourceURL.path] = model
        return model
    }

    private func predict(cgImage: CGImage, model: MLModel) throws -> CIImage {
        let visionModel = try VNCoreMLModel(for: model)
        let visionRequest = VNCoreMLRequest(model: visionModel)
        visionRequest.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
        try handler.perform([visionRequest])

        guard let observation = visionRequest.results?.first as? VNPixelBufferObservation else {
            throw CoreMLUpscaleError.invalidModelOutput
        }
        return CIImage(cvPixelBuffer: observation.pixelBuffer)
    }
}

public enum CoreMLUpscaleError: LocalizedError, Sendable {
    case incompatibleProfile
    case unsupportedArchitecture(InferenceArchitecture)
    case unsupportedScale(Int)
    case unsupportedTileSize(Int)
    case missingInputFile
    case modelNotFound(URL)
    case invalidInputImage(URL)
    case cannotCreateTile
    case invalidModelOutput

    public var errorDescription: String? {
        switch self {
        case .incompatibleProfile: "Profile 不是 Upscale 類型。"
        case let .unsupportedArchitecture(value): "不支援的 Upscale 架構：\(value.title)。"
        case let .unsupportedScale(value): "目前 Core ML Upscale 僅支援 2× 與 4×，收到 \(value)×。"
        case let .unsupportedTileSize(value): "此模型需要 512×512 tile，收到 \(value)。"
        case .missingInputFile: "圖片資產沒有本機檔案。"
        case let .modelNotFound(url): "找不到 Core ML 模型：\(url.path)"
        case let .invalidInputImage(url): "無法讀取圖片：\(url.path)"
        case .cannotCreateTile: "無法建立 Upscale 輸入 tile。"
        case .invalidModelOutput: "Core ML 模型沒有輸出圖片。"
        }
    }
}
