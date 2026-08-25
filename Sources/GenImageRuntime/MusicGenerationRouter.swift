import Foundation
import GenImageCore

public protocol MusicRuntimeAdapter: MusicGenerating {
    func supports(_ request: MusicGenerationRequest) -> Bool
}

public final class MusicGenerationRouter: MusicGenerating, Sendable {
    private let adapters: [any MusicRuntimeAdapter]

    public init(adapters: [any MusicRuntimeAdapter]) {
        self.adapters = adapters
    }

    public func generate(
        request: MusicGenerationRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> MediaAsset {
        guard let adapter = adapters.first(where: { $0.supports(request) }) else {
            throw MusicGenerationRouterError.unsupportedRequest(
                modelID: request.profile.modelID,
                architecture: request.profile.architecture
            )
        }
        return try await adapter.generate(request: request, progress: progress)
    }
}

public enum MusicGenerationRouterError: LocalizedError, Sendable {
    case unsupportedRequest(modelID: String, architecture: InferenceArchitecture)

    public var errorDescription: String? {
        switch self {
        case let .unsupportedRequest(modelID, architecture):
            "找不到支援模型「\(modelID)」與架構「\(architecture.title)」的音樂 Runtime。"
        }
    }
}
