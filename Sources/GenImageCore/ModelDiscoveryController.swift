import Foundation

/// Only the latest requested directory may publish a catalog, even when an old
/// filesystem operation finishes after cancellation.
@MainActor
public final class ModelDiscoveryController {
    public private(set) var isLoading = false
    private var generation = UUID()
    private var task: Task<Void, Never>?
    private let discover: @Sendable (URL) throws -> DiscoveredModelCatalog

    public init(discover: @escaping @Sendable (URL) throws -> DiscoveredModelCatalog = {
        LocalModelDiscovery.discover(at: $0)
    }) {
        self.discover = discover
    }

    @discardableResult
    public func load(
        at root: URL,
        allowMissingRoot: Bool = false,
        beforeDiscovery: @escaping @MainActor () async -> Void = {},
        completion: @escaping @MainActor (Result<DiscoveredModelCatalog, any Error>) -> Void
    ) -> Task<Void, Never> {
        task?.cancel()
        let token = UUID()
        generation = token
        isLoading = true
        let next = Task { [weak self, discover] in
            let result: Result<DiscoveredModelCatalog, any Error>
            do {
                await beforeDiscovery()
                try Task.checkCancellation()
                let catalog = try await BackgroundTask.run {
                    do {
                        let values = try root.resourceValues(forKeys: [.isDirectoryKey])
                        guard values.isDirectory == true else { throw CocoaError(.fileReadUnknown) }
                    } catch let error as CocoaError where allowMissingRoot && error.code == .fileReadNoSuchFile {
                        return DiscoveredModelCatalog()
                    }
                    return try discover(root)
                }
                result = .success(catalog)
            } catch {
                result = .failure(error)
            }
            guard !Task.isCancelled, let self, self.generation == token else { return }
            self.task = nil
            self.isLoading = false
            completion(result)
        }
        task = next
        return next
    }

    public func cancel() {
        generation = UUID()
        task?.cancel()
        task = nil
        isLoading = false
    }

    deinit { task?.cancel() }
}
