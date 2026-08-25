import Foundation

public struct WorkflowGraph: Sendable {
    public private(set) var assets: [MediaAsset]
    public private(set) var operations: [WorkflowOperation]

    public init(assets: [MediaAsset] = [], operations: [WorkflowOperation] = []) {
        self.assets = assets
        self.operations = operations
    }

    public mutating func append(asset: MediaAsset) {
        assets.append(asset)
    }

    public mutating func append(operation: WorkflowOperation) {
        operations.append(operation)
    }

    public func asset(id: UUID) -> MediaAsset? {
        assets.first { $0.id == id }
    }

    public func children(of assetID: UUID) -> [MediaAsset] {
        assets
            .filter { $0.parentAssetID == assetID }
            .sorted { $0.createdAt < $1.createdAt }
    }

    public func lineage(of assetID: UUID) -> [MediaAsset] {
        var result: [MediaAsset] = []
        var currentID: UUID? = assetID
        var visited = Set<UUID>()

        while let id = currentID, visited.insert(id).inserted, let current = asset(id: id) {
            result.append(current)
            currentID = current.parentAssetID
        }

        return result.reversed()
    }
}
