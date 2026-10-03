import Foundation

public struct WorkflowGraph: Sendable {
    public private(set) var assets: [MediaAsset]
    public private(set) var operations: [WorkflowOperation]
    private var assetIndices: [UUID: Int]

    public init(assets: [MediaAsset] = [], operations: [WorkflowOperation] = []) {
        self.assets = assets
        self.operations = operations
        assetIndices = [:]
        assetIndices.reserveCapacity(assets.count)
        for (index, asset) in assets.enumerated() where assetIndices[asset.id] == nil {
            assetIndices[asset.id] = index
        }
    }

    public mutating func append(asset: MediaAsset) {
        if assetIndices[asset.id] == nil {
            assetIndices[asset.id] = assets.count
        }
        assets.append(asset)
    }

    public mutating func append(operation: WorkflowOperation) {
        operations.append(operation)
    }

    public func asset(id: UUID) -> MediaAsset? {
        assetIndices[id].map { assets[$0] }
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
