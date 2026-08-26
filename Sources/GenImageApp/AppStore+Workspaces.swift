import Foundation
import GenImageCore

extension AppStore {
    func createWorkspace(name: String) {
        guard canSwitchWorkspace else { return }
        let baseName = String(
            name.trimmingCharacters(in: .whitespacesAndNewlines).prefix(80)
        )
        let workspace = Project(
            name: uniqueWorkspaceName(
                baseName.isEmpty ? "工作區 \(projects.count + 1)" : baseName
            )
        )
        projects.append(workspace)
        selectWorkspace(workspace.id, announce: false)
        statusMessage = "已新增工作區「\(workspace.name)」。"
    }

    func selectWorkspace(_ id: UUID) {
        selectWorkspace(id, announce: true)
    }

    func deleteWorkspace(_ id: UUID) {
        guard canSwitchWorkspace else { return }
        guard projects.count > 1 else {
            statusMessage = "至少需要保留一個工作區。"
            return
        }
        guard let deletedIndex = projects.firstIndex(where: { $0.id == id }) else {
            return
        }

        let deletedWorkspace = projects[deletedIndex]
        let replacementIndex = deletedIndex == projects.endIndex - 1
            ? projects.index(before: deletedIndex)
            : projects.index(after: deletedIndex)
        let replacementID = projects[replacementIndex].id

        assets.removeAll { $0.projectID == id }
        operations.removeAll { $0.projectID == id }
        if selectedProjectID == id {
            selectedProjectID = replacementID
            selectedAssetID = nil
            comparisonAssetID = nil
        }
        projects.remove(at: deletedIndex)
        statusMessage = "已刪除工作區「\(deletedWorkspace.name)」；輸出檔案仍保留於磁碟。"
    }

    private func selectWorkspace(_ id: UUID, announce: Bool) {
        guard canSwitchWorkspace,
              let workspace = projects.first(where: { $0.id == id }) else {
            return
        }
        guard selectedProjectID != id else { return }
        selectedProjectID = id
        selectedAssetID = nil
        comparisonAssetID = nil
        if announce {
            statusMessage = "已切換至工作區「\(workspace.name)」。"
        }
    }

    private var canSwitchWorkspace: Bool {
        guard !jobs.contains(where: { [.queued, .running, .cancelling].contains($0.state) }) else {
            statusMessage = "任務執行或取消中，完成後才能切換工作區。"
            return false
        }
        return true
    }

    private func uniqueWorkspaceName(_ baseName: String) -> String {
        let existingNames = Set(projects.map { $0.name.localizedLowercase })
        guard existingNames.contains(baseName.localizedLowercase) else {
            return baseName
        }
        var suffix = 2
        while existingNames.contains("\(baseName) \(suffix)".localizedLowercase) {
            suffix += 1
        }
        return "\(baseName) \(suffix)"
    }
}
