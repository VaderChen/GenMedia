import Foundation
import GenImageCore

// 生成工作的進度、取消與記憶體釋放。
extension AppStore {
    func cancelJob(_ id: UUID) {
        guard let job = jobs.first(where: { $0.id == id }) else { return }
        guard [.queued, .running].contains(job.state) else { return }
        guard let task = jobTasks[id] else {
            updateJob(id) { $0.state = .cancelled }
            return
        }
        cancellationRequestedJobIDs.insert(id)
        updateJob(id) { $0.state = .cancelling }
        task.cancel()
        Task { @MainActor [weak self] in
            await task.value
            self?.finishJobTask(id)
        }
        statusMessage = "正在取消任務，Runtime 停止後即可開始下一個任務。"
    }

    func finishJobTask(_ id: UUID) {
        jobTasks[id] = nil
        let cancellationRequested = cancellationRequestedJobIDs.remove(id) != nil
        guard cancellationRequested
            || jobs.first(where: { $0.id == id })?.state == .cancelling else { return }
        updateJob(id) { $0.state = .cancelled }
        statusMessage = "任務已取消，可以繼續操作。"
    }

    func releaseMemory() {
        guard !jobs.contains(where: { [.queued, .running, .cancelling].contains($0.state) }) else {
            statusMessage = "任務執行或取消中，完成後才能釋放記憶體。"
            return
        }
        guard !isReleasingMemory else { return }
        isReleasingMemory = true
        statusMessage = "正在釋放已載入的模型記憶體…"
        let textToImage = textToImageService
        let imageToText = imageToTextService
        let upscale = upscaleService
        Task { @MainActor [weak self] in
            await textToImage.unload()
            await imageToText.unload()
            await upscale.unload()
            guard let self else { return }
            isReleasingMemory = false
            statusMessage = "模型記憶體已釋放。"
        }
    }

    func clearFinishedJobs() {
        jobs.removeAll { [.completed, .cancelled, .failed].contains($0.state) }
    }

    func updateJobProgress(_ id: UUID, value: Double) {
        let normalizedValue = min(1, max(0, value))
        let now = Date()
        let lastUpdate = lastJobProgressUpdate[id] ?? .distantPast
        guard normalizedValue >= 1
            || now.timeIntervalSince(lastUpdate) >= Self.jobProgressUpdateInterval else {
            return
        }
        lastJobProgressUpdate[id] = now
        updateJob(id) { job in
            job.progress = max(job.progress, normalizedValue)
        }
        if normalizedValue >= 1 {
            lastJobProgressUpdate[id] = nil
        }
    }

    func updateJob(_ id: UUID, change: (inout GenerationJob) -> Void) {
        guard let index = jobs.firstIndex(where: { $0.id == id }) else { return }
        change(&jobs[index])
        let now = Date()
        switch jobs[index].state {
        case .running, .cancelling:
            if jobs[index].startedAt == nil {
                jobs[index].startedAt = now
            }
        case .completed, .cancelled, .failed:
            lastJobProgressUpdate[id] = nil
            if jobs[index].startedAt == nil {
                jobs[index].startedAt = jobs[index].createdAt
            }
            if jobs[index].finishedAt == nil {
                jobs[index].finishedAt = now
            }
        case .queued:
            break
        }
    }
}
