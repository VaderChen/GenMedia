import Foundation
import GenImageCore

extension AppStore {
    @discardableResult
    func setCivitaiToken(_ rawToken: String) -> Bool {
        do {
            try CivitaiTokenStore.save(rawToken)
            statusMessage = "Civitai API Token 已安全儲存；現在可以直接下載 Civitai LoRA。"
            return true
        } catch {
            statusMessage = "Civitai API Token 儲存失敗：\(error.localizedDescription)"
            return false
        }
    }

    @discardableResult
    func clearCivitaiToken() -> Bool {
        do {
            try CivitaiTokenStore.delete()
            statusMessage = "已清除 Civitai API Token。"
            return true
        } catch {
            statusMessage = "Civitai API Token 清除失敗：\(error.localizedDescription)"
            return false
        }
    }

    @discardableResult
    func setHuggingFaceToken(_ rawToken: String) -> Bool {
        do {
            try HuggingFaceTokenStore.save(rawToken)
            statusMessage = "Hugging Face API Token 已安全儲存；模型中心會自動用於下載。"
            return true
        } catch {
            statusMessage = "Hugging Face API Token 儲存失敗：\(error.localizedDescription)"
            return false
        }
    }

    @discardableResult
    func clearHuggingFaceToken() -> Bool {
        do {
            try HuggingFaceTokenStore.delete()
            statusMessage = "已清除 Hugging Face API Token。"
            return true
        } catch {
            statusMessage = "Hugging Face API Token 清除失敗：\(error.localizedDescription)"
            return false
        }
    }
}
