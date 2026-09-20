# Web Bridge

繁體中文 | [English](WEB_BRIDGE.en.md) | [日本語](WEB_BRIDGE.ja.md) | [한국어](WEB_BRIDGE.ko.md)

## 原則

- JavaScript 只傳送 JSON 相容資料。
- 每個命令都有唯一 request ID 及成功／失敗回覆。
- Swift 是真實狀態來源：內容變動推送完整 `WebAppState`；資源與進度變動透過 `receiveActivity(WebActivityState)` 只推送 `jobs`、`installations`、`statusMessage`、`systemMetrics` 與 `isReleasingMemory`。
- 大型圖片不放進 JSON；Web UI 使用 `genimage-asset://<asset-id>` 讀取。
- `AssetSchemeHandler` 只允許讀取 AppStore 已登記的資產 URL。

## JavaScript 呼叫

```js
await invoke("generate", { linkToSelectedAsset: false });
```

實際送出的訊息：

```json
{
  "id": "web-...",
  "method": "generate",
  "params": {
    "linkToSelectedAsset": false
  }
}
```

## 命令

### 工作區

- `bootstrap`
- `selectAsset`
- `renameAsset`
- `removeAsset`
- `closeWorkspaceProject`
- `createWorkspace`
- `selectWorkspace`
- `deleteWorkspace`
- `setOutputDirectory`
- `updateRecipe`
- `randomizeSeed`
- `generate`
- `describe`
- `upscale`
- `importImage`
- `cancelJob`
- `clearJobs`

### 模型

- `installModel`
- `pauseModel`
- `removeModel`
- `repairModel`
- `setModelRoot`
- `installProfileModels`

### Profile

- `selectProfile`
- `applyProfileDefaults`
- `createProfile`
- `duplicateProfile`
- `updateProfile`
- `deleteProfile`

## 相容性策略

目前 `WebAppState.schemaVersion` 為 `1`。新增欄位必須保持向後相容；破壞性變更則提高 major schema version，避免舊 UI 靜默使用錯誤資料。

## 操作順序與失敗回覆

`renameAsset`、`removeAsset`、`closeWorkspaceProject` 在工作執行或取消中回覆錯誤；JavaScript 必須等待 `invoke` 成功後才移除本地資產或分頁參照。刪除檔案若仍有其他資產引用，會保留檔案並透過 `statusMessage` 說明。

同一模型的安裝／修復／移除會等待舊任務清理。模型根目錄掃描期間不接受新生成與模型操作；掃描失敗保留原目錄。`setOutputDirectory` 同步更新服務，新工作使用新目錄，進行中工作保留原路徑。
