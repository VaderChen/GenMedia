# Web Bridge

[繁體中文](WEB_BRIDGE.md) | [English](WEB_BRIDGE.en.md) | 日本語 | [한국어](WEB_BRIDGE.ko.md)

## 原則

- JavaScript は JSON 互換データのみを送信します。
- 各コマンドには一意の request ID と成功／失敗の応答があります。
- Swift を状態の情報源とし、内容の変更は `WebAppState`、進捗とリソースは `receiveActivity(WebActivityState)` で `jobs`、`installations`、`statusMessage`、`systemMetrics`、`isReleasingMemory` のみを送信します。
- 大きな画像は JSON に含めず、Web UI が `genimage-asset://<asset-id>` を通じて読み込みます。
- `AssetSchemeHandler` は `AppStore` に登録されたアセット URL の読み込みだけを許可します。

## JavaScript 呼び出し

```js
await invoke("generate", { linkToSelectedAsset: false });
```

実際に送信されるメッセージ：

```json
{
  "id": "web-...",
  "method": "generate",
  "params": {
    "linkToSelectedAsset": false
  }
}
```

## コマンド

### ワークスペース

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

### モデル

- `installModel`
- `pauseModel`
- `removeModel`
- `repairModel`
- `setModelRoot`
- `installProfileModels`

### プロファイル

- `selectProfile`
- `applyProfileDefaults`
- `createProfile`
- `duplicateProfile`
- `updateProfile`
- `deleteProfile`

## 互換性方針

現在の `WebAppState.schemaVersion` は `1` です。新しいフィールドは後方互換性を維持する必要があります。破壊的変更では major schema version を上げ、古い UI が誤ったデータを暗黙に使用しないようにします。

## 操作順序と失敗応答

`renameAsset`、`removeAsset`、`closeWorkspaceProject` は処理実行中またはキャンセル中にエラーを返します。JavaScript は `invoke` の成功を待ってからローカルのアセットやタブ参照を削除します。他のアセットが参照するファイルは保持し、`statusMessage` で理由を通知します。

同じモデルのインストール、修復、削除は先行処理の後片付けを待ちます。モデル検出中は新しい生成とモデル操作を受け付けず、検出失敗時は元のディレクトリを保持します。`setOutputDirectory` は同期更新し、新しい処理から反映します。実行中の処理は元のパスを維持します。
