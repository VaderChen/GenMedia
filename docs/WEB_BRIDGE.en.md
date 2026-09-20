# Web Bridge

[繁體中文](WEB_BRIDGE.md) | English | [日本語](WEB_BRIDGE.ja.md) | [한국어](WEB_BRIDGE.ko.md)

## Principles

- JavaScript sends only JSON-compatible data.
- Every command has a unique request ID and a success or failure response.
- Swift is the source of truth: content changes push `WebAppState`; progress and metrics use `receiveActivity(WebActivityState)` with only `jobs`, `installations`, `statusMessage`, `systemMetrics`, and `isReleasingMemory`.
- Large images are not embedded in JSON; the Web UI reads them through `genimage-asset://<asset-id>`.
- `AssetSchemeHandler` permits access only to asset URLs registered in `AppStore`.

## JavaScript Calls

```js
await invoke("generate", { linkToSelectedAsset: false });
```

Actual message:

```json
{
  "id": "web-...",
  "method": "generate",
  "params": {
    "linkToSelectedAsset": false
  }
}
```

## Commands

### Workspace

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

### Models

- `installModel`
- `pauseModel`
- `removeModel`
- `repairModel`
- `setModelRoot`
- `installProfileModels`

### Profiles

- `selectProfile`
- `applyProfileDefaults`
- `createProfile`
- `duplicateProfile`
- `updateProfile`
- `deleteProfile`

## Compatibility Strategy

The current `WebAppState.schemaVersion` is `1`. New fields must remain backward-compatible. Breaking changes must increment the major schema version so older UIs do not silently consume invalid data.

## Operation ordering and failure responses

`renameAsset`, `removeAsset`, and `closeWorkspaceProject` return an error while a job runs or is cancelling. JavaScript must await a successful `invoke` before removing local asset or tab references. A file still referenced by another asset is retained, with an explanation in `statusMessage`.

Installation, repair, and removal for the same model wait for earlier cleanup. Model-root scans block new generation and model operations; failed scans retain the previous directory. `setOutputDirectory` updates services synchronously: new jobs use the new location and running jobs retain their original paths.
