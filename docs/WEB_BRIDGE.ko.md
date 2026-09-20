# Web Bridge

[繁體中文](WEB_BRIDGE.md) | [English](WEB_BRIDGE.en.md) | [日本語](WEB_BRIDGE.ja.md) | 한국어

## 원칙

- JavaScript는 JSON 호환 데이터만 전송합니다.
- 각 명령에는 고유한 request ID와 성공 또는 실패 응답이 있습니다.
- Swift가 실제 상태의 기준입니다. 콘텐츠 변경은 `WebAppState`로, 진행률과 리소스는 `receiveActivity(WebActivityState)`를 통해 `jobs`, `installations`, `statusMessage`, `systemMetrics`, `isReleasingMemory`만 전송합니다.
- 큰 이미지는 JSON에 포함하지 않고 Web UI가 `genimage-asset://<asset-id>`를 통해 읽습니다.
- `AssetSchemeHandler`는 `AppStore`에 등록된 에셋 URL만 읽을 수 있도록 허용합니다.

## JavaScript 호출

```js
await invoke("generate", { linkToSelectedAsset: false });
```

실제로 전송되는 메시지:

```json
{
  "id": "web-...",
  "method": "generate",
  "params": {
    "linkToSelectedAsset": false
  }
}
```

## 명령

### 작업 공간

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

### 모델

- `installModel`
- `pauseModel`
- `removeModel`
- `repairModel`
- `setModelRoot`
- `installProfileModels`

### 프로필

- `selectProfile`
- `applyProfileDefaults`
- `createProfile`
- `duplicateProfile`
- `updateProfile`
- `deleteProfile`

## 호환성 전략

현재 `WebAppState.schemaVersion`은 `1`입니다. 새 필드는 하위 호환성을 유지해야 합니다. 호환되지 않는 변경에서는 major schema version을 올려 이전 UI가 잘못된 데이터를 자동으로 사용하지 않도록 합니다.

## 작업 순서와 실패 응답

`renameAsset`, `removeAsset`, `closeWorkspaceProject`는 작업 실행 또는 취소 중에 오류를 반환합니다. JavaScript는 `invoke` 성공 후 로컬 에셋이나 탭 참조를 제거해야 합니다. 다른 에셋이 참조하는 파일은 보존하고 `statusMessage`로 이유를 알립니다.

같은 모델의 설치, 복구, 삭제는 이전 작업의 정리를 기다립니다. 모델 검색 중에는 새 생성 및 모델 작업을 받지 않고 검색 실패 시 기존 디렉터리를 유지합니다. `setOutputDirectory`는 서비스를 동기적으로 갱신하여 새 작업부터 적용하며 실행 중인 작업은 원래 경로를 유지합니다.
