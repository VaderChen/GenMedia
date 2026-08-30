# MLX metallib 版本管理

`mlx.metallib` 必須與載入它的 `mlx-swift` 版本及 Metal kernel 來源相容，不能只以檔名 `mlx.metallib` 判斷。GenMedia 目前有兩條相依版本不同的路徑：

- 主程式、Qwen Image Edit、MiniMax Music 3 與 LTX：`mlx-swift 0.31.6`
- Z-Image 獨立 Worker：`mlx-swift 0.30.6`

## 保存位置

`build.command` 會把產物保存為：

- `RuntimeSupport/mlx-swift-0.31.6.metallib`
- `RuntimeSupport/mlx-swift-0.31.6.metallib.sha256`
- `RuntimeSupport/mlx-swift-0.30.6-zimage.metallib`
- `RuntimeSupport/mlx-swift-0.30.6-zimage.metallib.sha256`

這些是本機建置快取，已由 Git 忽略，不應把不同 MLX 版本重新命名成同一個未標版本檔案。App Bundle 內的 `mlx.metallib` 是執行時必要的固定檔名，但其來源路徑仍保留版本名稱：主路徑位於 `Contents/MacOS/` 與 `Contents/Helpers/`，Z-Image 位於 `Contents/Helpers/ZImage/`。

## 重建規則

建置前會從根套件及四個新版 Worker 的 `Package.resolved` 讀取 `mlx-swift` 版本，確認它們一致；Z-Image 另行確認為舊版。每個保存的 metallib 都以 MLX 版本、Package.resolved、Metal build script 與 Metal kernel 原始碼計算指紋。指紋不一致時才重建，避免每次建置重複編譯，也避免沿用錯版快取。

若新增或升級 Worker，必須讓它的 `Package.resolved` 版本通過 `build.command` 的一致性檢查；不可直接複製其他版本的 `mlx.metallib`。
