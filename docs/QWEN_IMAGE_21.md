# Qwen-Image 2.1：Swift／MLX 整合

本 Runtime 不使用 Python、PyTorch、Diffusers 或外部 API 執行推論。Swift Worker 使用既有 MLX Metal 後端；NumPy 只用於產生小型測試參考值。

## 使用方式

1. 在模型中心安裝 **Qwen-Image 2.1 MLX 4-bit**。
2. 選取「文生圖 · Qwen-Image 2.1 4-bit」或「圖生圖 · Qwen-Image 2.1 4-bit」。兩者共用同一份下載。
3. 預設 1024×1024、40 步；寬高須為 256～2048 之間的 32 倍數。
4. 按「文生圖」生成新圖片；圖生圖則先選取一張本機圖片、輸入編輯指令，再按獨立的「圖生圖」按鈕。選取生成結果不會切換文生圖模式。輸出為保留 alpha 的 RGBA PNG。

**圖像編輯仍屬實驗性**：目前 256×256、20／40 步改色實跑能保留構圖、產生自然的綠蘋果。先前 512×512、40 步曾出現顆粒／金屬感；本輪依需求只做小尺寸驗證，未重跑 512，因此不將該問題標記為已解決，也不宣稱與官方圖片編輯品質一致。

目前採 guidance-free（CFG=1），不接受負面提示詞或 LoRA；送入時會明確報錯，避免參數被默默忽略。目前 App 只提供單張參考圖，尚未開放模型的多圖能力。

模型約 10.5 GB，建議 32 GB 以上統一記憶體。編碼器、Transformer、VAE 分階段載入；這不代表任意解析度均能在 16 GB 順利完成。尚未加入 VAE tiling 和條件 KV cache，高解析度的時間及記憶體用量較高。

模型權重不包含在 GitHub 原始碼或 Release DMG，請在模型中心另外下載。開發者若已有安裝器驗證過的 `Models/qwen-image-2.1-mlx-4bit`，可將 App 的模型目錄指向該 `Models` 目錄重用。

## 模型及格式

- 上游：[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)。
- MLX 轉換：[mlx-community/Qwen-Image-2.1-MLX-4bit](https://huggingface.co/mlx-community/Qwen-Image-2.1-MLX-4bit)。
- 固定 revision：`4db4e8c0c0e7a1debf0320415bec8388e888494c`。
- 安裝目錄：模型根目錄下的 `qwen-image-2.1-mlx-4bit`。
- 權重授權：**Qwen Research License**，以上游模型條款為準；程式碼授權另見 [第三方聲明](../THIRD_PARTY_NOTICES.md)。

不能將 2511 權重放入此目錄替代：2.1 使用 32 層單流 DiT、Qwen3-VL-8B 與 64 通道 VAE，架構不同。此實作對應上述固定 MLX 轉換，其 timestep 線性層與 modulation 的命名不同於原始 Diffusers 權重；不會自動把任意模型當成 2.1 載入。

## 建置

一般 App 建置使用根目錄的 `build.command`，會解析套件、套用版本固定的 Runtime patches，並建置／封裝 `GenImageQwen21Worker`。

手動開發建置前，必須先解析套件並依 `Patches/manifest.txt` 套用 `Qwen3VL-Image-Conditioning.patch` 與 `Qwen3VL-Vision-GELU.patch`。完整專案可使用既有的 `scripts/apply-runtime-patches.command`；該腳本也驗證其他獨立 Worker checkout，需先完成它們的套件解析。只建置根套件時，可在 `.build/checkouts/mlx-swift-lm` 以 `git apply` 分別套用這兩個 patch；重複套用前先使用 `git apply --reverse --check` 確認是否已套用。單純 `swift build` 不會自動修改相依套件。

```sh
swift build --product GenImageQwen21Worker -j 4
swift build --build-tests -j 4
swift test --skip-build -j 4
```

MLX 測試的 Metal library 準備方式見 [驗證文件](VALIDATION.md)。`GENIMAGE_QWEN21_WORKER` 可指定開發用 Worker 路徑。正式 App 使用內附 Worker，不需設定環境變數。

Worker 以 `--request /absolute/request.json` 接收 JSON：

```json
{
  "modelDirectory": "/absolute/models/qwen-image-2.1-mlx-4bit",
  "outputPaths": ["/absolute/output.png"],
  "prompt": "A red apple on a white table, studio photograph.",
  "negativePrompt": "",
  "width": 256,
  "height": 256,
  "steps": 20,
  "seed": 42
}
```

開發驗證先使用上述 256×256、20 步設定；40 步與較大尺寸只在需要確認時執行，App 預設仍為 1024×1024、40 步。

圖像編輯另加 `inputPath`。服務會建立輸出目錄；直接呼叫 Worker 時需先建立目錄。每張圖以 `seed + index` 產生獨立噪聲。Swift MLX 與 PyTorch 的 RNG 不同，相同 seed 不保證像素相同。

## 驗證與已知範圍

- 小型單層 DiT 與獨立 NumPy dense-attention oracle 比對，涵蓋純文字及穿插參考影像的區塊因果遮罩；允許 BF16 誤差。
- 獨立 Qwen3-VL NumPy oracle 涵蓋非正方形視覺 patch、位置插值、vision/text RoPE、deepstack 與最終 RMSNorm 前輸出；另驗證半透明 RGBA、sRGB 疊白底及零中心 RMSNorm 的 FP32 計算。
- 測試 affine INT4 matmul、FlowMatch schedule 終值、VAE 時間補零／通道重排、PNG alpha、參數拒絕、模型路由、工作失敗清理與素材連結。
- 模型目錄掃描要求 manifest、所有設定與非空權重，缺檔或零長度檔不會列為已安裝。
- 真實權重推論與完整測試結果記錄於 [驗證文件](VALIDATION.md)。數值單元測試不能代表正式圖片品質。

參考實作：[Diffusers QwenImage21](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21)、[Qwen 官方程式庫](https://github.com/QwenLM/Qwen-Image-2.1)。
