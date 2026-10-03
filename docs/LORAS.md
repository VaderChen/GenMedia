# 文生圖與文生影 LoRA

模型中心新增 13 組可直接下載的 LoRA：4 種 **Z-Image Turbo** 圖像風格、4 種 **LTX-2.3** 鏡頭運動，4 組 **MiniMax H3** 低步數加速，以及 1 組 **Qwen 2.1 Turbo** 文生圖／編輯加速。沿用現有的模型中心、LoRA 選單與 Profile 編輯器。

## 圖像風格

| 名稱 | 提示詞建議 | 下載量 | 來源 |
|---|---|---:|---|
| 鉛筆素描 | `pencil sketch` 或 `color pencil sketch` | 170 MB | [Ttio2 / Pencil Sketch](https://huggingface.co/Ttio2/Z-Image-Turbo-pencil-sketch) |
| 吉卜力風格 | `Ghibli style` | 170 MB | [Ttio2 / Ghibli Style](https://huggingface.co/Ttio2/Z-Image-Turbo-Ghibli-Style) |
| 週六早晨卡通 | `saturd4ym0rning` | 74 MB | [RenderArtist / Saturday Morning](https://huggingface.co/renderartist/Saturday-Morning-Z-Image-Turbo) |
| 復古彩色電影 | `t3chnic4lly` | 255 MB | [RenderArtist / Technically Color](https://huggingface.co/renderartist/Technically-Color-Z-Image-Turbo) |

在模型中心下載所需 LoRA，回到文生圖並選擇 Z-Image Turbo Profile，再從 LoRA 選單選取風格。提示詞加入表中的觸發詞並描述主體、構圖與光線；權重可先從 0.8 開始調整。這四組 LoRA 使用 Apache-2.0 授權，不能直接套用到 Qwen-Image 模型。

## 影片鏡頭

| 名稱 | 提示詞範例 | 下載量 | 來源 |
|---|---|---:|---|
| 鏡頭推近 | `The camera dollies in toward the subject.` | 327 MB | [Lightricks / Dolly In](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In) |
| 鏡頭拉遠 | `The camera dollies out, revealing the surroundings.` | 327 MB | [Lightricks / Dolly Out](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out) |
| 鏡頭左移 | `The camera dollies left.` | 327 MB | [Lightricks / Dolly Left](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left) |
| 鏡頭右移 | `The camera dollies right.` | 327 MB | [Lightricks / Dolly Right](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right) |

每個鏡頭各提供 MLX Q4 與 GGUF Q3 版本的文生影 Profile，共 8 個預設。選擇對應 Profile 並下載其相依模型；主模型、文字編碼器、VAE 與 LoRA 全部就緒後即可生成。預設 LoRA 權重為 0.8，可複製內建 Profile 後，在副本的編輯器調整，提示詞需自行描述鏡頭移動及其目的地，不會自動改寫。

這四組官方 LoRA 原以 **LTX-2 19B** 訓練；本專案提供的是 **LTX-2.3 的實驗性相容載入**，並非宣稱它們原生訓練於 2.3。層名、形狀相容與權重生效不等於影片鏡頭品質已驗證。授權依各來源頁面的 LTX-2 Community License。

支援的文生影基底：

- `dgrauet/ltx-2.3-mlx-q4`
- `unsloth/LTX-2.3-GGUF@distilled-1.1-Q3_K_M`

LTX 目前接受一般 Linear LoRA，可在 Profile 組合多個相容鏡頭 LoRA。保留「無條件控制」與控制強度 1；這些鏡頭 LoRA 不適用於 Canny／IC-LoRA、LTX-Video 0.9.6 或 MiniMax H3。LTX-2.3 基底仍建議 48 GB 以上記憶體，LoRA 不會降低基底模型的需求。

## 低步數文生影（MiniMax H3，實驗性）

| LoRA | 生成步數 | 支援的 FL2VA 基底 | 下載量 | 來源 |
|---|---:|---|---:|---|
| LightX2V 4-step v1.0 768p | 固定 4 | 完整 GGUF、Pruned GGUF | 1.96 GB | [LightX2V](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| LightX2V 8-step v1.0 768p | 固定 8 | 完整 GGUF、Pruned GGUF | 1.96 GB | [LightX2V](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| LightX2V 4 步 v1.2 768p | 4 | 完整／Pruned GGUF | 1.96 GB | [LightX2V v1.2](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |
| Turbo v4 step600 EMA | 4–8，預設 6 | 完整 GGUF | 780 MB | [Larry](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) |

在文生影選擇名稱含「H3 · LightX2V」或「H3 · Turbo v4」的 Profile，下載其相依模型，待就緒後生成。四組共提供 15 個預設，對應 Abiray FL2VA Q4_0／Q4_K_M／Q4_K_S，以及 LightX2V 可使用的 Unsloth FL2VA Pruned Q4_K。預設為 1344×768、124 幀、24 FPS、LoRA 權重 1；測試流程可自行降低尺寸與幀數。主模型、VAE、文字編碼器仍需另行安裝。

選擇 Profile 後使用「套用 Profile 預設值」帶入對應步數；若更改到不適用的步數，生成前會提示修正。Turbo v4 在 4 步的大幅快速動作可能產生拖影，建議保留 6 步或提高至 8 步。每次只使用一個加速 LoRA；目前提供文生影，未開放加速圖生影或 Ref2VA。四組 LoRA 皆採 Apache-2.0 授權。

LightX2V 768p 版本使用影片 shift 6／音訊 shift 3；Turbo v4 使用 12／3，皆採 Euler、simple 時間網格。程式會依 LoRA 自動設定，無需額外 UI 操作。LightX2V 使用 ComfyUI 格式權重；8 步版必須套用每層的 alpha/rank（1/16），4 步版為 1，不能混用縮放。採樣設定依 [LightX2V 模型規格](https://github.com/ModelTC/Minimax-H3-Turbo#1-model-specs) 與 [Turbo v4 作者說明](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)。

Turbo v4 含完整 AdaLN 時間條件權重；本專案尚未移植作者節點的 Pruned 時間條件重建，因此只開放完整 FL2VA。LightX2V 僅更新 Attention／MLP，可對應 Pruned 基底。完整 H3 基底建議 64 GB 以上、Pruned 建議 48 GB 以上記憶體；低步數只減少去噪次數，仍有模型載入、文字編碼、影片解碼與 LoRA 本身的成本，並不等於整段流程等比例加速。

### 其他加速 LoRA 的相容範圍

- 目前 LTX-2.3 MLX Q4／GGUF Q3 Profile 本身已採 8 步蒸餾基底。[官方 LTX-2.3 蒸餾 LoRA](https://huggingface.co/Lightricks/LTX-2.3) 供 Dev 基底轉換使用，此處不重複疊加。
- [H3 PDD](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs) 需要多個輸出頭及專用平行採樣器；[FlashGen](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen) 使用不同時間網格。本次尚未提供這兩者的 Profile，避免以不相容的採樣設定生成。

## 下載與相容性

十三個下載項目固定到已核對的 Hugging Face revision，保存原始檔名與來源 manifest。下載完成後會驗證檔案，再更新模型中心及 Profile 的就緒狀態。文生影 LoRA 不會混入文生圖的 LoRA 選單。

推論維持純 Swift／MLX。LTX 與 H3 使用低秩殘差套用 LoRA，保留原有量化基底；不相容的層、缺少的 A/B 配對、非有限倍率與條件控制請求會明確失敗。Z-Image 量化模型則使用邏輯輸入維度配對 LoRA，並支援這批風格所需的 AdaLN 權重。

## Qwen-Image 2.1 Viggle Turbo v0.3

[Viggle Turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) 的 r128 版本約 680 MB，提供文生圖及單圖編輯兩個 **6 步** Profile。先安裝 Qwen-Image 2.1 MLX 4-bit 基底與 LoRA，再選擇名稱含「Viggle Turbo」的 Profile，並將步數設為 6；Profile 已包含 LoRA，手動選單可保持「無」。固定使用 LoRA 權重 1、CFG=1；不接受負面提示詞。也能搭配名稱含「提示詞增強」的 Profile 使用，增強模型需另行下載。

這是 Qwen 2.1 專用 LoRA，不適用於 Qwen 2511、Z-Image 或影片模型。保留量化基底，以獨立低秩殘差套用；任意 Qwen LoRA、混合多個 LoRA 及其他步數目前不在支援範圍。權重依 Qwen Research License，限非商業研究／評估。

256×256、Seed 42 的實測完成紅色茶壺文生圖及改為藍色的單圖編輯，並能接續切回一般模型生成。這是限定案例驗證，不代表所有題材、高解析度品質或固定加速倍率。詳見 [Qwen 指南](QWEN_IMAGE_21.md)。

## 驗證紀錄（2026-10-03）

- 根套件的模型目錄、下載完整性、能力過濾、Profile 合併及請求驗證回歸通過。另以正式下載器實際安裝 Saturday Morning 與 Dolly In，驗證下載、manifest 與重新掃描流程。
- Z-Image：Saturday Morning 的 240 個 LoRA 目標全部配對成功；affine INT4／MXFP4 的殘差運算、原始權重保留及清除測試通過。
- 文生圖實測使用 andrevp MLX 4-bit、256×256、4 步、Seed 42、LoRA 權重 0.8。同提示詞下，LoRA 輸出與基底圖不同；常駐 Worker 先套用再清除 LoRA、連續生成兩張圖，兩者各自與獨立執行的像素完全一致。
- LTX：Dolly In 的 480 個配對全部對應到實際 LTX-2.3 模型結構；以真實 INT4 `to_q` 權重驗證輸出改變且為有限值。另通過多 LoRA 疊加、alpha、零權重及錯誤輸入拒絕測試。為降低驗證成本，只執行代表性投影層，沒有完整載入 22B 基底或執行整段影片推論。
- GenImage、Z-Image Worker 與 LTX Worker 的 Release 建置通過。

驗證紀錄與本機產物位於 `Outputs/lora-expansion-2026-10-03/`，不納入 Git。**完整影片輸出尚待補齊本機 Gemma 3 文字編碼器後驗證**；目前不將層級測試視為端到端影片驗證。其餘風格與鏡頭已核對公開權重標頭及下載版本，尚未逐一生成。

### H3 低步數驗證補充（2026-10-03）

- 三組 LoRA 的公開 safetensors 標頭逐層對照原生 H3 的預期結構：LightX2V 各 208 層，完整與 Pruned 均相容；Turbo v4 共 259 層，對照完整 FL2VA。
- 以正式下載器實際安裝 LightX2V 8 步版與 Turbo v4，通過 manifest 完整性驗證及本機重新掃描。8 步版的 208 個 alpha/rank 均實測為 0.0625。
- 兩組完整 LoRA 權重皆由 Swift／MLX 載入；以代表性 Attention 投影的人工輸入確認非零、有限的殘差。此測試使用模型定義的形狀，沒有載入真實 H3 基底權重。
- 小型 Dense／INT8 投影測試確認 alpha/rank、LoRA 清除後的輸出、原始量化權重保留及載入失敗時不替換現有 adapter。4／8 步採樣網格、影片與音訊時鐘、非法組合拒絕測試通過。
- 根套件目標回歸通過（82 項，實際下載測試另行執行）；H3 LoRA、排程、結構與量化回歸通過（29 項，實際權重測試另行執行）。Release Worker 另以 256×256、4 幀請求完成 5 組 JSON 協定檢查，僅檢查請求驗證，未生成影片。
- GenImage 與 H3 Worker 的 Release 建置通過。紀錄位於 `Outputs/lowstep-loras-2026-10-03/`，不納入 Git。

**本機尚未安裝完整 H3 基底與配套元件，未進行端到端文生影、畫質或實際加速倍率驗證**。三組 Profile 均標記為實驗性；LightX2V 4 步版本次只驗證公開標頭與採樣規格，尚未下載整份權重逐層執行。

### 1.26.1004 補充驗證

- H3 LightX2V v1.2：完成實際下載、雜湊驗證、安裝 manifest 與重新辨識；完整 624 個張量、208 組低秩配對及代表性殘差計算通過。尚未載入完整 H3 基底生成影片。
- Qwen Turbo：227 組配對全部相容，INT4 投影的增量非零且有限；6 步文生圖／編輯的 256×256 實跑通過。
- LTX-2.5 為獨立的蒸餾基底；目前不接受這裡的 LTX-2.3 鏡頭 LoRA。
