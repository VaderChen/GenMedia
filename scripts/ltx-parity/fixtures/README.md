# LTX-2 Transformer parity fixtures

這組 fixture 用來驗證 Block 2 的真實多模態序列，不使用單 token 退化案例。
由 `scripts/ltx-parity/run-transformer-matrix.command` 產生；該腳本只會讀取
模型目錄，並將 Python 參考輸出寫在本目錄、將 Swift 暫存輸出寫在 `$TMPDIR`。

## 固定規模

| Case | Video grid | Video tokens | Audio tokens | Text tokens | Tiled temporal layout |
|---|---:|---:|---:|---:|---|
| `small` | `4x8x8` | `256` | `256` | `64` | `2 tiles`, overlap `1` |
| `medium` | `8x16x16` | `2048` | `256` | `64` | `2 tiles`, overlap `1` |
| `large` | `16x16x32` | `8192` | `256` | `64` | `2 tiles`, overlap `1` |

`medium` 是預設代表性規模；`large` 用來確認誤差不會隨序列長度發散。
單 token 案例仍可透過 `dump.py` 的明確參數產生，但不屬於常設回歸 fixture。

每個 case 都會各產生 `bfloat16` 與 `float32` 兩份 Python fixture，並分別執行
`transformer-forward`（Level 2）與 `transformer-tiled`（Level 3）。權重參數
`--weights` 一律是相對於 `--model-dir` 的檔名，例如：

```bash
scripts/ltx-parity/run-transformer-matrix.command \
  "/Users/vader/AI Modes/Managa/ltx-2.3-mlx-q4"
```

也可以用 `LTX_PARITY_PYTHON` 與 `LTX_PARITY_SWIFT` 覆寫兩個執行檔位置。

## 有效計算精度

Python `LTXModel.__call__` 入口會把 latent、timestep 與 text embedding 統一
轉成 BF16；因此即使 fixture 的 `input_dtype` 是 `float32`，metadata 仍會記錄
`effective_input_dtype=bfloat16`。這代表該路徑不是 FP32 計算，權重中的
BF16/FP32/UINT32 只是來源儲存格式，不能單獨用來判定比較門檻。

`compare` 優先讀取 `effective_input_dtype`，再回退到 `input_dtype`；逐點張量
比較使用 BF16 `1e-2`、FP32 `1e-4` 門檻。最終媒體輸出不適用這套規則，應使用
對應的能量指標門檻。
