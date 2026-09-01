import Foundation

public enum ModelCatalog {
    public static let builtIn: [ModelDescriptor] = [
        ModelDescriptor(
            id: "mzbac/z-image-turbo-8bit",
            displayName: "Z-Image Turbo 8-bit",
            publisher: "mzbac / Tongyi-MAI",
            summary: "適合一般 Apple Silicon Mac 的量化文生圖版本。",
            capabilities: [.textToImage],
            quantization: .eightBit,
            approximateDownloadGB: 12.4,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/mzbac/z-image-turbo-8bit"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "andrevp/Z-Image-Turbo-MLX-2bit",
            displayName: "Z-Image Turbo MLX 2-bit",
            publisher: "andrevp",
            summary: "Z-Image Turbo 的 MLX 2-bit 量化版本；屬於實驗性低記憶體配置。",
            capabilities: [.textToImage],
            quantization: .twoBit,
            approximateDownloadGB: 3.8,
            recommendedMemoryGB: 12,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/andrevp/Z-Image-Turbo-MLX-2bit")
        ),
        ModelDescriptor(
            id: "andrevp/Z-Image-Turbo-MLX-4bit",
            displayName: "Z-Image Turbo MLX 4-bit",
            publisher: "andrevp",
            summary: "Z-Image Turbo 的 MLX 4-bit 量化版本，使用 quantize_config 自動建立 Runtime Manifest。",
            capabilities: [.textToImage],
            quantization: .fourBit,
            approximateDownloadGB: 6.0,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/andrevp/Z-Image-Turbo-MLX-4bit")
        ),
        ModelDescriptor(
            id: "andrevp/Z-Image-Turbo-MLX-8bit",
            displayName: "Z-Image Turbo MLX 8-bit",
            publisher: "andrevp",
            summary: "Z-Image Turbo 的 MLX 8-bit 量化版本，使用 quantize_config 自動建立 Runtime Manifest。",
            capabilities: [.textToImage],
            quantization: .eightBit,
            approximateDownloadGB: 10.6,
            recommendedMemoryGB: 24,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/andrevp/Z-Image-Turbo-MLX-8bit")
        ),
        ModelDescriptor(
            id: "Giniiki/Z-Image-Turbo-mlx-4bit",
            displayName: "Z-Image Turbo MLX 4-bit · Giniiki",
            publisher: "Giniiki",
            summary: "以 MLX affine 4-bit 量化的 Z-Image Turbo，保留原始 tensor 名稱並由設定檔建立 Manifest。",
            capabilities: [.textToImage],
            quantization: .fourBit,
            approximateDownloadGB: 5.5,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/Giniiki/Z-Image-Turbo-mlx-4bit")
        ),
        ModelDescriptor(
            id: "Tongyi-MAI/Z-Image-Turbo",
            displayName: "Z-Image Turbo 原始版",
            publisher: "Tongyi-MAI",
            summary: "官方原始權重，適合記憶體較大的 Apple Silicon Mac。",
            capabilities: [.textToImage, .controlNet],
            quantization: .fp16,
            approximateDownloadGB: 32.9,
            recommendedMemoryGB: 32,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo")
        ),
        ModelDescriptor(
            id: "tarn59/pixel_art_style_lora_z_image_turbo",
            displayName: "Z-Image Turbo Pixel Art LoRA",
            publisher: "tarn59",
            summary: "可直接搭配 Z-Image Turbo 使用的像素藝術風格 LoRA。",
            capabilities: [.lora],
            quantization: .lora,
            approximateDownloadGB: 0.16,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/tarn59/pixel_art_style_lora_z_image_turbo"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "suayptalha/Z-Image-Turbo-Realism-LoRA",
            displayName: "Z-Image Turbo Realism LoRA",
            publisher: "suayptalha",
            summary: "以 Realism 觸發詞增強人物與場景的寫實質感；基於官方 Z-Image-Turbo 訓練。",
            capabilities: [.lora],
            quantization: .lora,
            approximateDownloadGB: 0.085,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/suayptalha/Z-Image-Turbo-Realism-LoRA")
        ),
        ModelDescriptor(
            id: "renderartist/Classic-Painting-Z-Image-Turbo-LoRA",
            displayName: "Z-Image Turbo Classic Painting LoRA",
            publisher: "renderartist",
            summary: "以 class1cpa1nt 觸發詞套用古典油畫與博物館修復質感；基於官方 Z-Image-Turbo。",
            capabilities: [.lora],
            quantization: .lora,
            approximateDownloadGB: 0.170,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/renderartist/Classic-Painting-Z-Image-Turbo-LoRA")
        ),
        ModelDescriptor(
            id: "renderartist/Coloring-Book-Z-Image-Turbo-LoRA",
            displayName: "Z-Image Turbo Coloring Book LoRA",
            publisher: "renderartist",
            summary: "以 c0l0ringb00k 觸發詞產生黑白線稿與著色書風格；基於官方 Z-Image-Turbo。",
            capabilities: [.lora],
            quantization: .lora,
            approximateDownloadGB: 0.032,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/renderartist/Coloring-Book-Z-Image-Turbo-LoRA")
        ),
        ModelDescriptor(
            id: "civitai/2465401",
            displayName: "Civitai · Z-Image Asian Beauties",
            publisher: "DeViLDoNia / Civitai",
            summary: "Civitai 的 ZImageTurbo 人像 LoRA；可搭配現有 Z-Image Turbo 文生圖 Profile 使用。",
            capabilities: [.lora],
            quantization: .lora,
            approximateDownloadGB: 0.170,
            recommendedMemoryGB: 16,
            licenseName: "Civitai：Image / RentCivit",
            sourceURL: URL(string: "https://civitai.com/models/785643")
        ),
        ModelDescriptor(
            id: "civitai/2709343",
            displayName: "Civitai · Z-Image Turbo Lightning",
            publisher: "Felldude / Civitai",
            summary: "Civitai 的 ZImageTurbo Lightning LoRA，適合低步數快速生成；建議搭配 4–8 steps。",
            capabilities: [.lora],
            quantization: .lora,
            approximateDownloadGB: 0.035,
            recommendedMemoryGB: 16,
            licenseName: "Civitai：Image / RentCivit",
            sourceURL: URL(string: "https://civitai.com/models/2409672")
        ),
        ModelDescriptor(
            id: "civitai/2449645",
            displayName: "Civitai · Z-Image Flat AnimeStyle",
            publisher: "MenRiVy1 / Civitai",
            summary: "Civitai 的 ZImageTurbo 平面動畫風格 LoRA，支援 UU 觸發詞。",
            capabilities: [.lora],
            quantization: .lora,
            approximateDownloadGB: 0.170,
            recommendedMemoryGB: 16,
            licenseName: "Civitai：Image / RentCivit / Rent / Sell",
            sourceURL: URL(string: "https://civitai.com/models/2175307")
        ),
        ModelDescriptor(
            id: "civitai/2608073",
            displayName: "Civitai · Z-Image Diorama",
            publisher: "loonalone / Civitai",
            summary: "Civitai 的 ZImageTurbo Diorama 立體場景風格 LoRA。",
            capabilities: [.lora],
            quantization: .lora,
            approximateDownloadGB: 0.170,
            recommendedMemoryGB: 16,
            licenseName: "Civitai：RentCivit / Rent",
            sourceURL: URL(string: "https://civitai.com/models/2318236")
        ),
        ModelDescriptor(
            id: "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
            displayName: "LTX-2.3 IC-LoRA Union Control",
            publisher: "Lightricks",
            summary: "LTX-2.3 官方 Union Control LoRA，可用 Canny、深度或姿態控制逐幀結構，降低圖生影的人物與場景漂移。",
            capabilities: [.lora],
            quantization: .lora,
            approximateDownloadGB: 0.61,
            recommendedMemoryGB: 24,
            licenseName: "LTX-2 Community License",
            sourceURL: URL(string: "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "local-captioner-3b@q4",
            displayName: "Qwen3-VL 4B 4-bit",
            publisher: "Qwen / MLX Community",
            summary: "Qwen3-VL 多模態模型，可進行圖片理解、多語言描述與一般文字生成。",
            capabilities: [.imageToText, .textToText],
            quantization: .fourBit,
            approximateDownloadGB: 2.9,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-VL-4B-Instruct-4bit")
        ),
        ModelDescriptor(
            id: "qwen3-vl-8b-nsfw-caption-v45@mxfp4",
            displayName: "Qwen3-VL 8B NSFW Caption V4.5 mxfp4",
            publisher: "Disty0 / MLX Community",
            summary: "Qwen3-VL NSFW Caption V4.5 多模態 MLX mxfp4 版本；支援圖片理解與文字生成。",
            capabilities: [.imageToText, .textToText],
            quantization: .fourBit,
            approximateDownloadGB: 5.52,
            recommendedMemoryGB: 24,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-VL-8B-NSFW-Caption-V4.5-mxfp4")
        ),
        ModelDescriptor(
            id: "lmstudio-community/Qwen3.5-4B-MLX-4bit",
            displayName: "Qwen3.5 4B MLX 4-bit",
            publisher: "LM Studio Community / Qwen",
            summary: "Apple Silicon 原生 MLX 多模態模型，支援圖片理解、字幕翻譯、摘要、改寫與文字生成。",
            capabilities: [.imageToText, .textToText],
            quantization: .fourBit,
            approximateDownloadGB: 2.85,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/lmstudio-community/Qwen3.5-4B-MLX-4bit"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "lmstudio-community/Qwen3.5-9B-MLX-4bit",
            displayName: "Qwen3.5 9B MLX 4-bit",
            publisher: "LM Studio Community / Qwen",
            summary: "較高品質的 Apple Silicon 原生 MLX 多模態模型，支援圖片理解、長字幕翻譯與文字校訂。",
            capabilities: [.imageToText, .textToText],
            quantization: .fourBit,
            approximateDownloadGB: 5.57,
            recommendedMemoryGB: 24,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/lmstudio-community/Qwen3.5-9B-MLX-4bit")
        ),
        ModelDescriptor(
            id: "lmstudio-community/Qwen3.8-27B-MLX-4bit",
            displayName: "Qwen3.8 27B MLX 4-bit",
            publisher: "LM Studio Community / Qwen",
            summary: "大型 Apple Silicon 原生 MLX 多模態模型，支援圖片理解、高品質翻譯、潤稿與複雜文字處理。",
            capabilities: [.imageToText, .textToText],
            quantization: .fourBit,
            approximateDownloadGB: 14.98,
            recommendedMemoryGB: 64,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit")
        ),
        ModelDescriptor(
            id: "qwen-image-edit-2511@mlx-int4",
            displayName: "Qwen Image Edit 2511 INT4",
            publisher: "Qwen / xocialize",
            summary: "官方 2511 基礎模型搭配預量化 Swift/MLX INT4 權重，可直接在 Apple Silicon 載入。",
            capabilities: [.imageToImage],
            quantization: .fourBit,
            approximateDownloadGB: 35.8,
            recommendedMemoryGB: 32,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/Qwen/Qwen-Image-Edit-2511"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "qwen-image-edit-2511@mlx-int8",
            displayName: "Qwen Image Edit 2511 INT8",
            publisher: "Qwen",
            summary: "下載官方 2511 權重，首次使用時轉換並保存為 Swift/MLX INT8。",
            capabilities: [.imageToImage],
            quantization: .eightBit,
            approximateDownloadGB: 57.8,
            recommendedMemoryGB: 48,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/Qwen/Qwen-Image-Edit-2511")
        ),
        ModelDescriptor(
            id: "qwen-image-edit-2511@mlx-fp16",
            displayName: "Qwen Image Edit 2511 FP16",
            publisher: "Qwen",
            summary: "官方 Qwen Image Edit 2511 BF16/FP16 權重，提供最高品質基準。",
            capabilities: [.imageToImage],
            quantization: .fp16,
            approximateDownloadGB: 57.8,
            recommendedMemoryGB: 64,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/Qwen/Qwen-Image-Edit-2511")
        ),
        ModelDescriptor(
            id: "Lightricks/LTX-2.3@distilled-1.1",
            displayName: "LTX-2.3 Distilled 1.1",
            publisher: "Lightricks",
            summary: "官方開放權重圖生影模型，包含同步音訊生成；安裝項目含 Distilled 1.1、空間升頻器與 Gemma 3 12B 文字編碼器。",
            capabilities: [.imageToVideo],
            quantization: .bf16,
            approximateDownloadGB: 66.7,
            recommendedMemoryGB: 96,
            licenseName: "LTX-2 Community License / Gemma Terms",
            sourceURL: URL(string: "https://huggingface.co/Lightricks/LTX-2.3"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "dgrauet/ltx-2.3-mlx-q4",
            displayName: "LTX-2.3 MLX Q4",
            publisher: "dgrauet / LTX-2 MLX",
            summary: "原生 MLX INT4 圖生影模型，在 Apple Silicon 上透過 Metal 執行；包含影片、立體聲音訊與 Gemma 3 12B 文字編碼器。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 42.0,
            recommendedMemoryGB: 48,
            licenseName: "LTX-2 Community License / MLX Port MIT",
            sourceURL: URL(string: "https://huggingface.co/dgrauet/ltx-2.3-mlx-q4"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "city96/LTX-Video-0.9.6-distilled-gguf@Q4_K_M",
            displayName: "LTX-Video 0.9.6 GGUF Q4_K_M",
            publisher: "city96 / Lightricks",
            summary: "LTX-Video 0.9.6 2B Distilled 的 GGUF Q4_K_M 權重；下載 Profile 時會一併取得 T5 XXL 文字編碼器、Tokenizer 與 BF16 VAE。",
            capabilities: [.textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 5.70,
            recommendedMemoryGB: 24,
            licenseName: "LTX-Video Community License",
            sourceURL: URL(string: "https://huggingface.co/city96/LTX-Video-0.9.6-distilled-gguf"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "city96/t5-v1_1-xxl-encoder-gguf@Q4_K_M",
            displayName: "LTX-Video T5 v1.1 XXL GGUF Q4_K_M",
            publisher: "city96 / Google T5",
            summary: "LTX-Video 0.9.6 使用的 T5 v1.1 XXL Q4_K_M 文字編碼器；主模型 Profile 下載時會自動一併取得。",
            capabilities: [.textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 2.14,
            recommendedMemoryGB: 16,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf")
        ),
        ModelDescriptor(
            id: "city96/LTX-Video-0.9.6-VAE@BF16",
            displayName: "LTX-Video 0.9.6 VAE BF16",
            publisher: "city96 / Lightricks",
            summary: "LTX-Video 0.9.6 的 BF16 Video VAE；主模型 Profile 下載時會自動一併取得。",
            capabilities: [.textToVideo],
            quantization: .bf16,
            approximateDownloadGB: 2.32,
            recommendedMemoryGB: 16,
            licenseName: "LTX-Video Community License",
            sourceURL: URL(string: "https://huggingface.co/city96/LTX-Video-0.9.6-distilled-gguf")
        ),
        ModelDescriptor(
            id: "unsloth/LTX-2.3-GGUF@distilled-1.1-Q3_K_M",
            displayName: "LTX-2.3 22B Distilled 1.1 GGUF Q3_K_M",
            publisher: "Unsloth / Lightricks",
            summary: "LTX-2.3 22B Distilled 1.1 的 GGUF Q3_K_M 權重；下載 Profile 時會一併取得 Video/Audio VAE、Gemma 3、connector 與空間升頻器。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 22.50,
            recommendedMemoryGB: 48,
            licenseName: "LTX-2 Community License",
            sourceURL: URL(string: "https://huggingface.co/unsloth/LTX-2.3-GGUF"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "unsloth/LTX-2.3-GGUF@distilled-1.1-VAE",
            displayName: "LTX-2.3 Distilled 1.1 Video／Audio VAE",
            publisher: "Unsloth / Lightricks",
            summary: "LTX-2.3 Distilled 1.1 配套的 Video VAE 與 Audio VAE；主模型 Profile 下載時會自動一併取得。LTX-2.3 使用 Gemma 文字編碼器，不使用 T5。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .bf16,
            approximateDownloadGB: 1.69,
            recommendedMemoryGB: 16,
            licenseName: "LTX-2 Community License",
            sourceURL: URL(string: "https://huggingface.co/unsloth/LTX-2.3-GGUF")
        ),
        ModelDescriptor(
            id: "pipenetwork/MiniMax-H3-MLX-8bit",
            displayName: "MiniMax H3 MLX Q8",
            publisher: "PipeNetwork / MiniMaxAI",
            summary: "Apple Silicon 原生 MLX 8-bit 權重；安裝內容包含量化 Transformer 與官方 FL2VA 文字編碼器、Video/Audio VAE、Tokenizer。GenMedia 目前僅提供下載管理，尚未接入 H3 Swift 生成 Runtime。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .eightBit,
            approximateDownloadGB: 105.3,
            recommendedMemoryGB: 128,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/pipenetwork/MiniMax-H3-MLX-8bit")
        ),
        ModelDescriptor(
            id: "pipenetwork/MiniMax-H3-MLX-4bit",
            displayName: "MiniMax H3 MLX Q4",
            publisher: "PipeNetwork / MiniMaxAI",
            summary: "Apple Silicon 原生 MLX 4-bit 權重；安裝內容包含量化 Transformer 與官方 FL2VA 文字編碼器、Video/Audio VAE、Tokenizer。GenMedia 目前僅提供下載管理，尚未接入 H3 Swift 生成 Runtime。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 96.0,
            recommendedMemoryGB: 96,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/pipenetwork/MiniMax-H3-MLX-4bit"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "unsloth/MiniMax-H3-GGUF@fl2va-pruned-Q4_K",
            displayName: "MiniMax H3 GGUF FL2VA Pruned Q4_K",
            publisher: "Unsloth / MiniMaxAI",
            summary: "MiniMax H3 FL2VA Pruned GGUF Q4_K；下載此 Profile 時會一併取得相容的 Qwen3-VL Q4_K_M 文字編碼器、Video/Audio VAE，以及 MiniMax 上游的 tokenizer 與影像前處理設定（ComfyUI 轉檔的文字編碼器本身不含 tokenizer）。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 35.45,
            recommendedMemoryGB: 48,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/unsloth/MiniMax-H3-GGUF"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "Abiray/MiniMax-H3-GGUF@fl2va-Q4_0",
            displayName: "MiniMax H3 GGUF FL2VA Q4_0",
            publisher: "Abiray / MiniMaxAI",
            summary: "MiniMax H3 FL2VA GGUF Q4_0；下載此模型時會一併取得同倉庫的 Qwen3-VL Q4_K_M、Video VAE 與 Audio VAE，以及 MiniMax 上游的 tokenizer 與影像前處理設定（ComfyUI 轉檔的文字編碼器本身不含 tokenizer）。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 39.03,
            recommendedMemoryGB: 64,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/Abiray/MiniMax-H3-GGUF"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_M",
            displayName: "MiniMax H3 GGUF FL2VA Q4_K_M",
            publisher: "Abiray / MiniMaxAI",
            summary: "MiniMax H3 FL2VA GGUF Q4_K_M；下載此模型時會一併取得同倉庫的 Qwen3-VL Q4_K_M、Video VAE 與 Audio VAE，以及 MiniMax 上游的 tokenizer 與影像前處理設定（ComfyUI 轉檔的文字編碼器本身不含 tokenizer）。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 40.25,
            recommendedMemoryGB: 64,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/Abiray/MiniMax-H3-GGUF"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_S",
            displayName: "MiniMax H3 GGUF FL2VA Q4_K_S",
            publisher: "Abiray / MiniMaxAI",
            summary: "MiniMax H3 FL2VA GGUF Q4_K_S；下載此模型時會一併取得同倉庫的 Qwen3-VL Q4_K_M、Video VAE 與 Audio VAE，以及 MiniMax 上游的 tokenizer 與影像前處理設定（ComfyUI 轉檔的文字編碼器本身不含 tokenizer）。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 40.24,
            recommendedMemoryGB: 64,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/Abiray/MiniMax-H3-GGUF"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "Abiray/MiniMax-H3-GGUF@ref2va-Q4_0",
            displayName: "MiniMax H3 GGUF Ref2VA Q4_0",
            publisher: "Abiray / MiniMaxAI",
            summary: "MiniMax H3 Ref2VA GGUF Q4_0；支援多模態參考輸入，下載時會一併取得同倉庫的 Qwen3-VL Q4_K_M、Video VAE、Audio VAE，以及 MiniMax 上游的 tokenizer 與影像前處理設定（ComfyUI 轉檔的文字編碼器本身不含 tokenizer）。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 39.03,
            recommendedMemoryGB: 64,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/Abiray/MiniMax-H3-GGUF"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "Abiray/MiniMax-H3-GGUF@ref2va-Q4_K_M",
            displayName: "MiniMax H3 GGUF Ref2VA Q4_K_M",
            publisher: "Abiray / MiniMaxAI",
            summary: "MiniMax H3 Ref2VA GGUF Q4_K_M；支援多模態參考輸入，下載時會一併取得同倉庫的 Qwen3-VL Q4_K_M、Video VAE、Audio VAE，以及 MiniMax 上游的 tokenizer 與影像前處理設定（ComfyUI 轉檔的文字編碼器本身不含 tokenizer）。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 40.24,
            recommendedMemoryGB: 64,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/Abiray/MiniMax-H3-GGUF"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "Abiray/MiniMax-H3-GGUF@ref2va-Q4_K_S",
            displayName: "MiniMax H3 GGUF Ref2VA Q4_K_S",
            publisher: "Abiray / MiniMaxAI",
            summary: "MiniMax H3 Ref2VA GGUF Q4_K_S；支援多模態參考輸入，下載時會一併取得同倉庫的 Qwen3-VL Q4_K_M、Video VAE、Audio VAE，以及 MiniMax 上游的 tokenizer 與影像前處理設定（ComfyUI 轉檔的文字編碼器本身不含 tokenizer）。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 40.24,
            recommendedMemoryGB: 64,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/Abiray/MiniMax-H3-GGUF"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "ACE-Step/Ace-Step1.5",
            displayName: "ACE-Step 1.5 Turbo MLX",
            publisher: "ACE Studio / StepFun",
            summary: "Apple Silicon 原生 MLX 文生音樂模型，支援 Prompt、選填歌詞與最長 300 秒純音樂或歌曲生成。",
            capabilities: [.textToMusic],
            quantization: .bf16,
            approximateDownloadGB: 6.8,
            recommendedMemoryGB: 16,
            licenseName: "MIT",
            sourceURL: URL(string: "https://huggingface.co/ACE-Step/Ace-Step1.5"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "vanch007/MiniMax-Music3-MLX-8bit",
            displayName: "MiniMax Music 3 MLX 8-bit",
            publisher: "vanch007 / MiniMaxAI",
            summary: "Apple Silicon 原生 MLX 8-bit 文生音樂模型，可依音樂風格與歌詞生成 44.1 kHz 立體聲音訊。",
            capabilities: [.textToMusic],
            quantization: .eightBit,
            approximateDownloadGB: 13.2,
            recommendedMemoryGB: 64,
            licenseName: "MiniMax-Music3 Community License",
            sourceURL: URL(string: "https://huggingface.co/vanch007/MiniMax-Music3-MLX-8bit"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "mlx-community/MiniMax-Music3-4bit",
            displayName: "MiniMax Music 3 MLX 4-bit",
            publisher: "MLX Community / MiniMaxAI",
            summary: "Apple Silicon 原生 MLX affine 4-bit 完整文生音樂模型；包含語言模型、條件編碼器、Flow Transformer、RVQ 解碼器與 Vocoder 所需權重。",
            capabilities: [.textToMusic],
            quantization: .fourBit,
            approximateDownloadGB: 8.6,
            recommendedMemoryGB: 24,
            licenseName: "MiniMax Music 3 Community License",
            sourceURL: URL(string: "https://huggingface.co/mlx-community/MiniMax-Music3-4bit"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "audio-cpp/MiniMax-Music3-GGUF@Q4_K-LM-Q4_K-DiT-Q8_0-depth",
            displayName: "MiniMax Music 3 GGUF Q4_K / Q8_0",
            publisher: "audio.cpp / MiniMaxAI",
            summary: "完整 MiniMax Music 3 GGUF 元件包：語言模型與 Flow Transformer 使用 Q4_K，RVQ depth decoder 使用 Q8_0，包含條件編碼器、Vocoder 與 Tokenizer；實際下載約 8.96 GiB。",
            capabilities: [.textToMusic],
            quantization: .fourBit,
            approximateDownloadGB: 8.96,
            recommendedMemoryGB: 16,
            licenseName: "MiniMax-Music3 Community License",
            sourceURL: URL(string: "https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "Mothersuperior/minimax-music3-composer-5.7b-distilled",
            displayName: "MiniMax Music 3 Composer 5.7B Distilled",
            publisher: "Mothersuperior / MiniMaxAI",
            summary: "MiniMax Music 3 的 5.7B depth-pruned distilled Composer 加速元件；預設下載 lr-6e-5 權重，需搭配完整 Music 3 checkpoint 與相容 Runtime，不能單獨生成音樂。",
            capabilities: [.textToMusic],
            quantization: .bf16,
            approximateDownloadGB: 10.6,
            recommendedMemoryGB: 32,
            licenseName: "MiniMax Music 3 Terms",
            sourceURL: URL(string: "https://huggingface.co/Mothersuperior/minimax-music3-composer-5.7b-distilled")
        ),
        ModelDescriptor(
            id: "argmaxinc/whisperkit-coreml@large-v3-turbo",
            displayName: "Whisper Large v3 Turbo Core ML",
            publisher: "Argmax / OpenAI",
            summary: "支援多語言與自動語言偵測的 WhisperKit Core ML 模型，可從影片或音訊產生字幕時間軸。",
            capabilities: [.videoToText],
            quantization: .coreML,
            approximateDownloadGB: 0.96,
            recommendedMemoryGB: 16,
            licenseName: "MIT / Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/argmaxinc/whisperkit-coreml"),
            isRecommended: false
        ),
        ModelDescriptor(
            id: "argmaxinc/whisperkit-coreml@small",
            displayName: "Whisper Small Core ML",
            publisher: "Argmax / OpenAI",
            summary: "較輕量的多語 Whisper Small Core ML 模型，使用較少記憶體並提升字幕辨識速度。",
            capabilities: [.videoToText],
            quantization: .coreML,
            approximateDownloadGB: 0.216,
            recommendedMemoryGB: 8,
            licenseName: "MIT / Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/argmaxinc/whisperkit-coreml"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "FluidInference/paraformer-large-zh-coreml",
            displayName: "Paraformer Large 中文 Core ML",
            publisher: "FluidInference / FunASR",
            summary: "針對中文語音辨識最佳化的 Core ML INT8 模型，使用 Apple Neural Engine 產生逐字時間軸與字幕。",
            capabilities: [.videoToText],
            quantization: .coreML,
            approximateDownloadGB: 0.22,
            recommendedMemoryGB: 8,
            licenseName: "Paraformer Upstream License",
            sourceURL: URL(string: "https://huggingface.co/FluidInference/paraformer-large-zh-coreml"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "FluidInference/parakeet-0.6b-ja-coreml",
            displayName: "Parakeet 0.6B 日文 Core ML",
            publisher: "FluidInference / NVIDIA",
            summary: "針對日文語音辨識最佳化的 Core ML 模型，可產生逐字時間軸與日文字幕。",
            capabilities: [.videoToText],
            quantization: .coreML,
            approximateDownloadGB: 0.62,
            recommendedMemoryGB: 8,
            licenseName: "CC-BY-4.0",
            sourceURL: URL(string: "https://huggingface.co/FluidInference/parakeet-0.6b-ja-coreml"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "realesrgan-x4@coreml",
            displayName: "Real-ESRGAN 4×",
            publisher: "mlboydaisuke / xinntao",
            summary: "Core ML Real-ESRGAN 本機 4× 圖片放大與細節修復。",
            capabilities: [.upscale],
            quantization: .coreML,
            approximateDownloadGB: 0.1,
            recommendedMemoryGB: 8,
            licenseName: "BSD-3-Clause",
            sourceURL: URL(string: "https://huggingface.co/mlboydaisuke/Real-ESRGAN-x4-CoreML")
        ),
        ModelDescriptor(
            id: "realesrgan-x2@coreml",
            displayName: "Real-ESRGAN 2×",
            publisher: "mlboydaisuke / xinntao",
            summary: "使用 Core ML Real-ESRGAN 4× 修復後高品質縮放至 2×，效果較溫和。",
            capabilities: [.upscale],
            quantization: .coreML,
            approximateDownloadGB: 0.1,
            recommendedMemoryGB: 8,
            licenseName: "BSD-3-Clause",
            sourceURL: URL(string: "https://huggingface.co/mlboydaisuke/Real-ESRGAN-x4-CoreML")
        )
    ]

    public static let builtInProfiles: [InferenceProfile] = [
        InferenceProfile(
            name: "快速文生圖 · Z-Image 8-bit",
            capability: .textToImage,
            modelID: "mzbac/z-image-turbo-8bit",
            modelRevision: "main",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: 9, outputCount: 4),
            notes: "16GB Mac 的建議預設。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "實驗文生圖 · Z-Image MLX 2-bit",
            capability: .textToImage,
            modelID: "andrevp/Z-Image-Turbo-MLX-2bit",
            modelRevision: "main",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: 9, outputCount: 1),
            notes: "使用 quantize_config.json 轉換為 Runtime Manifest；2-bit 屬於實驗性配置。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "快速文生圖 · Z-Image MLX 4-bit",
            capability: .textToImage,
            modelID: "andrevp/Z-Image-Turbo-MLX-4bit",
            modelRevision: "main",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: 9, outputCount: 1),
            notes: "使用 quantize_config.json 轉換為 Runtime Manifest。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "高品質文生圖 · Z-Image MLX 8-bit",
            capability: .textToImage,
            modelID: "andrevp/Z-Image-Turbo-MLX-8bit",
            modelRevision: "main",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: 9, outputCount: 1),
            notes: "使用 quantize_config.json 轉換為 Runtime Manifest。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "快速文生圖 · Z-Image MLX 4-bit Giniiki",
            capability: .textToImage,
            modelID: "Giniiki/Z-Image-Turbo-mlx-4bit",
            modelRevision: "main",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: 9, outputCount: 1),
            notes: "transformer 與 text encoder 的 quantization 設定會自動轉換為 Runtime Manifest。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "高品質文生圖 · Z-Image 原始版",
            capability: .textToImage,
            modelID: "Tongyi-MAI/Z-Image-Turbo",
            modelRevision: "main",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: 9, outputCount: 4),
            notes: "適合 24GB 以上的 Mac。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "繁中圖生文 · 3B Q4",
            capability: .imageToText,
            modelID: "local-captioner-3b@q4",
            modelRevision: "main",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 512, languageCode: "zh-Hant"),
            notes: "輸出可直接交給文生圖編輯器。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "NSFW 圖生文 · Qwen3-VL 8B mxfp4",
            capability: .imageToText,
            modelID: "qwen3-vl-8b-nsfw-caption-v45@mxfp4",
            modelRevision: "V4.5-mxfp4",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 512, languageCode: "zh-Hant"),
            notes: "MLX Community 轉換版；內容標記為 Not-For-All-Audiences，請確認使用環境與內容政策。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生文 · Qwen3-VL 4B 4-bit",
            capability: .textToText,
            modelID: "local-captioner-3b@q4",
            modelRevision: "main",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 2_048, languageCode: "auto"),
            notes: "多模態 VLM Runtime 的純文字模式，可用於摘要、改寫與一般文字生成。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "NSFW 文生文 · Qwen3-VL 8B mxfp4",
            capability: .textToText,
            modelID: "qwen3-vl-8b-nsfw-caption-v45@mxfp4",
            modelRevision: "V4.5-mxfp4",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 2_048, languageCode: "auto"),
            notes: "多模態 VLM Runtime 的純文字模式；內容標記為 Not-For-All-Audiences。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生文 · Qwen3.5 4B MLX 4-bit",
            capability: .imageToText,
            modelID: "lmstudio-community/Qwen3.5-4B-MLX-4bit",
            modelRevision: "c43ee1d65576a5d98de1e8405cac93c371a655c1",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 512, languageCode: "zh-Hant"),
            notes: "建議的原生 MLX 多模態圖生文 Profile。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生文 · Qwen3.5 9B MLX 4-bit",
            capability: .imageToText,
            modelID: "lmstudio-community/Qwen3.5-9B-MLX-4bit",
            modelRevision: "b455506b0f574c74616dbcd56879bde38fafcff3",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 512, languageCode: "zh-Hant"),
            notes: "較高品質的原生 MLX 多模態圖生文 Profile。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生文 · Qwen3.8 27B MLX 4-bit",
            capability: .imageToText,
            modelID: "lmstudio-community/Qwen3.8-27B-MLX-4bit",
            modelRevision: "6067b15cf581666a4aecf6af3afaba4bb5efc20c",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 1_024, languageCode: "zh-Hant"),
            notes: "大型原生 MLX 多模態圖生文 Profile，建議 64GB 記憶體。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生文 · Qwen3.5 4B MLX 4-bit",
            capability: .textToText,
            modelID: "lmstudio-community/Qwen3.5-4B-MLX-4bit",
            modelRevision: "c43ee1d65576a5d98de1e8405cac93c371a655c1",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 2_048, languageCode: "auto"),
            notes: "多模態 VLM Runtime 的純文字模式；停用 thinking，適合字幕翻譯、摘要、改寫與一般文字生成。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生文 · Qwen3.5 9B MLX 4-bit",
            capability: .textToText,
            modelID: "lmstudio-community/Qwen3.5-9B-MLX-4bit",
            modelRevision: "b455506b0f574c74616dbcd56879bde38fafcff3",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 4_096, languageCode: "auto"),
            notes: "多模態 VLM Runtime 的純文字模式，適合長字幕翻譯與文字校訂。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生文 · Qwen3.8 27B MLX 4-bit",
            capability: .textToText,
            modelID: "lmstudio-community/Qwen3.8-27B-MLX-4bit",
            modelRevision: "6067b15cf581666a4aecf6af3afaba4bb5efc20c",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(maxTokens: 8_192, languageCode: "auto"),
            notes: "大型多模態 VLM Runtime 的純文字模式；適合高品質字幕翻譯、潤稿與複雜文字處理，建議 64GB 記憶體。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生圖 · Qwen Image Edit INT4",
            capability: .imageToImage,
            modelID: "qwen-image-edit-2511@mlx-int4",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: 20, outputCount: 1),
            notes: "Qwen Image Edit 2511 原生 Swift/MLX；建議 32GB 以上 Apple Silicon Mac。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生圖 · Qwen Image Edit INT8",
            capability: .imageToImage,
            modelID: "qwen-image-edit-2511@mlx-int8",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: 20, outputCount: 1),
            notes: "首次使用會將官方 2511 權重轉存為 MLX INT8；建議 48GB 以上記憶體。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生圖 · Qwen Image Edit FP16",
            capability: .imageToImage,
            modelID: "qwen-image-edit-2511@mlx-fp16",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(width: 1024, height: 1024, steps: 20, outputCount: 1),
            notes: "官方 Qwen Image Edit 2511 完整精度 Profile；建議 64GB 以上記憶體。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生影 · LTX-2.3 Distilled",
            capability: .imageToVideo,
            modelID: "Lightricks/LTX-2.3@distilled-1.1",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 720,
                steps: 8,
                outputCount: 1,
                frameCount: 121,
                frameRate: 24
            ),
            notes: "官方 LTX-2.3 Distilled 1.1 圖生影 Profile；5 秒、24 FPS。模型中心會下載主權重、x2 空間升頻器與 Gemma 3 12B 文字編碼器；原生 Worker 目前僅支援 MLX Q4 模型格式。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生影 · LTX-2.3 MLX Q4",
            capability: .imageToVideo,
            modelID: "dgrauet/ltx-2.3-mlx-q4",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 720,
                steps: 8,
                outputCount: 1,
                frameCount: 97,
                frameRate: 24
            ),
            loras: [
                ProfileLoRAConfiguration(
                    modelID: "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
                    scale: 1,
                    conditioning: .sourceImageCanny,
                    conditioningScale: 1
                )
            ],
            notes: "原生 MLX INT4 LTX-2.3 Profile；由 LTX Swift Worker 使用 Apple Silicon Metal 執行。文字生影已支援；image conditioning 與 LoRA fusion 尚未支援，建議 48GB 以上記憶體。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生影 · LTX-2.3 MLX Q4",
            capability: .textToVideo,
            modelID: "dgrauet/ltx-2.3-mlx-q4",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 720,
                steps: 8,
                outputCount: 1,
                frameCount: 97,
                frameRate: 24
            ),
            notes: "原生 MLX INT4 LTX-2.3 文生影 Profile；由 LTX Swift Worker 使用 Apple Silicon Metal 執行。建議 48GB 以上記憶體。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生影 · LTX-Video 0.9.6 GGUF Q4_K_M",
            capability: .textToVideo,
            modelID: "city96/LTX-Video-0.9.6-distilled-gguf@Q4_K_M",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 720,
                steps: 8,
                outputCount: 1,
                frameCount: 97,
                frameRate: 24
            ),
            notes: "LTX-Video 0.9.6 GGUF Q4_K_M Profile；安裝時會一併下載 T5 v1.1 XXL Q4_K_M、Tokenizer 與 BF16 VAE，由 LTX GGUF Swift Runtime 生成。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生影 · LTX-2.3 GGUF Q3_K_M",
            capability: .imageToVideo,
            modelID: "unsloth/LTX-2.3-GGUF@distilled-1.1-Q3_K_M",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 720,
                steps: 8,
                outputCount: 1,
                frameCount: 97,
                frameRate: 24
            ),
            notes: "LTX-2.3 GGUF Q3_K_M Profile；安裝時會一併下載 Video/Audio VAE、Gemma 3 tokenizer、connector 與空間升頻器，由 LTX GGUF Swift Runtime 生成。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生影 · LTX-2.3 GGUF Q3_K_M",
            capability: .textToVideo,
            modelID: "unsloth/LTX-2.3-GGUF@distilled-1.1-Q3_K_M",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 720,
                steps: 8,
                outputCount: 1,
                frameCount: 97,
                frameRate: 24
            ),
            notes: "LTX-2.3 GGUF Q3_K_M Profile；安裝時會一併下載 Video/Audio VAE、Gemma 3 tokenizer、connector 與空間升頻器，由 LTX GGUF Swift Runtime 生成。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生影 · MiniMax H3 MLX Q8",
            capability: .imageToVideo,
            modelID: "pipenetwork/MiniMax-H3-MLX-8bit",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 704,
                steps: 16,
                outputCount: 1,
                frameCount: 124,
                frameRate: 24
            ),
            notes: "PipeNetwork MiniMax H3 原生 MLX 8-bit 權重下載 Profile；GenMedia 尚未接入 H3 Swift 生成 Runtime，目前僅供下載。",
            isBuiltIn: true,
            supportsGeneration: false
        ),
        InferenceProfile(
            name: "圖生影 · MiniMax H3 MLX Q4",
            capability: .imageToVideo,
            modelID: "pipenetwork/MiniMax-H3-MLX-4bit",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 704,
                steps: 16,
                outputCount: 1,
                frameCount: 124,
                frameRate: 24
            ),
            notes: "PipeNetwork MiniMax H3 原生 MLX 4-bit 權重下載 Profile；GenMedia 尚未接入 H3 Swift 生成 Runtime，目前僅供下載。",
            isBuiltIn: true,
            supportsGeneration: false
        ),
        InferenceProfile(
            name: "圖生影 · MiniMax H3 GGUF Q4_K",
            capability: .imageToVideo,
            modelID: "unsloth/MiniMax-H3-GGUF@fl2va-pruned-Q4_K",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 704,
                steps: 16,
                outputCount: 1,
                frameCount: 124,
                frameRate: 24
            ),
            notes: "MiniMax H3 GGUF FL2VA Pruned Q4_K；安裝時會一併下載相容的 Qwen3-VL Q4_K_M、Video VAE、Audio VAE 與 tokenizer；由 H3 Swift Runtime 支援文生影與單一圖片圖生影。",
            isBuiltIn: true,
            supportsGeneration: true
        ),
        InferenceProfile(
            name: "文生影 · MiniMax H3 GGUF Q4_K",
            capability: .textToVideo,
            modelID: "unsloth/MiniMax-H3-GGUF@fl2va-pruned-Q4_K",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 704,
                steps: 16,
                outputCount: 1,
                frameCount: 124,
                frameRate: 24
            ),
            notes: "MiniMax H3 GGUF FL2VA Pruned Q4_K；安裝時會一併下載相容的 Qwen3-VL Q4_K_M、Video VAE、Audio VAE 與 tokenizer；由 H3 Swift Runtime 支援文生影與單一圖片圖生影。",
            isBuiltIn: true,
            supportsGeneration: true
        ),
        InferenceProfile(
            name: "文生音樂 · ACE-Step 1.5 Turbo MLX",
            capability: .textToMusic,
            modelID: "ACE-Step/Ace-Step1.5",
            modelRevision: "19671f406d603126926c1b7e2adc169acbcade22",
            architecture: .mlxSwift,
            defaults: ProfileDefaults(steps: 8, outputCount: 1, durationSeconds: 30),
            music: ProfileMusicConfiguration(
                minimumDurationSeconds: 10,
                maximumDurationSeconds: 300,
                durationSemantics: .target
            ),
            notes: "ACE-Step 1.5 Turbo Apple Silicon 原生 Swift／MLX Profile；支援音樂 Prompt、選填歌詞與純音樂生成，長度 10–300 秒。輸出由 App 內建 FFmpeg 轉為 MP3、M4A、AAC 或 FLAC。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生音樂 · MiniMax Music 3 MLX 8-bit",
            capability: .textToMusic,
            modelID: "vanch007/MiniMax-Music3-MLX-8bit",
            modelRevision: "57d87a63181336634a9557fd31aacc2ad6762935",
            architecture: .externalCLI,
            defaults: ProfileDefaults(steps: 20, outputCount: 1, durationSeconds: 10),
            music: ProfileMusicConfiguration(
                minimumDurationSeconds: 5,
                maximumDurationSeconds: 300,
                durationSemantics: .maximum
            ),
            notes: "MiniMax Music 3 原生 MLX 8-bit Profile；依音樂風格 Prompt 與歌詞生成最長 5–300 秒的 44.1 kHz 立體聲音訊，模型可能依歌曲結構提前自然結束。建議 64GB 記憶體；由 App 隨附的 Swift Worker 執行，輸出轉碼使用 App 內建 FFmpeg。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生音樂 · MiniMax Music 3 MLX 4-bit",
            capability: .textToMusic,
            modelID: "mlx-community/MiniMax-Music3-4bit",
            modelRevision: "c7ea32923b245fe5afc22d740a1936ad2ac590f3",
            architecture: .externalCLI,
            defaults: ProfileDefaults(steps: 20, outputCount: 1, durationSeconds: 10),
            music: ProfileMusicConfiguration(
                minimumDurationSeconds: 5,
                maximumDurationSeconds: 300,
                durationSemantics: .maximum
            ),
            notes: "MiniMax Music 3 原生 MLX affine 4-bit Profile；由 App 隨附的 Swift Worker 執行，依音樂風格 Prompt 與歌詞生成最長 5–300 秒音訊。模型可能依歌曲結構提前自然結束；Composer 加速元件目前獨立管理，尚未自動套用。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生音樂 · MiniMax Music 3 GGUF Q4_K / Q8_0",
            capability: .textToMusic,
            modelID: "audio-cpp/MiniMax-Music3-GGUF@Q4_K-LM-Q4_K-DiT-Q8_0-depth",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(steps: 20, outputCount: 1, durationSeconds: 10),
            music: ProfileMusicConfiguration(
                minimumDurationSeconds: 5,
                maximumDurationSeconds: 300,
                durationSemantics: .maximum
            ),
            notes: "MiniMax Music 3 GGUF Q4_K／Q8_0 Profile；由 App 隨附的 Swift Worker 直接載入 GGUF，語言模型與 Flow Transformer 使用 INT4，RVQ depth decoder 使用 INT8，條件編碼器與 Vocoder 保持原生浮點格式。建議 16GB 以上 Apple Silicon Mac。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "多語字幕 · Whisper Large v3 Turbo",
            capability: .videoToText,
            modelID: "argmaxinc/whisperkit-coreml@large-v3-turbo",
            modelRevision: "0f63a7800b00dd0226abd051b906c246e1907482",
            architecture: .coreML,
            defaults: ProfileDefaults(languageCode: "auto"),
            notes: "多語言 Core ML Profile；自動偵測來源語言並輸出字幕時間軸。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "多語字幕 · Whisper Small",
            capability: .videoToText,
            modelID: "argmaxinc/whisperkit-coreml@small",
            modelRevision: "0f63a7800b00dd0226abd051b906c246e1907482",
            architecture: .coreML,
            defaults: ProfileDefaults(languageCode: "auto"),
            notes: "較輕量的多語言 Core ML Profile；以較低記憶體需求換取更快的字幕辨識速度。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "中文字幕 · Paraformer Large Core ML",
            capability: .videoToText,
            modelID: "FluidInference/paraformer-large-zh-coreml",
            modelRevision: "5dd557bd06342a3cd07ceccb909d8a45e48b053a",
            architecture: .coreML,
            defaults: ProfileDefaults(languageCode: "zh"),
            notes: "中文語音專用 Core ML INT8 Profile；由 Apple Neural Engine 執行，支援長音訊分段辨識與字幕時間軸合併。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "日文字幕 · Parakeet 0.6B Core ML",
            capability: .videoToText,
            modelID: "FluidInference/parakeet-0.6b-ja-coreml",
            modelRevision: "2952296ff1da4a6d6a7aec545e226367db80c612",
            architecture: .coreML,
            defaults: ProfileDefaults(languageCode: "ja"),
            notes: "日文語音專用 Core ML Profile；輸出逐字時間軸並整理為日文字幕。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "一般照片放大 · Real-ESRGAN 4×",
            capability: .upscale,
            modelID: "realesrgan-x4@coreml",
            modelRevision: "1",
            architecture: .coreML,
            defaults: ProfileDefaults(upscaleScale: 4, tileSize: 512),
            notes: "使用分塊降低記憶體需求。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "一般照片放大 · Real-ESRGAN 2×",
            capability: .upscale,
            modelID: "realesrgan-x2@coreml",
            modelRevision: "1",
            architecture: .coreML,
            defaults: ProfileDefaults(upscaleScale: 2, tileSize: 512),
            notes: "先以 Real-ESRGAN 4× 修復，再以 Lanczos 縮放為 2×。",
            isBuiltIn: true
        )
    ] + miniMaxH3GGUFProfiles

    private static var miniMaxH3GGUFProfiles: [InferenceProfile] {
        let variants: [(modelID: String, displayName: String)] = [
            (
                "Abiray/MiniMax-H3-GGUF@fl2va-Q4_0",
                "MiniMax H3 GGUF FL2VA Q4_0"
            ),
            (
                "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_M",
                "MiniMax H3 GGUF FL2VA Q4_K_M"
            ),
            (
                "Abiray/MiniMax-H3-GGUF@fl2va-Q4_K_S",
                "MiniMax H3 GGUF FL2VA Q4_K_S"
            ),
            (
                "Abiray/MiniMax-H3-GGUF@ref2va-Q4_0",
                "MiniMax H3 GGUF Ref2VA Q4_0"
            ),
            (
                "Abiray/MiniMax-H3-GGUF@ref2va-Q4_K_M",
                "MiniMax H3 GGUF Ref2VA Q4_K_M"
            ),
            (
                "Abiray/MiniMax-H3-GGUF@ref2va-Q4_K_S",
                "MiniMax H3 GGUF Ref2VA Q4_K_S"
            )
        ]
        let revision = "9fc3454d3ebe1be1bade862cd4a5011f325a22cb"
        let defaults = ProfileDefaults(
            width: 1280,
            height: 704,
            steps: 16,
            outputCount: 1,
            frameCount: 124,
            frameRate: 24
        )

        return variants.flatMap { variant in
            let isFL2VA = variant.modelID.lowercased().contains("@fl2va-")
            let notes = isFL2VA
                ? "\(variant.displayName)；安裝時會一併下載同倉庫的 Qwen3-VL Q4_K_M、Video VAE 與 Audio VAE；目前可用 H3 Swift Runtime 進行文生影與單一圖片圖生影。"
                : "\(variant.displayName)；安裝時會一併下載同倉庫的 Qwen3-VL Q4_K_M、Video VAE 與 Audio VAE；H3 Swift Runtime 支援文生影，Ref2VA 的圖片參考條件尚未接入。"
            return [
                InferenceProfile(
                    name: "圖生影 · \(variant.displayName)",
                    capability: .imageToVideo,
                    modelID: variant.modelID,
                    modelRevision: revision,
                    architecture: .externalCLI,
                    defaults: defaults,
                    notes: notes,
                    isBuiltIn: true,
                    supportsGeneration: isFL2VA
                ),
                InferenceProfile(
                    name: "文生影 · \(variant.displayName)",
                    capability: .textToVideo,
                    modelID: variant.modelID,
                    modelRevision: revision,
                    architecture: .externalCLI,
                    defaults: defaults,
                    notes: notes,
                    isBuiltIn: true,
                    supportsGeneration: true
                )
            ]
        }
    }
}
