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
            summary: "以 Qwen3-VL 將圖片轉成可編輯的多語言描述。",
            capabilities: [.imageToText],
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
            summary: "Qwen3-VL NSFW Caption V4.5 的 MLX mxfp4 版本；適合需要無審核內容描述的圖生文工作流。",
            capabilities: [.imageToText],
            quantization: .fourBit,
            approximateDownloadGB: 5.52,
            recommendedMemoryGB: 24,
            licenseName: "Apache-2.0",
            sourceURL: URL(string: "https://huggingface.co/mlx-community/Qwen3-VL-8B-NSFW-Caption-V4.5-mxfp4")
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
            summary: "原生 MLX INT4 圖生影模型，在 Apple Silicon 上透過 Metal 執行；包含影片與立體聲音訊生成所需元件。",
            capabilities: [.imageToVideo, .textToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 20.5,
            recommendedMemoryGB: 24,
            licenseName: "LTX-2 Community License / MLX Port MIT",
            sourceURL: URL(string: "https://huggingface.co/dgrauet/ltx-2.3-mlx-q4"),
            isRecommended: true
        ),
        ModelDescriptor(
            id: "pipenetwork/MiniMax-H3-MLX-8bit",
            displayName: "MiniMax H3 MLX Q8",
            publisher: "PipeNetwork / MiniMaxAI",
            summary: "Apple Silicon 原生 MLX 8-bit 圖生影模型，可同步生成影片與立體聲音訊；安裝內容包含量化 Transformer 與官方 FL2VA 文字編碼器、Video/Audio VAE、Tokenizer。",
            capabilities: [.imageToVideo],
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
            summary: "Apple Silicon 原生 MLX 4-bit 圖生影模型，可同步生成影片與立體聲音訊；安裝內容包含量化 Transformer 與官方 FL2VA 文字編碼器、Video/Audio VAE、Tokenizer。",
            capabilities: [.imageToVideo],
            quantization: .fourBit,
            approximateDownloadGB: 96.0,
            recommendedMemoryGB: 96,
            licenseName: "MiniMax H3 Community License",
            sourceURL: URL(string: "https://huggingface.co/pipenetwork/MiniMax-H3-MLX-4bit"),
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
            notes: "官方 LTX-2.3 Distilled 1.1 圖生影 Profile；5 秒、24 FPS。模型中心會下載主權重、x2 空間升頻器與 Gemma 3 12B 文字編碼器，推論仍需官方 Python Runtime。",
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
            notes: "原生 MLX INT4 LTX-2.3 Profile；預設使用 Union Control IC-LoRA 與來源圖片 Canny 控制影片，降低逐幀人物與場景漂移。由 ltx-2-mlx 使用 Apple Silicon Metal 執行，建議 24GB 以上記憶體。",
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
            notes: "原生 MLX INT4 LTX-2.3 文生影 Profile；由 ltx-2-mlx 使用 Apple Silicon Metal 執行。建議 24GB 以上記憶體，需安裝 ltx-2-mlx Python Runtime。",
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
                height: 720,
                steps: 16,
                outputCount: 1,
                frameCount: 124,
                frameRate: 24
            ),
            notes: "PipeNetwork MiniMax H3 原生 MLX 8-bit Profile；預設約 5 秒、24 FPS、768p，支援首幀或首尾幀圖生影與同步音訊。推論需 pipenetwork/minimax-h3-mlx Python Runtime。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "圖生影 · MiniMax H3 MLX Q4",
            capability: .imageToVideo,
            modelID: "pipenetwork/MiniMax-H3-MLX-4bit",
            modelRevision: "main",
            architecture: .externalCLI,
            defaults: ProfileDefaults(
                width: 1280,
                height: 720,
                steps: 16,
                outputCount: 1,
                frameCount: 124,
                frameRate: 24
            ),
            notes: "PipeNetwork MiniMax H3 原生 MLX 4-bit Profile；預設約 5 秒、24 FPS、768p，支援首幀或首尾幀圖生影與同步音訊。推論需 pipenetwork/minimax-h3-mlx Python Runtime。",
            isBuiltIn: true
        ),
        InferenceProfile(
            name: "文生音樂 · MiniMax Music 3 MLX 8-bit",
            capability: .textToMusic,
            modelID: "vanch007/MiniMax-Music3-MLX-8bit",
            modelRevision: "57d87a63181336634a9557fd31aacc2ad6762935",
            architecture: .externalCLI,
            defaults: ProfileDefaults(steps: 30, outputCount: 1, durationSeconds: 10),
            notes: "MiniMax Music 3 原生 MLX 8-bit Profile；依音樂風格 Prompt 與歌詞生成 44.1 kHz 立體聲音訊，建議 64GB 記憶體。推論需外部 mlx-minimax-music3 Runtime 與 FFmpeg。",
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
    ]
}
