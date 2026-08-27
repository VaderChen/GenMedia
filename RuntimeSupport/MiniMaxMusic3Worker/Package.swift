// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "GenImageMiniMaxMusic3Worker",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MiniMaxMusic3SwiftRuntime", targets: ["MiniMaxMusic3SwiftRuntime"]),
        .executable(name: "GenImageMiniMaxMusic3PoC", targets: ["GenImageMiniMaxMusic3PoC"]),
        .executable(
            name: "GenImageMiniMaxMusic3Worker",
            targets: ["GenImageMiniMaxMusic3Worker"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.31.6"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            exact: "3.31.4"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            exact: "1.3.3"
        )
    ],
    targets: [
        .target(
            name: "MiniMaxMusic3SwiftRuntime",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ]
        ),
        .executableTarget(
            name: "GenImageMiniMaxMusic3PoC",
            dependencies: [
                "MiniMaxMusic3SwiftRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "GenImageMiniMaxMusic3Worker",
            dependencies: [
                "MiniMaxMusic3SwiftRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        ),
        .testTarget(
            name: "MiniMaxMusic3SwiftRuntimeTests",
            dependencies: [
                "MiniMaxMusic3SwiftRuntime",
                .product(name: "MLX", package: "mlx-swift")
            ]
        )
    ]
)
