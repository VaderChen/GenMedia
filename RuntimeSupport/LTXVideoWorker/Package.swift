// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "GenImageLTXVideoWorker",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "LTXVideoSwiftRuntime", targets: ["LTXVideoSwiftRuntime"]),
        .executable(name: "GenImageLTXVideoPoC", targets: ["GenImageLTXVideoPoC"]),
        .executable(
            name: "GenImageLTXVideoWorker",
            targets: ["GenImageLTXVideoWorker"]
        ),
        .executable(
            name: "GenImageLTXVideoGGUFWorker",
            targets: ["GenImageLTXVideoGGUFWorker"]
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
            name: "LTXVideoSwiftRuntime",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ]
        ),
        .executableTarget(
            name: "GenImageLTXVideoPoC",
            dependencies: [
                "LTXVideoSwiftRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ]
        ),
        .executableTarget(
            name: "GenImageLTXVideoWorker",
            dependencies: [
                "LTXVideoSwiftRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "GenImageLTXVideoGGUFWorker",
            dependencies: [
                "LTXVideoSwiftRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        ),
        .testTarget(
            name: "LTXVideoSwiftRuntimeTests",
            dependencies: [
                "LTXVideoSwiftRuntime",
                .product(name: "MLX", package: "mlx-swift")
            ]
        )
    ]
)
