// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "GenImageMiniMaxH3Worker",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MiniMaxH3SwiftRuntime", targets: ["MiniMaxH3SwiftRuntime"]),
        .executable(
            name: "GenImageMiniMaxH3Worker",
            targets: ["GenImageMiniMaxH3Worker"]
        )
    ],
    dependencies: [
        .package(path: "../.."),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.31.6"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            exact: "1.3.3"
        )
    ],
    targets: [
        .target(
            name: "MiniMaxH3SwiftRuntime",
            dependencies: [
                .product(name: "GenImageGGUF", package: "GenMedia"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        ),
        .executableTarget(
            name: "GenImageMiniMaxH3Worker",
            dependencies: [
                "MiniMaxH3SwiftRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        ),
        .testTarget(
            name: "MiniMaxH3SwiftRuntimeTests",
            dependencies: [
                "MiniMaxH3SwiftRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]
        )
    ]
)
