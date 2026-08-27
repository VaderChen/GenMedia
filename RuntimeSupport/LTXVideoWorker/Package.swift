// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "GenImageLTXVideoWorker",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "LTXVideoSwiftRuntime", targets: ["LTXVideoSwiftRuntime"]),
        .executable(name: "GenImageLTXVideoPoC", targets: ["GenImageLTXVideoPoC"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.31.6"
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
                .product(name: "MLX", package: "mlx-swift")
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
