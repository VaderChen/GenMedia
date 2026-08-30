// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "GenImageZImageWorker",
    platforms: [.macOS(.v14)],
    products: [
        .executable(
            name: "GenImageZImageWorker",
            targets: ["GenImageZImageWorker"]
        )
    ],
    dependencies: [
        .package(
            url: "https://github.com/zhutao100/Z-Image.swift.git",
            revision: "28bfcf3148c041a554629247170eb54d9ac46830"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.30.6"
        ),
        .package(
            url: "https://github.com/apple/swift-log.git",
            from: "1.6.4"
        )
    ],
    targets: [
        .executableTarget(
            name: "GenImageZImageWorker",
            dependencies: [
                .product(name: "ZImage", package: "z-image.swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Logging", package: "swift-log")
            ]
        )
    ]
)
