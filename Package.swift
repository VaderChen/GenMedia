// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "GenImage",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "GenImage", targets: ["GenImageApp"]),
        .executable(name: "GenImageDoctor", targets: ["GenImageDoctor"]),
        .executable(name: "GenImageMCP", targets: ["GenImageMCP"]),
        .executable(name: "ACEStepSwiftPoC", targets: ["ACEStepSwiftPoC"]),
        .library(name: "GenImageCore", targets: ["GenImageCore"]),
        .library(name: "GenImageRuntime", targets: ["GenImageRuntime"]),
        .library(name: "ACEStepSwiftRuntime", targets: ["ACEStepSwiftRuntime"]),
        .library(name: "GenImageMCPServer", targets: ["GenImageMCPServer"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/zhutao100/Z-Image.swift.git",
            revision: "28bfcf3148c041a554629247170eb54d9ac46830"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            revision: "7da33441c7c08b010ff1aa8da9dc3d82277272f5"
        ),
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git",
            exact: "0.30.6"
        ),
        .package(
            url: "https://github.com/apple/swift-log.git",
            from: "1.6.4"
        ),
        .package(
            url: "https://github.com/argmaxinc/argmax-oss-swift.git",
            revision: "ea872ffd35705aa757f33033500b9b0d40bd38df"
        ),
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            revision: "6428e29186573c6d33c598e25d460e6690bc0ee1"
        )
    ],
    targets: [
        .target(name: "GenImageCore"),
        .target(
            name: "ACEStepSwiftRuntime",
            dependencies: [
                .product(name: "ZImage", package: "z-image.swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ]
        ),
        .executableTarget(
            name: "GenImageApp",
            dependencies: ["GenImageCore", "GenImageRuntime", "GenImageMCPServer"],
            resources: [
                .copy("Resources/WebUI")
            ]
        ),
        .executableTarget(
            name: "GenImageDoctor",
            dependencies: ["GenImageCore", "GenImageRuntime"]
        ),
        .target(
            name: "GenImageRuntime",
            dependencies: [
                "GenImageCore",
                "ACEStepSwiftRuntime",
                .product(name: "ZImage", package: "z-image.swift"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "WhisperKit", package: "argmax-oss-swift"),
                .product(name: "FluidAudio", package: "FluidAudio"),
                .product(name: "Logging", package: "swift-log")
            ]
        ),
        .target(
            name: "GenImageMCPServer",
            dependencies: ["GenImageCore", "GenImageRuntime"]
        ),
        .executableTarget(
            name: "GenImageMCP",
            dependencies: ["GenImageMCPServer"]
        ),
        .executableTarget(
            name: "ACEStepSwiftPoC",
            dependencies: [
                "ACEStepSwiftRuntime",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ]
        ),
        .executableTarget(
            name: "GenImageASRPoC",
            dependencies: [
                "GenImageCore",
                .product(name: "WhisperKit", package: "argmax-oss-swift")
            ]
        ),
        .testTarget(
            name: "GenImageCoreTests",
            dependencies: ["GenImageCore"]
        ),
        .testTarget(
            name: "GenImageRuntimeTests",
            dependencies: ["GenImageCore", "GenImageRuntime"]
        ),
        .testTarget(
            name: "GenImageMCPServerTests",
            dependencies: ["GenImageMCPServer"]
        )
    ]
)
