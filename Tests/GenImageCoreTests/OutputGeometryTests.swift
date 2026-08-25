import Testing

@testable import GenImageCore

struct OutputGeometryTests {
    @Test func quantizeSnapsToTheNearestAlignedValue() {
        #expect(OutputGeometry.quantize(1024) == 1024)
        #expect(OutputGeometry.quantize(1020) == 1024)
        #expect(OutputGeometry.quantize(1017) == 1024)
        #expect(OutputGeometry.quantize(1015) == 1008)
        #expect(OutputGeometry.quantize(100) == 96)
        #expect(OutputGeometry.quantize(104) == 112)
    }

    @Test func quantizeClampsToTheSupportedRange() {
        #expect(OutputGeometry.quantize(0) == OutputGeometry.minimumDimension)
        #expect(OutputGeometry.quantize(-512) == OutputGeometry.minimumDimension)
        #expect(OutputGeometry.quantize(16) == OutputGeometry.minimumDimension)
        #expect(OutputGeometry.quantize(999_999) == OutputGeometry.maximumDimension)
        #expect(OutputGeometry.isSupported(OutputGeometry.minimumDimension))
        #expect(OutputGeometry.isSupported(OutputGeometry.maximumDimension))
    }

    @Test func quantizedValuesAreAlwaysAccepted() {
        for value in stride(from: -64, through: 5000, by: 7) {
            #expect(OutputGeometry.isSupported(OutputGeometry.quantize(value)))
        }
    }

    @Test func supportRequiresAlignmentAndRange() {
        #expect(!OutputGeometry.isSupported(1000))
        #expect(!OutputGeometry.isSupported(48))
        #expect(!OutputGeometry.isSupported(4112))
        #expect(OutputGeometry.isSupported(width: 512, height: 768))
        #expect(!OutputGeometry.isSupported(width: 512, height: 770))
    }

    // The Runtime floors rather than rounds, so a canvas planned here has to survive that
    // floor unchanged — otherwise the size generated is not the size that was planned.
    @Test func runtimeAlignedFloorsAndSurvivesReapplication() {
        #expect(OutputGeometry.runtimeAligned(1080) == 1072)
        #expect(OutputGeometry.runtimeAligned(1088) == 1088)
        #expect(OutputGeometry.runtimeAligned(8) == 16)
        for value in stride(from: 1, through: 4096, by: 13) {
            let aligned = OutputGeometry.runtimeAligned(value)
            #expect(OutputGeometry.runtimeAligned(aligned) == aligned)
        }
    }

    // Parity with diffusers calculate_dimensions / QwenVLPromptEncoder.calculateDimensions.
    @Test func canvasMatchesTheDiffusersReference() {
        let square = OutputGeometry.canvas(area: 1024 * 1024, ratio: 1)
        #expect(square == (width: 1024, height: 1024))

        let portrait = OutputGeometry.canvas(area: 1024 * 1024, ratio: 2.0 / 3.0)
        #expect(portrait == (width: 832, height: 1248))

        let landscape = OutputGeometry.canvas(area: 1024 * 1024, ratio: 3.0 / 2.0)
        #expect(landscape == (width: 1248, height: 832))
    }

    @Test func canvasSidesAreAlwaysRuntimeAligned() {
        for ratio in [0.5, 9.0 / 16, 2.0 / 3, 1.0, 4.0 / 3, 16.0 / 9, 2.0] {
            let canvas = OutputGeometry.canvas(area: 1024 * 1024, ratio: ratio)
            #expect(OutputGeometry.runtimeAligned(canvas.width) == canvas.width)
            #expect(OutputGeometry.runtimeAligned(canvas.height) == canvas.height)
        }
    }

    @Test func canvasRejectsDegenerateRatios() {
        #expect(OutputGeometry.canvas(area: 0, ratio: 1).width > 0)
        #expect(OutputGeometry.canvas(area: 1024 * 1024, ratio: 0).width > 0)
        #expect(OutputGeometry.canvas(area: 1024 * 1024, ratio: .nan).width > 0)
        #expect(OutputGeometry.canvas(area: 1024 * 1024, ratio: .infinity).width > 0)
    }

    @Test func smallImageEditRequestsGenerateAtTheNativeArea() {
        let tiny = OutputGeometry.imageEditPlan(width: 128, height: 192)
        #expect(tiny.generationWidth == 832)
        #expect(tiny.generationHeight == 1248)
        #expect(tiny.outputWidth == 128)
        #expect(tiny.outputHeight == 192)
        #expect(tiny.needsResample)

        let small = OutputGeometry.imageEditPlan(width: 512, height: 768)
        #expect(small.generationWidth == 832)
        #expect(small.generationHeight == 1248)
        #expect(small.needsResample)
    }

    @Test func nativeAndLargerImageEditRequestsGenerateAsAsked() {
        let native = OutputGeometry.imageEditPlan(width: 1024, height: 1024)
        #expect(native.generationWidth == 1024)
        #expect(native.generationHeight == 1024)
        #expect(!native.needsResample)

        let large = OutputGeometry.imageEditPlan(width: 1088, height: 1632)
        #expect(large.generationWidth == 1088)
        #expect(large.generationHeight == 1632)
        #expect(!large.needsResample)
    }

    // The conditioning grid and the target grid must be the same token grid — an unequal one
    // is what cropped the source to its centre.
    @Test func everyPlanKeepsTheAspectAndAtLeastTheNativeTokenCount() {
        let sizes = [
            (64, 64), (128, 192), (256, 384), (512, 768), (768, 512), (1024, 1024),
            (1088, 1632), (576, 1024), (1920, 1088), (2048, 2048),
        ]
        for (width, height) in sizes {
            let plan = OutputGeometry.imageEditPlan(width: width, height: height)
            #expect(plan.generationWidth == OutputGeometry.runtimeAligned(plan.generationWidth))
            #expect(plan.generationHeight == OutputGeometry.runtimeAligned(plan.generationHeight))
            // Rounding each side to /32 costs a little area; the canvas still has to stay
            // within a few percent of the trained one.
            #expect(
                plan.generationWidth * plan.generationHeight
                    >= OutputGeometry.nativeCanvasArea / 100 * 95
            )

            let requested = Double(width) / Double(height)
            let generated = Double(plan.generationWidth) / Double(plan.generationHeight)
            #expect(abs(generated - requested) / requested < 0.02)
        }
    }

    @Test func imageEditPlanSurvivesDegenerateInput() {
        let plan = OutputGeometry.imageEditPlan(width: 0, height: 0)
        #expect(plan.generationWidth > 0)
        #expect(plan.generationHeight > 0)
        #expect(plan.outputWidth > 0)
        #expect(plan.outputHeight > 0)
    }
}
