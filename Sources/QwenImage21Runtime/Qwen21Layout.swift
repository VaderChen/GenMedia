import Foundation
import MLX

/// Interleaved text / reference-image / text / target blocks. A VL image slot expands to four latents.
struct Qwen21Layout {
    struct Segment {
        let range: Range<Int>
        let image: Bool
    }
    let segments: [Segment]
    let textIndices: [Int32]
    let imageIndices: [Int32]
    let targetStart: Int
    let positions: [[Float]]
    var count: Int { positions.count }

    init(imageSlots: [Bool], referenceHeight: Int?, referenceWidth: Int?, height: Int, width: Int) throws {
        var segments: [Segment] = [], text: [Int32] = [], images: [Int32] = [], positions: [[Float]] = []
        var cursor = 0, position = 0
        func appendImage(_ h: Int, _ w: Int) {
            let start = positions.count
            for y in 0..<h { for x in 0..<w {
                images.append(Int32(positions.count))
                positions.append([Float(position), Float(y - (h - h / 2)), Float(x - (w - w / 2))])
            }}
            segments.append(Segment(range: start..<positions.count, image: true))
            position += max(h, w)
        }
        var referenceSeen = false
        while cursor < imageSlots.count {
            if imageSlots[cursor] {
                guard !referenceSeen, let h = referenceHeight, let w = referenceWidth else {
                    throw Qwen21Error.invalid("參考影像 token 佈局不符。")
                }
                let start = cursor
                while cursor < imageSlots.count && imageSlots[cursor] { cursor += 1 }
                guard (cursor - start) * 4 == h * w else {
                    throw Qwen21Error.invalid("視覺編碼器與 VAE 的參考影像尺寸不一致。")
                }
                appendImage(h, w); referenceSeen = true
            } else {
                let start = positions.count
                while cursor < imageSlots.count && !imageSlots[cursor] {
                    text.append(Int32(cursor))
                    positions.append([Float(position), Float(position), Float(position)])
                    position += 1; cursor += 1
                }
                segments.append(Segment(range: start..<positions.count, image: false))
            }
        }
        guard referenceSeen == (referenceHeight != nil) else {
            throw Qwen21Error.invalid("缺少參考影像 token。")
        }
        targetStart = positions.count
        appendImage(height, width)
        self.segments = segments; textIndices = text; imageIndices = images; self.positions = positions
    }

    func rotary(axes: [Int]) -> (MLXArray, MLXArray) {
        var angles: [Float] = []
        for position in positions {
            for (axis, dimension) in axes.enumerated() {
                for i in stride(from: 0, to: dimension, by: 2) {
                    angles.append(position[axis] / pow(10_000, Float(i) / Float(dimension)))
                }
            }
        }
        let a = MLXArray(angles).reshaped([1, 1, count, axes.reduce(0, +) / 2])
        return (cos(a), sin(a))
    }
}
