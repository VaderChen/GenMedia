#!/usr/bin/env python3
"""Encode browser captures with one shared palette and readable hold durations."""
import json
import sys
from pathlib import Path

from PIL import Image


def encode(captures: Path, output: Path) -> None:
    entries = json.loads((captures / "frames.json").read_text())
    frames = []
    for entry in entries:
        with Image.open(captures / entry["name"]) as source:
            size = (1280, round(source.height * 1280 / source.width))
            frames.append(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))

    # Sample every view for a stable palette; per-frame palettes cause flicker.
    samples = frames[::max(1, len(frames) // 12)] + [frames[-1]]
    sheet = Image.new("RGB", (320, 215 * len(samples)))
    for index, frame in enumerate(samples):
        sheet.paste(frame.resize((320, 215)), (0, 215 * index))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for frame in frames]
    # GIF timing uses centiseconds. Keep original UI/cursor motion timing.
    durations = [max(20, round(entry["duration"] * 100) * 10) for entry in entries]
    output.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(output, save_all=True, append_images=indexed[1:], duration=durations,
                    loop=0, optimize=True, disposal=1)
    with Image.open(output) as result:
        assert result.is_animated and result.info["loop"] == 0
        for index in range(result.n_frames):
            result.seek(index)
            result.load()


if __name__ == "__main__":
    encode(Path(sys.argv[1]), Path(sys.argv[2]))
