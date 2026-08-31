"""Compare a Swift safetensors dump against a reference .npy.

Reports relative max diff — the metric this port uses throughout, since cosine
saturates — plus SNR, which is the meaningful one for audio.

Usage:
    python3 compare.py <reference.npy> <swift.safetensors> <key> [--audio]
"""
import argparse
import json
import struct

import numpy as np

# Thresholds established across the Music3 and LTX ports.
FP32_THRESHOLD = 1e-4
BF16_THRESHOLD = 1e-2


def read_safetensors(path, key):
    with open(path, "rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
        base = 8 + header_length
        entry = header[key]
        start, stop = entry["data_offsets"]
        handle.seek(base + start)
        raw = handle.read(stop - start)
    dtype = {"F32": np.float32, "F16": np.float16}[entry["dtype"]]
    return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"]).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("key")
    parser.add_argument("--audio", action="store_true", help="also report SNR")
    parser.add_argument("--threshold", type=float, default=FP32_THRESHOLD)
    args = parser.parse_args()

    reference = np.load(args.reference).astype(np.float64)
    candidate = read_safetensors(args.candidate, args.key)
    if reference.shape != candidate.shape:
        raise SystemExit(f"shape mismatch: {reference.shape} vs {candidate.shape}")

    difference = np.abs(reference - candidate)
    denominator = max(np.abs(reference).max(), 1e-12)
    relative = difference.max() / denominator

    print(f"  shape             : {reference.shape}")
    print(f"  abs max diff      : {difference.max():.6e}")
    print(f"  relative max diff : {relative:.6e}")
    print(f"  mean abs diff     : {difference.mean():.6e}")
    if args.audio:
        noise = np.sum((reference - candidate) ** 2)
        snr = 10 * np.log10(np.sum(reference ** 2) / max(noise, 1e-30))
        print(f"  SNR               : {snr:.2f} dB")
    verdict = "PASS" if relative <= args.threshold else "FAIL"
    print(f"  threshold {args.threshold:.0e}    : {verdict}")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
