"""Minimal GGUF reader for the MiniMax H3 parity harness.

Decodes tensors independently of the Swift loader, so a comparison exercises
both sides' weight handling rather than only the model maths. Handles the
subset H3 uses: F32, F16 and Q4_0, plus ComfyUI's `comfy.gguf.orig_shape.*`
metadata (the converter reshapes some tensors before quantizing them and
records the true shape separately).
"""
import struct
from collections import Counter

import numpy as np

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 20: "IQ4_NL", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 30: "BF16",
}

ORIG_SHAPE_PREFIX = "comfy.gguf.orig_shape."

_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
    7: "<B", 10: "<Q", 11: "<q", 12: "<d",
}


class GGUFFile:
    """Header-only parse plus lazy, memory-mapped tensor decode."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as handle:
            assert handle.read(4) == b"GGUF", "not a GGUF file"
            self.version = struct.unpack("<I", handle.read(4))[0]
            tensor_count = struct.unpack("<Q", handle.read(8))[0]
            kv_count = struct.unpack("<Q", handle.read(8))[0]

            def read_string():
                length = struct.unpack("<Q", handle.read(8))[0]
                return handle.read(length).decode("utf-8", "replace")

            def read_value(kind):
                if kind == 8:
                    return read_string()
                if kind == 9:
                    element = struct.unpack("<I", handle.read(4))[0]
                    count = struct.unpack("<Q", handle.read(8))[0]
                    return [read_value(element) for _ in range(count)]
                raw = handle.read(_SCALAR_SIZES[kind])
                value = struct.unpack(_SCALAR_FORMATS[kind], raw)[0]
                return bool(value) if kind == 7 else value

            self.metadata = {}
            for _ in range(kv_count):
                key = read_string()
                self.metadata[key] = read_value(struct.unpack("<I", handle.read(4))[0])

            self.tensors = {}
            for _ in range(tensor_count):
                name = read_string()
                ndim = struct.unpack("<I", handle.read(4))[0]
                dims = [struct.unpack("<Q", handle.read(8))[0] for _ in range(ndim)]
                type_code = struct.unpack("<I", handle.read(4))[0]
                offset = struct.unpack("<Q", handle.read(8))[0]
                self.tensors[name] = {
                    "name": name,
                    "dims": dims,
                    "type": GGML_TYPES.get(type_code, str(type_code)),
                    "offset": offset,
                }
            header_end = handle.tell()

        alignment = self.metadata.get("general.alignment", 32)
        self.data_offset = (header_end + alignment - 1) // alignment * alignment
        self.orig_shapes = {
            key[len(ORIG_SHAPE_PREFIX):]: list(value)
            for key, value in self.metadata.items()
            if key.startswith(ORIG_SHAPE_PREFIX)
        }
        self._map = np.memmap(path, dtype=np.uint8, mode="r")

    @property
    def architecture(self):
        return self.metadata.get("general.architecture")

    def type_counts(self):
        return Counter(t["type"] for t in self.tensors.values())

    def shape(self, name):
        """Logical shape: the ComfyUI override when present, else reversed dims."""
        tensor = self.tensors[name]
        return self.orig_shapes.get(name, list(reversed(tensor["dims"])))

    def decode(self, name):
        """Decode one tensor to float32 with its logical shape."""
        tensor = self.tensors[name]
        shape = self.shape(name)
        count = int(np.prod(shape))
        start = self.data_offset + tensor["offset"]
        kind = tensor["type"]

        if kind == "F32":
            raw = self._map[start:start + count * 4].tobytes()
            values = np.frombuffer(raw, dtype=np.float32)
        elif kind == "F16":
            raw = self._map[start:start + count * 2].tobytes()
            values = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif kind == "Q4_0":
            assert count % 32 == 0, f"{name}: {count} is not a multiple of 32"
            blocks = count // 32
            raw = np.frombuffer(
                self._map[start:start + blocks * 18].tobytes(), dtype=np.uint8
            ).reshape(blocks, 18)
            # 2-byte f16 scale, then 16 bytes holding 32 nibbles.
            scale = raw[:, :2].copy().view(np.float16).astype(np.float32)
            quants = raw[:, 2:]
            # Low nibbles are elements 0..15, high nibbles 16..31; both biased by 8.
            low = (quants & 0x0F).astype(np.float32) - 8.0
            high = (quants >> 4).astype(np.float32) - 8.0
            values = (np.concatenate([low, high], axis=1) * scale).reshape(-1)
        else:
            raise ValueError(f"{name}: unsupported GGUF type {kind}")

        return values.reshape(shape).copy()


if __name__ == "__main__":
    import sys

    gguf = GGUFFile(sys.argv[1])
    print(f"version {gguf.version}  arch {gguf.architecture}")
    print(f"tensors {len(gguf.tensors)}  kv {len(gguf.metadata)}")
    print("types:", dict(gguf.type_counts()))
    print(f"orig_shape overrides: {len(gguf.orig_shapes)}")
