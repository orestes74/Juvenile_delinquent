"""Check model presence, recorded sizes and safetensors layout without loading tensors."""
import argparse
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models_dir", type=Path, help="ComfyUI/models directory")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "models.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    failures = 0
    for item in manifest["files"]:
        path = args.models_dir / item["path"]
        try:
            size = path.stat().st_size
            if size != item["expected_bytes"]:
                raise ValueError(f"size {size}, expected {item['expected_bytes']}")
            with path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                if not 0 < header_size <= min(size - 8, 100_000_000):
                    raise ValueError("invalid header length")
                header = json.loads(stream.read(header_size))
            payload_size = size - 8 - header_size
            tensors = [value for key, value in header.items() if key != "__metadata__"]
            for tensor in tensors:
                start, end = tensor["data_offsets"]
                if not 0 <= start <= end <= payload_size:
                    raise ValueError("tensor offset outside file")
            if not tensors or max(t["data_offsets"][1] for t in tensors) != payload_size:
                raise ValueError("unexpected payload length")
            print(f"OK  {item['path']}")
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            failures += 1
            print(f"FAIL {item['path']}: {exc}")
    print("Header/size checks only; tensor contents are not checksummed.")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
