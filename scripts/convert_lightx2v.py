"""Adapt the official LightX2V 1022 LoRAs to native ComfyUI; keep originals."""
import argparse
import json
import struct
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path, help="ComfyUI/models")
    parser.add_argument("--lora-dir", type=Path, help="Original LoRA directory; default models/loras")
    parser.add_argument("--output-dir", type=Path, help="Default models/loras")
    args = parser.parse_args()
    source_dir = args.lora_dir or args.models_dir / "loras"
    output_dir = args.output_dir or args.models_dir / "loras"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for level in ("high", "low"):
        model = args.models_dir / "diffusion_models" / f"wan2.2_i2v_{level}_noise_14B_fp8_scaled.safetensors"
        with model.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size < 100_000_000:
                raise ValueError(f"Invalid safetensors header: {model.name}")
            header = json.loads(stream.read(size))
        valid = {"diffusion_model." + key for key in header if key != "__metadata__"}
        filename = f"wan2.2_i2v_A14b_{level}_noise_lora_rank64_lightx2v_4step_1022.safetensors"
        original = load_file(source_dir / filename)
        converted, renamed, omitted, unknown = {}, [], [], []
        for key, tensor in original.items():
            new_key = key.replace(".diff_m", ".modulation.diff")
            target = None
            for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha", ".dora_scale"):
                if new_key.endswith(suffix):
                    target = new_key[:-len(suffix)] + ".weight"
                    break
            if new_key.endswith(".diff_b"):
                target = new_key[:-7] + ".bias"
            elif new_key.endswith(".diff"):
                stem = new_key[:-5]
                target = stem if stem in valid else stem + ".weight"
            if target not in valid:
                # The published low-noise adapter contains Wan 2.1 CLIP-image modules;
                # those modules are absent from this Wan 2.2 base architecture.
                if any(part in key for part in (".k_img.", ".v_img.", ".norm_k_img.", ".img_emb.")):
                    omitted.append(key)
                    continue
                unknown.append({"key": key, "target": target})
                continue
            if key != new_key:
                renamed.append({"from": key, "to": new_key})
            converted[new_key] = tensor
        if unknown:
            raise ValueError(f"Unrecognized adapter entries: {unknown}")
        destination = output_dir / f"lightx2v_1022_{level}_native.safetensors"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        save_file(converted, str(destination), metadata={
            "source": "lightx2v/Wan2.2-Distill-Loras",
            "conversion": "diff_m -> modulation.diff; omit image-conditioning modules absent in Wan2.2 base; retained tensor values unchanged",
        })
        report.append({"source": filename, "output": destination.name,
                       "retained": len(converted), "renamed": renamed,
                       "omitted_absent_image_modules": omitted})
        print(f"{level}: retained={len(converted)}, renamed={len(renamed)}, omitted={len(omitted)}")
    (output_dir / "lightx2v-conversion.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
