from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Support `python -I` (isolated mode), which does not put the script directory
# on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from common import build_checksums, ensure_dir, sha256_file, write_json


NORMALIZER_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize LuxRT model weights into canonical safetensors.")
    parser.add_argument(
        "--family",
        choices=["pi3", "pi3x", "rmbg", "triposr", "triposg", "vggt"],
        required=True,
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--raw-model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_builtin(inner) for inner in value]
    return value


def ordered_tensor_dict(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensors[key].detach().cpu().contiguous() for key in sorted(tensors.keys())}


def histogram(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
    counts = Counter(str(tensor.dtype).replace("torch.", "") for tensor in tensors.values())
    return dict(sorted(counts.items()))


def provenance_keys(raw_files: list[Path]) -> list[str]:
    """Provenance keys for raw weight files.

    Files whose basename is unique across the raw list keep the historical bare
    ``<basename>`` key. Basenames that occur more than once are keyed as
    ``<parent-dir-name>/<basename>`` so sharded raw layouts (e.g. the TripoSG
    ``transformer/`` and ``vae/`` shards that share a filename) stay
    distinguishable. Every pre-existing family has unique basenames, so this is
    a strict generalization: existing manifests keep byte-identical keys.
    Must stay in lockstep with ``weights::provenance_keys`` in
    ``crates/lux3d-core/src/weights/mod.rs``.
    """
    counts = Counter(path.name for path in raw_files)
    keys: list[str] = []
    for path in raw_files:
        if counts[path.name] > 1:
            parent = path.parent.name
            keys.append(f"{parent}/{path.name}" if parent else path.name)
        else:
            keys.append(path.name)
    return keys


def write_outputs(
    *,
    family: str,
    output_dir: Path,
    raw_files: list[Path],
    resolved_config: dict,
    tensors: dict[str, torch.Tensor],
) -> None:
    target_root = ensure_dir(output_dir)
    canonical_file = target_root / "model.safetensors"
    resolved_config_file = target_root / "resolved_config.json"
    manifest_file = target_root / "manifest.json"
    checksums_file = target_root / "checksums.json"
    keys = provenance_keys(raw_files)
    source_checksums = {key: sha256_file(path) for key, path in zip(keys, raw_files)}

    if manifest_file.is_file() and canonical_file.is_file() and resolved_config_file.is_file():
        try:
            existing_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            existing_resolved_config = json.loads(
                resolved_config_file.read_text(encoding="utf-8")
            )
            if (
                existing_manifest.get("family") == family
                and existing_manifest.get("normalizer_version") == NORMALIZER_VERSION
                and existing_manifest.get("raw_files") == keys
                and existing_manifest.get("source_checksums") == source_checksums
                and existing_resolved_config == resolved_config
            ):
                return
        except Exception:
            pass

    ordered = ordered_tensor_dict(tensors)
    save_file(ordered, str(canonical_file))
    write_json(resolved_config_file, resolved_config)

    manifest = {
        "family": family,
        "normalizer_version": NORMALIZER_VERSION,
        "raw_files": keys,
        "canonical_file": canonical_file.name,
        "resolved_config_file": resolved_config_file.name,
        "tensor_count": len(ordered),
        "dtype_histogram": histogram(ordered),
        "source_checksums": source_checksums,
    }
    write_json(manifest_file, manifest)
    checksums = build_checksums(
        repo_root=target_root,
        paths=[canonical_file, resolved_config_file, manifest_file],
    )
    write_json(checksums_file, checksums)


def normalize_pi3(raw_model_dir: Path, output_dir: Path) -> None:
    model_root = raw_model_dir
    raw_weight = model_root / "model.safetensors"
    raw_config = model_root / "config.json"

    tensors = load_file(str(raw_weight))
    with safe_open(str(raw_weight), framework="pt", device="cpu") as handle:
        if len(list(handle.keys())) != len(tensors):
            raise RuntimeError("failed to read all Pi3 safetensors keys")

    resolved_config = json.loads(raw_config.read_text(encoding="utf-8"))
    write_outputs(
        family="pi3",
        output_dir=output_dir,
        raw_files=[raw_weight],
        resolved_config=resolved_config,
        tensors=tensors,
    )


def normalize_pi3x(raw_model_dir: Path, output_dir: Path) -> None:
    model_root = raw_model_dir
    raw_weight = model_root / "model.safetensors"
    raw_config = model_root / "config.json"

    tensors = load_file(str(raw_weight))
    with safe_open(str(raw_weight), framework="pt", device="cpu") as handle:
        if len(list(handle.keys())) != len(tensors):
            raise RuntimeError("failed to read all Pi3X safetensors keys")

    resolved_config = json.loads(raw_config.read_text(encoding="utf-8"))
    write_outputs(
        family="pi3x",
        output_dir=output_dir,
        raw_files=[raw_weight, raw_config],
        resolved_config=resolved_config,
        tensors=tensors,
    )


def normalize_triposr(raw_model_dir: Path, output_dir: Path) -> None:
    from omegaconf import OmegaConf

    model_root = raw_model_dir
    raw_weight = model_root / "model.ckpt"
    raw_config = model_root / "config.yaml"

    cfg = OmegaConf.load(raw_config)
    OmegaConf.resolve(cfg)
    resolved_config = to_builtin(OmegaConf.to_container(cfg, resolve=True))

    checkpoint = torch.load(raw_weight, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"unexpected TripoSR checkpoint payload type: {type(state_dict)!r}")

    tensors = {
        str(key): value
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }
    if not tensors:
        raise RuntimeError("TripoSR checkpoint did not contain any tensors")

    write_outputs(
        family="triposr",
        output_dir=output_dir,
        raw_files=[raw_weight, raw_config],
        resolved_config=resolved_config,
        tensors=tensors,
    )

def normalize_vggt(raw_model_dir: Path, output_dir: Path) -> None:
    model_root = raw_model_dir
    raw_weight = model_root / "model.safetensors"
    raw_config = model_root / "config.json"

    # The facebook/VGGT-1B checkpoint already uses the module hierarchy the
    # runtime expects; keep the keys exactly as shipped.
    tensors = load_file(str(raw_weight))
    resolved_config = json.loads(raw_config.read_text(encoding="utf-8"))
    write_outputs(
        family="vggt",
        output_dir=output_dir,
        raw_files=[raw_weight, raw_config],
        resolved_config=resolved_config,
        tensors=tensors,
    )


# RMBG-1.4 is an auxiliary package: it is NOT a `ModelFamily` and must live in
# its own canonical directory because its BRIA non-commercial license differs
# from TripoSG's MIT license. The key names are already the module hierarchy
# used by `models/RMBG-1.4/briarmbg.py`; keep them exactly as shipped.
RMBG_RAW_FILES = [
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
]


def normalize_rmbg(raw_model_dir: Path, output_dir: Path) -> None:
    model_root = raw_model_dir
    raw_weight = model_root / "model.safetensors"
    raw_config = model_root / "config.json"
    raw_preprocessor = model_root / "preprocessor_config.json"

    tensors = load_file(str(raw_weight))
    with safe_open(str(raw_weight), framework="pt", device="cpu") as handle:
        if len(list(handle.keys())) != len(tensors):
            raise RuntimeError("failed to read all RMBG safetensors keys")
    if not tensors:
        raise RuntimeError("RMBG checkpoint did not contain any tensors")

    model_config = json.loads(raw_config.read_text(encoding="utf-8"))
    preprocessor = json.loads(raw_preprocessor.read_text(encoding="utf-8"))
    resolved_config = {
        "family": "rmbg",
        "architecture": "BriaRMBG",
        "model": model_config,
        "preprocessor": preprocessor,
        # TripoSG's `scripts/image_process.py` resizes RGB to 1024x1024 with
        # torchvision bilinear antialias, then applies the local `rmbg()`
        # wrapper: `TF.normalize(image, [0.5,0.5,0.5], [1.0,1.0,1.0])`.
        # Record that observed reference behavior explicitly so the Rust
        # runtime cannot silently substitute the model-card preprocessing.
        "reference_inference_input": {
            "resize": [1024, 1024],
            "resize_resample": "bilinear_antialias",
            "value_range": [0.0, 1.0],
            "mean": [0.5, 0.5, 0.5],
            "std": [1.0, 1.0, 1.0],
        },
        "in_ch": int(model_config.get("in_ch", 3)),
        "out_ch": int(model_config.get("out_ch", 1)),
    }

    write_outputs(
        family="rmbg",
        output_dir=output_dir,
        raw_files=[raw_weight, raw_config, raw_preprocessor],
        resolved_config=resolved_config,
        tensors=tensors,
    )


# Exact raw file order consumed by the TripoSG normalizer and mirrored by
# `WeightLocator::locate_triposg` and `raw_model_required_files(TripoSg)`.
TRIPOSG_RAW_FILES = [
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "image_encoder_dinov2/model.safetensors",
    "model_index.json",
    "feature_extractor_dinov2/preprocessor_config.json",
    "image_encoder_dinov2/config.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "vae/config.json",
]


def normalize_triposg(raw_model_dir: Path, output_dir: Path) -> None:
    model_root = raw_model_dir
    raw_files = [model_root / relative for relative in TRIPOSG_RAW_FILES]

    # Merge the three diffusers shards under a single canonical namespace;
    # no other key remapping happens.
    prefixes = {
        "transformer/diffusion_pytorch_model.safetensors": "transformer",
        "vae/diffusion_pytorch_model.safetensors": "vae",
        "image_encoder_dinov2/model.safetensors": "image_encoder",
    }
    tensors: dict[str, torch.Tensor] = {}
    for relative, prefix in prefixes.items():
        shard = load_file(str(model_root / relative))
        if not shard:
            raise RuntimeError(f"TripoSG shard {relative} did not contain any tensors")
        for key, value in shard.items():
            tensors[f"{prefix}.{key}"] = value

    def read_json(relative: str) -> dict:
        return json.loads((model_root / relative).read_text(encoding="utf-8"))

    resolved_config = {
        "family": "triposg",
        "model_index": read_json("model_index.json"),
        "preprocessor": read_json("feature_extractor_dinov2/preprocessor_config.json"),
        "image_encoder": read_json("image_encoder_dinov2/config.json"),
        "transformer": read_json("transformer/config.json"),
        "vae": read_json("vae/config.json"),
        "scheduler_config": read_json("scheduler/scheduler_config.json"),
        "pipeline_defaults": {
            "num_tokens": 2048,
            "num_inference_steps": 50,
            "guidance_scale": 7.0,
            "bounds": [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005],
            "dense_octree_depth": 8,
            "hierarchical_octree_depth": 9,
            "sdf_level": 0.0,
        },
    }

    write_outputs(
        family="triposg",
        output_dir=output_dir,
        raw_files=raw_files,
        resolved_config=resolved_config,
        tensors=tensors,
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return Path(args.raw_model_dir).resolve(), Path(args.output_dir).resolve()


def self_test_provenance_keys() -> None:
    """Pure-function checks for the duplicate-basename provenance rule."""
    raw = Path("/raw")
    # Real TripoSG layout: two shards share a basename and three configs share
    # another; every other basename stays bare.
    files = [raw / relative for relative in TRIPOSG_RAW_FILES]
    assert provenance_keys(files) == [
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "model.safetensors",
        "model_index.json",
        "preprocessor_config.json",
        "image_encoder_dinov2/config.json",
        "scheduler_config.json",
        "transformer/config.json",
        "vae/config.json",
    ], provenance_keys(files)

    # Synthetic duplicate shard basenames.
    dup = [raw / "a/model.safetensors", raw / "b/model.safetensors", raw / "config.json"]
    assert provenance_keys(dup) == [
        "a/model.safetensors",
        "b/model.safetensors",
        "config.json",
    ], provenance_keys(dup)

    # Unique names stay bare (historical form).
    unique = [raw / "model.safetensors", raw / "config.json"]
    assert provenance_keys(unique) == ["model.safetensors", "config.json"]
    print("provenance key self-test passed")


def main() -> None:
    if "--self-test-provenance-keys" in sys.argv:
        self_test_provenance_keys()
        return

    args = parse_args()
    raw_model_dir, output_dir = resolve_paths(args)

    if args.family == "pi3":
        normalize_pi3(raw_model_dir, output_dir)
    elif args.family == "pi3x":
        normalize_pi3x(raw_model_dir, output_dir)
    elif args.family == "rmbg":
        normalize_rmbg(raw_model_dir, output_dir)
    elif args.family == "triposr":
        normalize_triposr(raw_model_dir, output_dir)
    elif args.family == "triposg":
        normalize_triposg(raw_model_dir, output_dir)
    elif args.family == "vggt":
        normalize_vggt(raw_model_dir, output_dir)
    else:
        raise RuntimeError(f"unsupported family: {args.family}")


if __name__ == "__main__":
    main()
