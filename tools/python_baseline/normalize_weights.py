from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from common import build_checksums, ensure_dir, relative_to, resolve_repo_root, sha256_file, write_json


NORMALIZER_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize LuxRT model weights into canonical safetensors.")
    parser.add_argument("--family", choices=["pi3", "pi3x", "triposr"], required=True)
    parser.add_argument("--repo-root", type=str, default=None)
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


def canonical_root(repo_root: Path, family: str) -> Path:
    return repo_root / "3d" / "canonical-weights" / family


def write_outputs(
    *,
    repo_root: Path,
    family: str,
    raw_files: list[Path],
    resolved_config: dict,
    tensors: dict[str, torch.Tensor],
) -> None:
    target_root = ensure_dir(canonical_root(repo_root, family))
    canonical_file = target_root / "model.safetensors"
    resolved_config_file = target_root / "resolved_config.json"
    manifest_file = target_root / "manifest.json"
    checksums_file = target_root / "checksums.json"
    source_checksums = {
        relative_to(path, repo_root): sha256_file(path) for path in raw_files
    }

    if manifest_file.is_file() and canonical_file.is_file() and resolved_config_file.is_file():
        try:
            existing_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if (
                existing_manifest.get("family") == family
                and existing_manifest.get("normalizer_version") == NORMALIZER_VERSION
                and existing_manifest.get("raw_files") == [relative_to(path, repo_root) for path in raw_files]
                and existing_manifest.get("source_checksums") == source_checksums
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
        "raw_files": [relative_to(path, repo_root) for path in raw_files],
        "canonical_file": relative_to(canonical_file, repo_root),
        "resolved_config_file": relative_to(resolved_config_file, repo_root),
        "tensor_count": len(ordered),
        "dtype_histogram": histogram(ordered),
        "source_checksums": source_checksums,
    }
    write_json(manifest_file, manifest)
    checksums = build_checksums(
        repo_root=repo_root,
        paths=[canonical_file, resolved_config_file, manifest_file],
    )
    write_json(checksums_file, checksums)


def normalize_pi3(repo_root: Path) -> None:
    model_root = repo_root / "3d" / "models" / "yyfz233-Pi3"
    raw_weight = model_root / "model.safetensors"
    raw_config = model_root / "config.json"

    tensors = load_file(str(raw_weight))
    with safe_open(str(raw_weight), framework="pt", device="cpu") as handle:
        if len(list(handle.keys())) != len(tensors):
            raise RuntimeError("failed to read all Pi3 safetensors keys")

    resolved_config = json.loads(raw_config.read_text(encoding="utf-8"))
    write_outputs(
        repo_root=repo_root,
        family="pi3",
        raw_files=[raw_weight],
        resolved_config=resolved_config,
        tensors=tensors,
    )


def normalize_pi3x(repo_root: Path) -> None:
    model_root = repo_root / "3d" / "models" / "yyfz233-Pi3X"
    raw_weight = model_root / "model.safetensors"
    raw_config = model_root / "config.json"

    tensors = load_file(str(raw_weight))
    with safe_open(str(raw_weight), framework="pt", device="cpu") as handle:
        if len(list(handle.keys())) != len(tensors):
            raise RuntimeError("failed to read all Pi3X safetensors keys")

    resolved_config = json.loads(raw_config.read_text(encoding="utf-8"))
    write_outputs(
        repo_root=repo_root,
        family="pi3x",
        raw_files=[raw_weight, raw_config],
        resolved_config=resolved_config,
        tensors=tensors,
    )


def normalize_triposr(repo_root: Path) -> None:
    from omegaconf import OmegaConf

    model_root = repo_root / "3d" / "models" / "stabilityai-TripoSR"
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
        repo_root=repo_root,
        family="triposr",
        raw_files=[raw_weight, raw_config],
        resolved_config=resolved_config,
        tensors=tensors,
    )


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)

    if args.family == "pi3":
        normalize_pi3(repo_root)
    elif args.family == "pi3x":
        normalize_pi3x(repo_root)
    elif args.family == "triposr":
        normalize_triposr(repo_root)
    else:
        raise RuntimeError(f"unsupported family: {args.family}")


if __name__ == "__main__":
    main()
