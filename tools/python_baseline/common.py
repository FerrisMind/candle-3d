from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_root(repo_root: str | None) -> Path:
    return Path(repo_root).resolve() if repo_root is not None else DEFAULT_REPO_ROOT


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def tensor_dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.float32:
        return "f32"
    if tensor.dtype == torch.float16:
        return "f16"
    if tensor.dtype == torch.bfloat16:
        return "bf16"
    if tensor.dtype == torch.bool:
        return "bool"
    raise ValueError(f"Unsupported tensor dtype for manifest: {tensor.dtype}")


def prepared_tensor(tensor: torch.Tensor, force_fp32: bool = True) -> torch.Tensor:
    out = tensor.detach().cpu().contiguous()
    if force_fp32 and out.dtype != torch.bool:
        out = out.to(torch.float32)
    return out.contiguous()


def tensor_artifact(
    *,
    name: str,
    semantic: str,
    tensor: torch.Tensor,
    storage_relpath: str,
    storage_key: str,
    force_fp32: bool = True,
) -> tuple[dict, torch.Tensor]:
    stored = prepared_tensor(tensor, force_fp32=force_fp32)
    artifact = {
        "name": name,
        "semantic": semantic,
        "dtype": tensor_dtype_name(stored),
        "shape": list(stored.shape),
        "storage_relpath": storage_relpath,
        "storage_key": storage_key,
    }
    return artifact, stored


def save_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    ensure_dir(path.parent)
    payload = {key: value.contiguous() for key, value in tensors.items()}
    save_file(payload, str(path))


def write_json(path: Path, payload: dict | list) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_checksums(
    *,
    repo_root: Path,
    paths: Iterable[Path],
) -> dict:
    entries = []
    for path in paths:
        if not path.exists():
            continue
        entries.append(
            {
                "relative_path": relative_to(path, repo_root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {"files": entries}


def save_preview_from_rgb(rgb: torch.Tensor | np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    if isinstance(rgb, torch.Tensor):
        array = rgb.detach().cpu().numpy()
    else:
        array = np.asarray(rgb)

    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255.0).round().astype(np.uint8)

    Image.fromarray(array).save(path)


def bbox_from_xyz(xyz: torch.Tensor | np.ndarray) -> tuple[list[float], list[float]]:
    if isinstance(xyz, torch.Tensor):
        array = xyz.detach().cpu().numpy()
    else:
        array = np.asarray(xyz)
    mins = array.min(axis=0).astype(np.float32).tolist()
    maxs = array.max(axis=0).astype(np.float32).tolist()
    return mins, maxs


def sample_triplets(xyz: torch.Tensor | np.ndarray, limit: int = 8) -> list[list[float]]:
    if isinstance(xyz, torch.Tensor):
        array = xyz.detach().cpu().numpy()
    else:
        array = np.asarray(xyz)
    if array.size == 0:
        return []
    flat = array.reshape(-1, array.shape[-1])
    return flat[:limit].astype(np.float32).tolist()

