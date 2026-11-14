from __future__ import annotations

import argparse
import json
import struct
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
        choices=["pi3", "pi3x", "rmbg", "triposr", "triposg", "vggt", "trellis2"],
        required=True,
    )
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--raw-model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the raw tree and print a size report without writing "
        "the canonical package (currently implemented for trellis2).",
    )
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


# TRELLIS.2-4B: nine raw safetensors merged under stage prefixes, streaming
# tensor-by-tensor (the merged package is ~15.6 GB; no stage holds the whole
# payload in RAM). Raw list order must stay in lockstep with
# `WeightLocator::locate_trellis2` in `crates/lux3d-core/src/weights/mod.rs`:
# per-ckpt `<stem>.json` then `<stem>.safetensors`, then the two external
# components, then `pipeline.json` — the Rust provenance validator compares
# manifest `raw_files` key order against the plan.
TRELLIS2_CKPT_STEMS = [
    "ss_flow_img_dit_1_3B_64_bf16",
    "slat_flow_img2shape_dit_1_3B_512_bf16",
    "slat_flow_img2shape_dit_1_3B_1024_bf16",
    "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    "shape_dec_next_dc_f16c32_fp16",
    "tex_dec_next_dc_f16c32_fp16",
]

# (raw safetensors relative path, canonical stage prefix, expected tensor
# count — from the weights inventory; expected counts guard against silent
# truncation or stray files in the raw tree).
TRELLIS2_STAGES = [
    ("external/dinov3/model.safetensors", "image_cond", 415),
    ("ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", "ss_flow", 640),
    ("external/ss_dec/ckpts/ss_dec_conv3d_16l8_fp16.safetensors", "ss_dec", 74),
    ("ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", "shape_dit_512", 640),
    ("ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors", "shape_dit_1024", 640),
    ("ckpts/shape_dec_next_dc_f16c32_fp16.safetensors", "shape_dec", 292),
    ("ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", "tex_dit_512", 640),
    ("ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors", "tex_dit_1024", 640),
    ("ckpts/tex_dec_next_dc_f16c32_fp16.safetensors", "tex_dec", 284),
]

TRELLIS2_STAGE_CONFIG_FILES = {
    "image_cond": "external/dinov3/config.json",
    "ss_flow": "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
    "ss_dec": "external/ss_dec/ckpts/ss_dec_conv3d_16l8_fp16.json",
    "shape_dit_512": "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
    "shape_dit_1024": "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json",
    "shape_dec": "ckpts/shape_dec_next_dc_f16c32_fp16.json",
    "tex_dit_512": "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
    "tex_dit_1024": "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
    "tex_dec": "ckpts/tex_dec_next_dc_f16c32_fp16.json",
}

# safetensors JSON header dtype names → element sizes (bytes). These names are
# produced directly by `safe_open(...).get_slice(key).get_dtype()`.
SLICE_DTYPE_ELEMENT_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U8": 1, "BOOL": 1,
}


def trellis2_raw_file_order(raw_model_dir: Path) -> list[Path]:
    """Raw files in the exact `locate_trellis2` order (manifest keys follow)."""
    files: list[Path] = []
    for stem in TRELLIS2_CKPT_STEMS:
        files.append(raw_model_dir / "ckpts" / f"{stem}.json")
        files.append(raw_model_dir / "ckpts" / f"{stem}.safetensors")
    files.append(
        raw_model_dir / "external" / "ss_dec" / "ckpts" / "ss_dec_conv3d_16l8_fp16.json"
    )
    files.append(
        raw_model_dir
        / "external"
        / "ss_dec"
        / "ckpts"
        / "ss_dec_conv3d_16l8_fp16.safetensors"
    )
    files.append(raw_model_dir / "external" / "dinov3" / "config.json")
    files.append(raw_model_dir / "external" / "dinov3" / "model.safetensors")
    files.append(raw_model_dir / "pipeline.json")
    return files


def trellis2_stage_report(raw_model_dir: Path) -> dict:
    """Header-only scan of the nine raw safetensors.

    Validates file presence, per-stage tensor counts, key collisions after
    prefixing, and computes the merged byte size without materializing
    tensors (`get_slice` reads the safetensors header only).
    """
    report: dict = {"stages": [], "tensor_count": 0, "tensor_bytes": 0}
    seen_keys: set[str] = set()
    for relative, prefix, expected_count in TRELLIS2_STAGES:
        path = raw_model_dir / relative
        if not path.is_file():
            raise RuntimeError(f"TRELLIS.2 raw file missing: {path}")
        count = 0
        stage_bytes = 0
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            if expected_count is not None and len(keys) != expected_count:
                raise RuntimeError(
                    f"TRELLIS.2 stage `{prefix}` ({relative}) has {len(keys)} tensors, "
                    f"expected {expected_count}"
                )
            for key in keys:
                merged_key = f"{prefix}.{key}"
                if merged_key in seen_keys:
                    raise RuntimeError(f"duplicate merged key `{merged_key}`")
                seen_keys.add(merged_key)
                slc = handle.get_slice(key)
                shape = slc.get_shape()
                dtype = slc.get_dtype()
                element_size = SLICE_DTYPE_ELEMENT_SIZES.get(dtype)
                if element_size is None:
                    raise RuntimeError(f"unsupported safetensors dtype `{dtype}` in {relative}")
                numel = 1
                for dim in shape:
                    numel *= dim
                stage_bytes += numel * element_size
                count += 1
        report["stages"].append(
            {
                "stage": prefix,
                "raw_file": relative,
                "tensor_count": count,
                "tensor_bytes": stage_bytes,
            }
        )
        report["tensor_count"] += count
        report["tensor_bytes"] += stage_bytes
    report["merged_keys_unique"] = True
    return report


def merge_trellis2_safetensors_streaming(raw_model_dir: Path, canonical_file: Path) -> dict:
    """Merge the nine TRELLIS.2 safetensors into one file, tensor-by-tensor.

    Writes the safetensors container manually (8-byte header length + JSON
    header + payload) so only one tensor is resident at a time. Pass 1 reads
    only the source headers (`get_slice`) to lay out the JSON header and its
    data offsets; pass 2 streams the tensor payloads in the same order. The
    payload must start *after* the header, so the header is written first and
    tensor bytes follow sequentially. Original dtypes are preserved
    byte-exact; keys gain `<stage>.` prefixes.
    """
    header: dict = {}
    histogram: Counter = Counter()
    stage_key_order: list[tuple[str, list[str]]] = []
    offset = 0
    seen_keys: set[str] = set()
    for relative, prefix, expected_count in TRELLIS2_STAGES:
        path = raw_model_dir / relative
        stage_keys: list[str] = []
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
            if expected_count is not None and len(keys) != expected_count:
                raise RuntimeError(
                    f"TRELLIS.2 stage `{prefix}` ({relative}) has {len(keys)} tensors, "
                    f"expected {expected_count}"
                )
            for key in keys:
                merged_key = f"{prefix}.{key}"
                if merged_key in seen_keys:
                    raise RuntimeError(f"duplicate merged key `{merged_key}`")
                seen_keys.add(merged_key)
                slc = handle.get_slice(key)
                shape = slc.get_shape()
                dtype_name = slc.get_dtype()
                if dtype_name not in SLICE_DTYPE_ELEMENT_SIZES:
                    raise RuntimeError(
                        f"unsupported safetensors dtype `{dtype_name}` for `{merged_key}`"
                    )
                numel = 1
                for dim in shape:
                    numel *= dim
                nbytes = numel * SLICE_DTYPE_ELEMENT_SIZES[dtype_name]
                header[merged_key] = {
                    "dtype": dtype_name,
                    "shape": shape,
                    "data_offsets": [offset, offset + nbytes],
                }
                histogram[dtype_name] += 1
                offset += nbytes
                stage_keys.append(key)
        stage_key_order.append((relative, stage_keys))

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad the header so the payload starts 8-byte aligned (matches the
    # reference serializer; not strictly required by the format).
    header_json += b" " * ((-(8 + len(header_json))) % 8)

    with open(canonical_file, "wb") as out:
        out.write(struct.pack("<Q", len(header_json)))
        out.write(header_json)
        for (relative, stage_keys), (_, prefix, _expected) in zip(
            stage_key_order, TRELLIS2_STAGES
        ):
            path = raw_model_dir / relative
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in stage_keys:
                    tensor = handle.get_tensor(key).contiguous()
                    # Reinterpret as raw bytes; `view(torch.uint8)` never
                    # copies and handles bf16, which numpy cannot represent.
                    out.write(tensor.view(torch.uint8).numpy().tobytes())
                    del tensor
    return {
        "tensor_count": len(header),
        "dtype_histogram": dict(sorted(histogram.items())),
        "tensor_bytes": offset,
    }


def verify_trellis2_merge(raw_model_dir: Path, canonical_file: Path) -> None:
    """Re-open the merged package and spot-check one tensor per stage."""
    with safe_open(str(canonical_file), framework="pt", device="cpu") as merged:
        for relative, prefix, _ in TRELLIS2_STAGES:
            with safe_open(
                str(raw_model_dir / relative), framework="pt", device="cpu"
            ) as source:
                source_keys = sorted(source.keys())
                merged_stage_keys = sorted(
                    key for key in merged.keys() if key.startswith(prefix + ".")
                )
                expected = [f"{prefix}.{key}" for key in source_keys]
                if merged_stage_keys != expected:
                    raise RuntimeError(
                        f"TRELLIS.2 stage `{prefix}` key mismatch after merge"
                    )
                probe = source_keys[0]
                reference = source.get_tensor(probe)
                copy = merged.get_tensor(f"{prefix}.{probe}")
                if list(copy.shape) != list(reference.shape) or copy.dtype != reference.dtype:
                    raise RuntimeError(
                        f"TRELLIS.2 tensor `{prefix}.{probe}` shape/dtype drift after merge"
                    )
                if not torch.equal(copy, reference):
                    raise RuntimeError(
                        f"TRELLIS.2 tensor `{prefix}.{probe}` bytes differ after merge"
                    )


def build_trellis2_resolved_config(raw_model_dir: Path) -> dict:
    """`pipeline.json` args + every stage ckpt config + normalization vectors."""
    pipeline_json = json.loads(
        (raw_model_dir / "pipeline.json").read_text(encoding="utf-8")
    )
    pipeline_args = pipeline_json.get("args", pipeline_json)
    stage_configs = {}
    for prefix, relative in TRELLIS2_STAGE_CONFIG_FILES.items():
        stage_configs[prefix] = json.loads(
            (raw_model_dir / relative).read_text(encoding="utf-8")
        )
    resolved_config = {
        "family": "trellis2",
        "architecture": "Trellis2ImageTo3DPipeline",
        "pipeline_args": pipeline_args,
        "stage_configs": stage_configs,
        # Normalization vectors also live inside `pipeline_args`
        # (`*_slat_normalization`); surface them at the top level so the
        # runtime does not depend on the reference pipeline's arg layout.
        "normalization": {
            "shape_slat": pipeline_args.get("shape_slat_normalization"),
            "tex_slat": pipeline_args.get("tex_slat_normalization"),
        },
    }
    return resolved_config


def normalize_trellis2(raw_model_dir: Path, output_dir: Path, dry_run: bool = False) -> None:
    if dry_run:
        report = trellis2_stage_report(raw_model_dir)
        raw_files = trellis2_raw_file_order(raw_model_dir)
        for path in raw_files:
            if not path.is_file():
                raise RuntimeError(f"TRELLIS.2 raw file missing: {path}")
        report["raw_files"] = [path.name for path in raw_files]
        report["raw_weight_bytes"] = sum(path.stat().st_size for path in raw_files)
        report["dry_run"] = True
        print(json.dumps(report, indent=2, sort_keys=True))
        print(
            "trellis2 dry-run passed: "
            f"{report['tensor_count']} tensors, "
            f"{report['tensor_bytes']} tensor bytes merged; "
            "no canonical package written"
        )
        return

    raw_files = trellis2_raw_file_order(raw_model_dir)
    for path in raw_files:
        if not path.is_file():
            raise RuntimeError(f"TRELLIS.2 raw file missing: {path}")
    keys = provenance_keys(raw_files)
    source_checksums = {key: sha256_file(path) for key, path in zip(keys, raw_files)}
    resolved_config = build_trellis2_resolved_config(raw_model_dir)

    target_root = ensure_dir(output_dir)
    canonical_file = target_root / "model.safetensors"
    resolved_config_file = target_root / "resolved_config.json"
    manifest_file = target_root / "manifest.json"
    checksums_file = target_root / "checksums.json"

    # Idempotency fast path: identical raw provenance + resolved config means
    # the previous merge is still valid (avoids re-hashing 15.6 GB on rerun).
    if (
        manifest_file.is_file()
        and canonical_file.is_file()
        and resolved_config_file.is_file()
        and checksums_file.is_file()
    ):
        try:
            existing_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            existing_resolved_config = json.loads(
                resolved_config_file.read_text(encoding="utf-8")
            )
            if (
                existing_manifest.get("family") == "trellis2"
                and existing_manifest.get("normalizer_version") == NORMALIZER_VERSION
                and existing_manifest.get("raw_files") == keys
                and existing_manifest.get("source_checksums") == source_checksums
                and existing_resolved_config == resolved_config
            ):
                print("trellis2 canonical package already up to date; skipping merge")
                return
        except Exception:
            pass

    merged = merge_trellis2_safetensors_streaming(raw_model_dir, canonical_file)
    verify_trellis2_merge(raw_model_dir, canonical_file)
    write_json(resolved_config_file, resolved_config)

    manifest = {
        "family": "trellis2",
        "normalizer_version": NORMALIZER_VERSION,
        "raw_files": keys,
        "canonical_file": canonical_file.name,
        "resolved_config_file": resolved_config_file.name,
        "tensor_count": merged["tensor_count"],
        "dtype_histogram": merged["dtype_histogram"],
        "source_checksums": source_checksums,
    }
    write_json(manifest_file, manifest)
    checksums = build_checksums(
        repo_root=target_root,
        paths=[canonical_file, resolved_config_file, manifest_file],
    )
    write_json(checksums_file, checksums)
    print(
        "trellis2 canonical package written: "
        f"{merged['tensor_count']} tensors, {merged['tensor_bytes']} tensor bytes"
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

    # TRELLIS.2 layout: every basename is unique, so keys stay bare and follow
    # the exact `locate_trellis2` order.
    trellis = [raw / "ckpts" / "a.json", raw / "ckpts" / "a.safetensors", raw / "pipeline.json"]
    assert provenance_keys(trellis) == ["a.json", "a.safetensors", "pipeline.json"]
    print("provenance key self-test passed")


def main() -> None:
    if "--self-test-provenance-keys" in sys.argv:
        self_test_provenance_keys()
        return

    args = parse_args()
    raw_model_dir, output_dir = resolve_paths(args)

    if args.dry_run and args.family != "trellis2":
        raise RuntimeError(f"--dry-run is only implemented for family: trellis2")

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
    elif args.family == "trellis2":
        normalize_trellis2(raw_model_dir, output_dir, dry_run=args.dry_run)
    else:
        raise RuntimeError(f"unsupported family: {args.family}")


if __name__ == "__main__":
    main()
