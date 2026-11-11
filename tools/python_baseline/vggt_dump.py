from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    bbox_from_xyz,
    ensure_dir,
    resolve_repo_root,
    sample_triplets,
    save_preview_from_rgb,
    save_safetensors,
    select_device,
    set_deterministic_seed,
    tensor_artifact,
    write_json,
)


VGGT_TAP_SEMANTICS = {
    "vggt.loader.rgb_frames": "preprocessed RGB frames in [0, 1] after VGGT crop-resize",
    "vggt.model.normalized_frames": "ImageNet-normalized frames fed to the DINOv2 patch embed",
    "vggt.encoder.patch_tokens": "DINOv2 ViT-L/14 patch tokens after the final LayerNorm",
    "vggt.aggregator.layer4": "aggregator frame+global tokens cached at layer 4",
    "vggt.aggregator.layer11": "aggregator frame+global tokens cached at layer 11",
    "vggt.aggregator.layer17": "aggregator frame+global tokens cached at layer 17",
    "vggt.aggregator.layer23": "aggregator frame+global tokens cached at layer 23",
    "vggt.camera.pose_enc.0": "camera-head pose encoding after iteration 0",
    "vggt.camera.pose_enc.1": "camera-head pose encoding after iteration 1",
    "vggt.camera.pose_enc.2": "camera-head pose encoding after iteration 2",
    "vggt.camera.pose_enc.3": "camera-head pose encoding after iteration 3",
    "vggt.world_points": "point-head world point map [B, S, H, W, 3]",
    "vggt.world_points_conf": "point-head confidence [B, S, H, W]",
    "vggt.depth": "depth-head depth map [B, S, H, W, 1]",
    "vggt.depth_conf": "depth-head confidence [B, S, H, W]",
    "vggt.camera.extrinsics": "camera extrinsics [B, S, 3, 4]",
    "vggt.camera.intrinsics": "camera intrinsics [B, S, 3, 3]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a VGGT Python baseline artifact.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--ref-root", type=str, required=True)
    parser.add_argument("--raw-model-dir", type=str, required=True)
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--sample-kind", choices=["golden", "smoke"], required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=["f32", "bf16", "f16"], default="f32")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--image-pattern", type=str, default="*")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def collect_images(source: Path, pattern: str, max_images: int) -> list[Path]:
    if source.is_file():
        return [source]
    images = sorted(path for path in source.glob(pattern) if path.is_file())
    if not images:
        raise ValueError(f"no images matched {pattern!r} under {source}")
    if max_images > 0:
        images = images[:max_images]
    return images


def attach_vggt_hooks(model, captured: dict[str, torch.Tensor]) -> None:
    aggregator_forward = model.aggregator.forward
    patch_embed_forward = model.aggregator.patch_embed.forward
    camera_forward = model.camera_head.forward
    point_forward = model.point_head.forward
    depth_forward = model.depth_head.forward

    def traced_patch_embed(self, *args, **kwargs):
        output = patch_embed_forward(*args, **kwargs)
        tensor = output["x_norm_patchtokens"] if isinstance(output, dict) else output
        captured["vggt.encoder.patch_tokens"] = tensor.detach()
        return output

    def traced_aggregator(self, *args, **kwargs):
        tokens, patch_start_idx = aggregator_forward(*args, **kwargs)
        for idx, tensor in enumerate(tokens):
            if tensor is None:
                continue
            captured[f"vggt.aggregator.layer{idx}"] = tensor.detach()
        return tokens, patch_start_idx

    def traced_camera(self, *args, **kwargs):
        outputs = camera_forward(*args, **kwargs)
        for idx, tensor in enumerate(outputs):
            captured[f"vggt.camera.pose_enc.{idx}"] = tensor.detach()
        return outputs

    def traced_point(self, *args, **kwargs):
        preds, conf = point_forward(*args, **kwargs)
        captured["vggt.world_points"] = preds.detach()
        captured["vggt.world_points_conf"] = conf.detach()
        return preds, conf

    def traced_depth(self, *args, **kwargs):
        preds, conf = depth_forward(*args, **kwargs)
        captured["vggt.depth"] = preds.detach()
        captured["vggt.depth_conf"] = conf.detach()
        return preds, conf

    model.aggregator.forward = types.MethodType(traced_aggregator, model.aggregator)
    model.aggregator.patch_embed.forward = types.MethodType(
        traced_patch_embed, model.aggregator.patch_embed
    )
    model.camera_head.forward = types.MethodType(traced_camera, model.camera_head)
    model.point_head.forward = types.MethodType(traced_point, model.point_head)
    model.depth_head.forward = types.MethodType(traced_depth, model.depth_head)


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    vggt_code_root = Path(args.ref_root).resolve()
    source_root = Path(args.source).resolve()
    raw_model_dir = Path(args.raw_model_dir).resolve()
    device_name = select_device(args.device)
    device = torch.device(device_name)
    set_deterministic_seed(args.seed)

    sys.path.insert(0, str(vggt_code_root))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_paths = collect_images(source_root, args.image_pattern, args.max_images)
    images = load_and_preprocess_images([str(path) for path in image_paths])
    dtype = {
        "f32": torch.float32,
        "bf16": torch.bfloat16,
        "f16": torch.float16,
    }[args.dtype]
    images = images.to(device=device, dtype=dtype)

    model = VGGT()
    state = load_file(str(raw_model_dir / "model.safetensors"))
    model.load_state_dict(state)
    model = model.to(device=device).to(dtype=dtype)
    model.eval()

    captured: dict[str, torch.Tensor] = {}
    attach_vggt_hooks(model, captured)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 1, 3, 1, 1)
    images_batched = images.unsqueeze(0) if images.dim() == 4 else images
    normalized = (images_batched - mean) / std
    captured["vggt.loader.rgb_frames"] = images.detach()
    captured["vggt.model.normalized_frames"] = normalized.detach()

    with torch.no_grad():
        predictions = model(images)

    pose_enc = predictions["pose_enc"]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, images.shape[-2:]
    )
    captured["vggt.camera.extrinsics"] = extrinsics.detach()
    captured["vggt.camera.intrinsics"] = intrinsics.detach()

    sample_dir = ensure_dir(
        repo_root
        / "3d"
        / "_generated"
        / "python-baseline"
        / "vggt"
        / args.sample_id
    )
    artifacts_path = sample_dir / "artifacts.safetensors"
    tensors = {}
    artifacts = []
    for name, tensor in captured.items():
        semantic = VGGT_TAP_SEMANTICS.get(name, name)
        artifact, stored = tensor_artifact(
            name=name,
            semantic=semantic,
            tensor=tensor,
            storage_relpath="artifacts.safetensors",
            storage_key=name,
            force_fp32=True,
        )
        artifacts.append(artifact)
        tensors[name] = stored
    save_safetensors(tensors, artifacts_path)

    world_points = captured["vggt.world_points"].to(torch.float32)
    world_conf = captured["vggt.world_points_conf"].to(torch.float32)
    flat_points = world_points.reshape(-1, 3)
    flat_conf = world_conf.reshape(-1)
    keep = flat_conf > 1.0
    kept = flat_points[keep]
    bbox_min, bbox_max = bbox_from_xyz(kept if kept.numel() else flat_points)

    rgb = captured["vggt.loader.rgb_frames"][0]
    preview_path = sample_dir / "preview.png"
    save_preview_from_rgb(rgb[0], preview_path)

    summary = {
        "family": "vggt",
        "sample_id": args.sample_id,
        "sample_kind": args.sample_kind,
        "source_path": str(source_root),
        "device": device_name,
        "dtype": args.dtype,
        "image_paths": [str(path) for path in image_paths],
        "image_shape": list(images.shape),
        "target_size": {"height": int(images.shape[-2]), "width": int(images.shape[-1])},
        "sampled_frames": int(images.shape[0]),
        "point_count": int(kept.numel() // 3),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "first_point_samples": sample_triplets(kept if kept.numel() else flat_points),
        "pose_enc_shape": list(pose_enc.shape),
        "world_points_shape": list(world_points.shape),
        "depth_shape": list(captured["vggt.depth"].shape),
        "tap_names": sorted(captured.keys()),
    }
    write_json(sample_dir / "summary.json", summary)
    write_json(sample_dir / "tensor-manifest.json", {"tensor_artifacts": artifacts})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
