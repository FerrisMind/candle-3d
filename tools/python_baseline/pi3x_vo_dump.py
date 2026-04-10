from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

from common import (
    build_checksums,
    bbox_from_xyz,
    ensure_dir,
    relative_to,
    resolve_repo_root,
    sample_triplets,
    save_preview_from_rgb,
    save_safetensors,
    select_device,
    set_deterministic_seed,
    tensor_artifact,
    write_json,
)


PI3X_VO_TAP_SEMANTICS = {
    "pi3xvo.points": "Pi3X VO merged world-space points",
    "pi3xvo.camera_poses": "Pi3X VO merged camera poses",
    "pi3xvo.confidence_logits": "Pi3X VO merged confidence logits",
    "pi3xvo.sim3_transforms": "Pi3X VO overlap alignment transforms",
}


class FixedTorchRand:
    def __enter__(self):
        self._orig = torch.rand

        def fixed_rand(*args, **kwargs):
            if len(args) == 1 and isinstance(args[0], (tuple, list)):
                shape = tuple(args[0])
            else:
                shape = tuple(args)
            dtype = kwargs.get("dtype", torch.float32)
            device = kwargs.get("device", None)
            return torch.full(shape, 0.5, dtype=dtype, device=device)

        torch.rand = fixed_rand
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.rand = self._orig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Pi3X VO Python baseline artifacts.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--raw-model-dir", type=str, required=True)
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--sample-kind", choices=["golden", "smoke"], required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=4)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--inject-condition", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    vendor_root = repo_root / "tp" / "3d" / "Pi3"
    sys.path.insert(0, str(vendor_root))

    from pi3.models.pi3x import Pi3X
    from pi3.pipe.pi3x_vo import Pi3XVO
    from pi3.utils.basic import load_multimodal_data, write_ply
    from safetensors.torch import load_file

    set_deterministic_seed(args.seed)
    device = select_device(args.device)
    if args.interval < 0:
        args.interval = 10 if str(args.source).lower().endswith(".mp4") else 1
    raw_model_dir = Path(args.raw_model_dir).resolve()

    weight_path = raw_model_dir / "model.safetensors"
    config_path = raw_model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    model = Pi3X(**config).to(device).eval()
    model.load_state_dict(load_file(str(weight_path)))
    pipe = Pi3XVO(model)

    captured: dict[str, torch.Tensor] = {}
    transforms: list[torch.Tensor] = []
    orig_compute = pipe._compute_sim3_umeyama_masked

    def traced_compute(self, *inner_args, **inner_kwargs):
        out = orig_compute(*inner_args, **inner_kwargs)
        transforms.append(out.detach())
        return out

    pipe._compute_sim3_umeyama_masked = types.MethodType(traced_compute, pipe)

    imgs, _ = load_multimodal_data(
        args.source,
        conditions=None,
        interval=args.interval,
        verbose=False,
        device=device,
    )

    with torch.no_grad():
        with FixedTorchRand():
            result = pipe(
                imgs=imgs,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                conf_thre=args.conf_threshold,
                inject_condition=args.inject_condition,
                dtype=torch.float32,
            )

    captured["pi3xvo.points"] = result["points"].detach()
    captured["pi3xvo.camera_poses"] = result["camera_poses"].detach()
    captured["pi3xvo.confidence_logits"] = result["conf"].unsqueeze(-1).detach()
    if transforms:
        captured["pi3xvo.sim3_transforms"] = torch.stack(transforms)
    else:
        captured["pi3xvo.sim3_transforms"] = torch.empty((0, 4, 4), device=result["points"].device)

    masks = result["conf"][0] > args.conf_threshold
    selected_points = result["points"][0][masks].detach().cpu()
    selected_colors = imgs[0].permute(0, 2, 3, 1)[masks.detach().cpu()].detach().cpu()
    bbox_min, bbox_max = bbox_from_xyz(selected_points) if selected_points.numel() else (None, None)

    light_root = ensure_dir(repo_root / "3d" / "baselines" / "pi3x" / args.sample_id)
    heavy_root = ensure_dir(repo_root / "3d" / "_generated" / "python-baseline" / "pi3x" / args.sample_id)
    preview_path = light_root / "input_preview.png"
    save_preview_from_rgb(imgs[0, 0], preview_path)

    tensor_artifacts = []
    stored_tensors: dict[str, torch.Tensor] = {}
    heavy_paths: list[Path] = []
    if args.sample_kind == "golden":
        for name, semantic in PI3X_VO_TAP_SEMANTICS.items():
            artifact, stored = tensor_artifact(
                name=name,
                semantic=semantic,
                tensor=captured[name],
                storage_relpath="artifacts.safetensors",
                storage_key=name,
                force_fp32=True,
            )
            tensor_artifacts.append(artifact)
            stored_tensors[name] = stored
        artifacts_path = heavy_root / "artifacts.safetensors"
        save_safetensors(stored_tensors, artifacts_path)
        heavy_paths.append(artifacts_path)
        if selected_points.numel():
            ply_path = heavy_root / "point_cloud.ply"
            write_ply(selected_points, selected_colors, str(ply_path))
            heavy_paths.append(ply_path)

    summary = {
        "family": "pi3x",
        "sample_id": args.sample_id,
        "sample_kind": args.sample_kind,
        "source_path": str(Path(args.source).resolve()),
        "weight_files": [weight_path.name, config_path.name],
        "device": device,
        "sampled_frames": int(imgs.shape[1]),
        "interval": args.interval,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "conf_threshold": args.conf_threshold,
        "inject_condition": args.inject_condition,
        "point_count": int(selected_points.shape[0]),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "first_point_samples": sample_triplets(selected_points),
        "first_color_samples": sample_triplets(selected_colors),
        "transform_count": int(captured["pi3xvo.sim3_transforms"].shape[0]),
    }

    geometry_summary = {
        "device": device,
        "sampled_frames": int(imgs.shape[1]),
        "interval": args.interval,
        "target_size": {
            "width": int(imgs.shape[-1]),
            "height": int(imgs.shape[-2]),
        },
        "point_count": int(selected_points.shape[0]),
        "mask_true_count": int(masks.sum().item()),
        "vertex_count": None,
        "face_count": None,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "first_point_samples": summary["first_point_samples"],
        "first_color_samples": summary["first_color_samples"],
        "first_vertex_samples": [],
        "scene_codes_shape": None,
        "mc_resolution": None,
        "mc_threshold": None,
    }

    summary_path = light_root / "summary.json"
    write_json(summary_path, summary)
    checksums_path = light_root / "checksums.json"
    checksums = build_checksums(
        repo_root=repo_root,
        paths=[summary_path, preview_path, *heavy_paths],
    )
    write_json(checksums_path, checksums)

    manifest = {
        "family": "pi3x",
        "sample_id": args.sample_id,
        "sample_kind": args.sample_kind,
        "source_path": str(Path(args.source).resolve()),
        "light_root": str(light_root.resolve()),
        "heavy_root": str(heavy_root.resolve()) if args.sample_kind == "golden" else None,
        "weight_files": [weight_path.name, config_path.name],
        "tensor_artifacts": tensor_artifacts if args.sample_kind == "golden" else [],
        "geometry_summary": geometry_summary,
        "summary_relpath": relative_to(summary_path, light_root),
        "checksums_relpath": relative_to(checksums_path, light_root),
        "previews": [
            {
                "label": "input_preview",
                "relative_path": relative_to(preview_path, light_root),
            }
        ],
        "notes": [
            "Official Pi3X VO pipeline invoked without vendor edits.",
            "model.eval(), torch.no_grad(), deterministic torch.rand override -> scale_aug = 1.0.",
            f"chunk_size={args.chunk_size}, overlap={args.overlap}, conf_threshold={args.conf_threshold}, inject_condition={args.inject_condition}.",
        ],
    }
    write_json(light_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
