from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
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


PI3X_TAP_SEMANTICS = {
    "pi3x.loader.rgb_frames": "sampled RGB frames after resize",
    "pi3x.model.normalized_frames": "ImageNet-normalized input frames",
    "pi3x.conditions.depths": "resized depth condition tensor",
    "pi3x.conditions.intrinsics": "rescaled camera intrinsics",
    "pi3x.conditions.poses": "camera-to-world conditioning poses",
    "pi3x.encoder.patch_tokens": "Pi3X image encoder patch tokens",
    "pi3x.depth_encoder.patch_tokens": "Pi3X depth encoder patch tokens",
    "pi3x.ray_embed.tokens": "Pi3X ray embedding patch tokens",
    "pi3x.decoder.hidden": "Pi3X concatenated decoder hidden states",
    "pi3x.decoder.positions": "Pi3X decoder positions",
    "pi3x.point_decoder.hidden": "Pi3X point decoder hidden states",
    "pi3x.conf_decoder.hidden": "Pi3X confidence decoder hidden states",
    "pi3x.camera_decoder.hidden": "Pi3X camera decoder hidden states",
    "pi3x.metric_decoder.hidden": "Pi3X metric decoder hidden states",
    "pi3x.local_points": "Pi3X local point maps",
    "pi3x.confidence_logits": "Pi3X raw confidence logits",
    "pi3x.camera_poses": "Pi3X camera-to-world matrices",
    "pi3x.points": "Pi3X world-space point cloud",
    "pi3x.rays": "Pi3X normalized ray directions",
    "pi3x.metric": "Pi3X metric scale output",
    "pi3x.export_mask": "Pi3X combined export mask",
    "pi3x.non_edge_mask": "Pi3X non-edge depth mask",
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


def attach_pi3x_hooks(model, captured: dict[str, torch.Tensor]) -> None:
    encoder_forward = model.encoder.forward
    depth_encoder_forward = model.depth_encoder.forward
    ray_embed_forward = model.ray_embed.forward
    decode = model.decode
    point_decoder_forward = model.point_decoder.forward
    conf_decoder_forward = model.conf_decoder.forward
    camera_decoder_forward = model.camera_decoder.forward
    metric_decoder_forward = model.metric_decoder.forward

    def traced_encoder(self, *args, **kwargs):
        output = encoder_forward(*args, **kwargs)
        tensor = output["x_norm_patchtokens"] if isinstance(output, dict) else output
        captured["pi3x.encoder.patch_tokens"] = tensor.detach()
        return output

    def traced_depth_encoder(self, *args, **kwargs):
        output = depth_encoder_forward(*args, **kwargs)
        tensor = output["x_norm_patchtokens"] if isinstance(output, dict) else output
        captured["pi3x.depth_encoder.patch_tokens"] = tensor.detach()
        return output

    def traced_ray_embed(self, *args, **kwargs):
        output = ray_embed_forward(*args, **kwargs)
        captured["pi3x.ray_embed.tokens"] = output.detach()
        return output

    def traced_decode(self, hidden, *args, **kwargs):
        hidden_out, positions = decode(hidden, *args, **kwargs)
        captured["pi3x.decoder.hidden"] = hidden_out.detach()
        captured["pi3x.decoder.positions"] = positions.detach().to(torch.float32)
        return hidden_out, positions

    def traced_point_decoder(self, hidden, xpos=None):
        output = point_decoder_forward(hidden, xpos=xpos)
        captured["pi3x.point_decoder.hidden"] = output.detach()
        return output

    def traced_conf_decoder(self, hidden, xpos=None):
        output = conf_decoder_forward(hidden, xpos=xpos)
        captured["pi3x.conf_decoder.hidden"] = output.detach()
        return output

    def traced_camera_decoder(self, hidden, xpos=None):
        output = camera_decoder_forward(hidden, xpos=xpos)
        captured["pi3x.camera_decoder.hidden"] = output.detach()
        return output

    def traced_metric_decoder(self, hidden, context, xpos=None, ypos=None):
        output = metric_decoder_forward(hidden, context, xpos=xpos, ypos=ypos)
        captured["pi3x.metric_decoder.hidden"] = output.detach()
        return output

    model.encoder.forward = types.MethodType(traced_encoder, model.encoder)
    model.depth_encoder.forward = types.MethodType(traced_depth_encoder, model.depth_encoder)
    model.ray_embed.forward = types.MethodType(traced_ray_embed, model.ray_embed)
    model.decode = types.MethodType(traced_decode, model)
    model.point_decoder.forward = types.MethodType(traced_point_decoder, model.point_decoder)
    model.conf_decoder.forward = types.MethodType(traced_conf_decoder, model.conf_decoder)
    model.camera_decoder.forward = types.MethodType(traced_camera_decoder, model.camera_decoder)
    model.metric_decoder.forward = types.MethodType(traced_metric_decoder, model.metric_decoder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Pi3X Python baseline artifacts.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--raw-model-dir", type=str, required=True)
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--sample-kind", choices=["golden", "smoke"], required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    vendor_root = repo_root / "tp" / "3d" / "Pi3"
    sys.path.insert(0, str(vendor_root))

    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_multimodal_data, write_ply
    from pi3.utils.geometry import depth_edge
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

    captured: dict[str, torch.Tensor] = {}
    attach_pi3x_hooks(model, captured)

    raw_conditions = None
    if args.conditions is not None:
        data_npz = np.load(args.conditions, allow_pickle=True)
        raw_conditions = {
            "intrinsics": data_npz["intrinsics"],
            "poses": data_npz["poses"],
            "depths": data_npz["depths"],
        }

    imgs, conditions = load_multimodal_data(
        args.source,
        conditions=raw_conditions,
        interval=args.interval,
        verbose=False,
        device=device,
    )
    captured["pi3x.loader.rgb_frames"] = imgs.detach()
    captured["pi3x.model.normalized_frames"] = (
        (imgs - model.image_mean) / model.image_std
    ).detach()
    if conditions["depths"] is not None:
        captured["pi3x.conditions.depths"] = conditions["depths"].detach()
    if conditions["intrinsics"] is not None:
        captured["pi3x.conditions.intrinsics"] = conditions["intrinsics"].detach()
    if conditions["poses"] is not None:
        captured["pi3x.conditions.poses"] = conditions["poses"].detach()

    with torch.no_grad():
        with FixedTorchRand():
            result = model(imgs, **conditions)

    captured["pi3x.local_points"] = result["local_points"].detach()
    captured["pi3x.confidence_logits"] = result["conf"].detach()
    captured["pi3x.camera_poses"] = result["camera_poses"].detach()
    captured["pi3x.points"] = result["points"].detach()
    captured["pi3x.rays"] = result["rays"].detach()
    captured["pi3x.metric"] = result["metric"].detach()

    masks = torch.sigmoid(result["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(result["local_points"][..., 2], rtol=0.03)
    export_mask = torch.logical_and(masks, non_edge)
    captured["pi3x.export_mask"] = export_mask.detach()
    captured["pi3x.non_edge_mask"] = non_edge.detach()

    selected_points = result["points"][0][export_mask[0]].detach().cpu()
    selected_colors = imgs[0].permute(0, 2, 3, 1)[export_mask[0].detach().cpu()].detach().cpu()
    bbox_min, bbox_max = bbox_from_xyz(selected_points) if selected_points.numel() else (None, None)

    light_root = ensure_dir(repo_root / "3d" / "baselines" / "pi3x" / args.sample_id)
    heavy_root = ensure_dir(repo_root / "3d" / "_generated" / "python-baseline" / "pi3x" / args.sample_id)

    preview_path = light_root / "input_preview.png"
    save_preview_from_rgb(imgs[0, 0], preview_path)

    tensor_artifacts = []
    stored_tensors: dict[str, torch.Tensor] = {}
    heavy_paths: list[Path] = []

    if args.sample_kind == "golden":
        for name, semantic in PI3X_TAP_SEMANTICS.items():
            if name not in captured:
                continue
            artifact, stored = tensor_artifact(
                name=name,
                semantic=semantic,
                tensor=captured[name],
                storage_relpath="artifacts.safetensors",
                storage_key=name,
                force_fp32=name not in {"pi3x.export_mask", "pi3x.non_edge_mask"},
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
        "conditions_path": str(Path(args.conditions).resolve()) if args.conditions else None,
        "weight_files": [weight_path.name, config_path.name],
        "device": device,
        "sampled_frames": int(imgs.shape[1]),
        "interval": args.interval,
        "target_size": {
            "width": int(imgs.shape[-1]),
            "height": int(imgs.shape[-2]),
        },
        "point_count": int(selected_points.shape[0]),
        "mask_true_count": int(export_mask[0].sum().item()),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "first_point_samples": sample_triplets(selected_points),
        "first_color_samples": sample_triplets(selected_colors),
        "metric": float(result["metric"][0].detach().cpu().item()),
    }

    geometry_summary = {
        "device": device,
        "sampled_frames": int(imgs.shape[1]),
        "interval": args.interval,
        "target_size": summary["target_size"],
        "point_count": int(selected_points.shape[0]),
        "mask_true_count": int(export_mask[0].sum().item()),
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
            "Official Pi3X source tree invoked without vendor edits.",
            "model.eval(), torch.no_grad(), deterministic torch.rand override -> scale_aug = 1.0.",
            f"Sampling interval resolved to {args.interval}.",
        ],
    }
    write_json(light_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
