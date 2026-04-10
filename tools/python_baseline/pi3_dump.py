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


PI3_TAP_SEMANTICS = {
    "pi3.loader.rgb_frames": "sampled RGB frames after resize",
    "pi3.model.normalized_frames": "ImageNet-normalized input frames",
    "pi3.encoder.prepared_tokens": "DINOv2 tokens after cls/pos/register preparation",
    "pi3.encoder.patch_tokens": "DINOv2 patch tokens",
    "pi3.decoder.hidden": "concatenated decoder hidden states",
    "pi3.decoder.positions": "RoPE2D positions for decoder",
    "pi3.point_decoder.hidden": "hidden states before point head",
    "pi3.conf_decoder.hidden": "hidden states before confidence head",
    "pi3.camera_decoder.hidden": "hidden states before camera head",
    "pi3.local_points": "per-view local point maps",
    "pi3.confidence_logits": "raw confidence logits",
    "pi3.camera_poses": "camera-to-world matrices in OpenCV convention",
    "pi3.points": "world-space point cloud after homogenize + einsum",
}

PI3_EXTRA_SEMANTICS = {
    "pi3.export_mask": "combined confidence and non-edge export mask",
    "pi3.non_edge_mask": "non-edge depth mask",
}


def attach_pi3_hooks(model, captured: dict[str, torch.Tensor]) -> None:
    encoder_forward = model.encoder.forward
    prepare_tokens_with_masks = model.encoder.prepare_tokens_with_masks
    decode = model.decode
    point_decoder_forward = model.point_decoder.forward
    conf_decoder_forward = model.conf_decoder.forward
    camera_decoder_forward = model.camera_decoder.forward

    def traced_prepare_tokens_with_masks(self, *args, **kwargs):
        output = prepare_tokens_with_masks(*args, **kwargs)
        captured["pi3.encoder.prepared_tokens"] = output.detach()
        return output

    def traced_encoder(self, *args, **kwargs):
        output = encoder_forward(*args, **kwargs)
        tensor = output["x_norm_patchtokens"] if isinstance(output, dict) else output
        captured["pi3.encoder.patch_tokens"] = tensor.detach()
        return output

    def traced_decode(self, hidden, n_views, height, width):
        hidden_out, positions = decode(hidden, n_views, height, width)
        captured["pi3.decoder.hidden"] = hidden_out.detach()
        captured["pi3.decoder.positions"] = positions.detach().to(torch.float32)
        return hidden_out, positions

    def traced_point_decoder(self, hidden, xpos=None):
        output = point_decoder_forward(hidden, xpos=xpos)
        captured["pi3.point_decoder.hidden"] = output.detach()
        return output

    def traced_conf_decoder(self, hidden, xpos=None):
        output = conf_decoder_forward(hidden, xpos=xpos)
        captured["pi3.conf_decoder.hidden"] = output.detach()
        return output

    def traced_camera_decoder(self, hidden, xpos=None):
        output = camera_decoder_forward(hidden, xpos=xpos)
        captured["pi3.camera_decoder.hidden"] = output.detach()
        return output

    model.encoder.prepare_tokens_with_masks = types.MethodType(
        traced_prepare_tokens_with_masks, model.encoder
    )
    model.encoder.forward = types.MethodType(traced_encoder, model.encoder)
    model.decode = types.MethodType(traced_decode, model)
    model.point_decoder.forward = types.MethodType(traced_point_decoder, model.point_decoder)
    model.conf_decoder.forward = types.MethodType(traced_conf_decoder, model.conf_decoder)
    model.camera_decoder.forward = types.MethodType(traced_camera_decoder, model.camera_decoder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Pi3 Python baseline artifacts.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--raw-model-dir", type=str, required=True)
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--sample-kind", choices=["golden", "smoke"], required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    vendor_root = repo_root / "tp" / "3d" / "Pi3"
    sys.path.insert(0, str(vendor_root))

    from pi3.models.pi3 import Pi3
    from pi3.utils.basic import load_images_as_tensor, write_ply
    from pi3.utils.geometry import depth_edge
    from safetensors.torch import load_file

    set_deterministic_seed(args.seed)
    device = select_device(args.device)
    if args.interval < 0:
        args.interval = 10 if str(args.source).lower().endswith(".mp4") else 1
    raw_model_dir = Path(args.raw_model_dir).resolve()

    config = json.loads((raw_model_dir / "config.json").read_text(encoding="utf-8"))
    weight_path = raw_model_dir / "model.safetensors"

    model = Pi3(**config).to(device).eval()
    model.load_state_dict(load_file(str(weight_path)))

    captured: dict[str, torch.Tensor] = {}
    attach_pi3_hooks(model, captured)

    imgs = load_images_as_tensor(args.source, interval=args.interval, verbose=False)
    captured["pi3.loader.rgb_frames"] = imgs.detach()
    imgs_device = imgs.unsqueeze(0).to(device)
    normalized = (imgs_device - model.image_mean) / model.image_std
    captured["pi3.model.normalized_frames"] = normalized.detach()

    with torch.no_grad():
        result = model(imgs_device)

    captured["pi3.local_points"] = result["local_points"].detach()
    captured["pi3.confidence_logits"] = result["conf"].detach()
    captured["pi3.camera_poses"] = result["camera_poses"].detach()
    captured["pi3.points"] = result["points"].detach()

    masks = torch.sigmoid(result["conf"][..., 0]) > 0.1
    non_edge = ~depth_edge(result["local_points"][..., 2], rtol=0.03)
    export_mask = torch.logical_and(masks, non_edge)
    captured["pi3.export_mask"] = export_mask.detach()
    captured["pi3.non_edge_mask"] = non_edge.detach()

    selected_points = result["points"][0][export_mask[0]].detach().cpu()
    selected_colors = imgs.permute(0, 2, 3, 1)[export_mask[0].detach().cpu()].detach().cpu()
    bbox_min, bbox_max = bbox_from_xyz(selected_points) if selected_points.numel() else (None, None)

    light_root = ensure_dir(repo_root / "3d" / "baselines" / "pi3" / args.sample_id)
    heavy_root = ensure_dir(repo_root / "3d" / "_generated" / "python-baseline" / "pi3" / args.sample_id)

    preview_path = light_root / "input_preview.png"
    save_preview_from_rgb(imgs[0], preview_path)

    tensor_artifacts = []
    stored_tensors: dict[str, torch.Tensor] = {}
    heavy_paths: list[Path] = []

    if args.sample_kind == "golden":
        for name, semantic in PI3_TAP_SEMANTICS.items():
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

        for name, semantic in PI3_EXTRA_SEMANTICS.items():
            artifact, stored = tensor_artifact(
                name=name,
                semantic=semantic,
                tensor=captured[name],
                storage_relpath="artifacts.safetensors",
                storage_key=name,
                force_fp32=name not in {"pi3.export_mask", "pi3.non_edge_mask"},
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
        "family": "pi3",
        "sample_id": args.sample_id,
        "sample_kind": args.sample_kind,
        "source_path": str(Path(args.source).resolve()),
        "weight_files": [weight_path.name],
        "device": device,
        "sampled_frames": int(imgs.shape[0]),
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
    }

    geometry_summary = {
        "device": device,
        "sampled_frames": int(imgs.shape[0]),
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
        "family": "pi3",
        "sample_id": args.sample_id,
        "sample_kind": args.sample_kind,
        "source_path": str(Path(args.source).resolve()),
        "light_root": str(light_root.resolve()),
        "heavy_root": str(heavy_root.resolve()) if args.sample_kind == "golden" else None,
        "weight_files": [weight_path.name],
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
            "Official Pi3 source tree invoked without vendor edits.",
            "model.eval(), torch.no_grad(), autocast disabled, tensors stored in fp32.",
            f"Sampling interval resolved to {args.interval}.",
        ],
    }
    write_json(light_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
