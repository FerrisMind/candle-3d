from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

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


TRIPOSR_TAP_SEMANTICS = {
    "triposr.preprocessed_image": "RGBA-composited and resized conditioning image",
    "triposr.image_tokens": "DINO ViT-B/16 image tokenizer output",
    "triposr.triplane_seed_tokens": "learnable triplane seed tokens",
    "triposr.backbone_tokens": "transformer output tokens before detokenize",
    "triposr.detokenized_triplanes": "detokenized triplane features before upsample network",
    "triposr.scene_codes": "post-processed scene code tensor",
    "triposr.query_positions": "positions passed to query_triplane",
    "triposr.query_features": "concatenated XY/XZ/YZ triplane features",
    "triposr.density_act": "density after exp activation and bias",
    "triposr.color": "sigmoid-composited color features",
    "triposr.mesh_vertices": "mesh vertices after marching cubes and axis reorder",
}

TRIPOSR_EXTRA_SEMANTICS = {
    "triposr.mesh_faces": "triangle index buffer from marching cubes",
    "triposr.vertex_colors": "vertex colors sampled from the scene code",
}

TORCHMCUBES_BACKEND = "torchmcubes"
REMBG_BACKEND = "rembg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture TripoSR Python baseline artifacts.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--sample-kind", choices=["golden", "smoke"], required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-grid-resolution", type=int, default=10)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--mc-threshold", type=float, default=25.0)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument("--no-remove-bg", action="store_true")
    return parser.parse_args()


def load_conditioning_image(source: Path, no_remove_bg: bool, foreground_ratio: float):
    from tsr.utils import remove_background, resize_foreground

    image = Image.open(source)
    has_transparency = image.mode == "RGBA" and image.getextrema()[3][0] < 255
    if no_remove_bg:
        rgb = np.array(image.convert("RGB"))
        preview_rgb = rgb.astype(np.float32) / 255.0
        return Image.fromarray(rgb), preview_rgb

    if REMBG_BACKEND != "rembg" and not has_transparency:
        raise RuntimeError(
            "rembg backend is unavailable and the selected input does not already contain transparency"
        )

    rembg_session = None
    if not has_transparency:
        import rembg

        rembg_session = rembg.new_session()
    image = remove_background(image, rembg_session)
    image = resize_foreground(image, foreground_ratio)
    rgba = np.array(image).astype(np.float32) / 255.0
    rgb = rgba[:, :, :3] * rgba[:, :, 3:4] + (1.0 - rgba[:, :, 3:4]) * 0.5
    preview_rgb = rgb
    return Image.fromarray((rgb * 255.0).astype(np.uint8)), preview_rgb


def install_torchmcubes_fallback() -> None:
    global TORCHMCUBES_BACKEND
    try:
        import torchmcubes  # noqa: F401
    except ModuleNotFoundError:
        from skimage.measure import marching_cubes as skimage_marching_cubes

        module = types.ModuleType("torchmcubes")

        def marching_cubes(level: torch.Tensor, isovalue: float = 0.0):
            if torch.is_tensor(level):
                array = level.detach().cpu().numpy()
                device = level.device
            else:
                array = np.asarray(level, dtype=np.float32)
                device = torch.device("cpu")

            vertices, faces, _normals, _values = skimage_marching_cubes(
                array.astype(np.float32),
                level=isovalue,
            )
            return (
                torch.from_numpy(vertices.astype(np.float32)).to(device),
                torch.from_numpy(faces.astype(np.int64)).to(device),
            )

        module.marching_cubes = marching_cubes
        sys.modules["torchmcubes"] = module
        TORCHMCUBES_BACKEND = "skimage_fallback"


def install_rembg_fallback() -> None:
    global REMBG_BACKEND
    if importlib.util.find_spec("onnxruntime") is None:
        module = types.ModuleType("rembg")

        def new_session():
            return None

        def remove(image, session=None, **_kwargs):
            return image

        module.new_session = new_session
        module.remove = remove
        sys.modules["rembg"] = module
        REMBG_BACKEND = "identity_fallback"
        return

    try:
        import rembg  # noqa: F401
    except BaseException:
        module = types.ModuleType("rembg")

        def new_session():
            return None

        def remove(image, session=None, **_kwargs):
            return image

        module.new_session = new_session
        module.remove = remove
        sys.modules["rembg"] = module
        REMBG_BACKEND = "identity_fallback"


def build_probe_positions(radius: float, resolution: int, device: str) -> torch.Tensor:
    axis = torch.linspace(-radius, radius, resolution, device=device)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((x, y, z), dim=-1).reshape(-1, 3)


def query_probe(model, scene_code: torch.Tensor, positions: torch.Tensor):
    from tsr.utils import scale_tensor

    scaled = scale_tensor(positions, (-model.renderer.cfg.radius, model.renderer.cfg.radius), (-1, 1))
    indices_2d = torch.stack(
        (scaled[..., [0, 1]], scaled[..., [0, 2]], scaled[..., [1, 2]]),
        dim=-3,
    )
    sampled = F.grid_sample(
        rearrange(scene_code, "Np Cp Hp Wp -> Np Cp Hp Wp", Np=3),
        rearrange(indices_2d, "Np N Nd -> Np () N Nd", Np=3),
        align_corners=False,
        mode="bilinear",
    )
    features = rearrange(sampled, "Np Cp () N -> N (Np Cp)", Np=3)
    net_out = model.renderer.query_triplane(model.decoder, positions, scene_code)
    return features, net_out


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    vendor_root = repo_root / "tp" / "3d" / "TripoSR"
    sys.path.insert(0, str(vendor_root))
    install_rembg_fallback()
    install_torchmcubes_fallback()

    from tsr.system import TSR

    set_deterministic_seed(args.seed)
    device = select_device(args.device)
    weight_root = repo_root / "3d" / "models" / "stabilityai-TripoSR"

    model = TSR.from_pretrained(
        str(weight_root),
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    model.eval()

    source_path = Path(args.source).resolve()
    conditioning_image, preview_rgb = load_conditioning_image(
        source_path,
        no_remove_bg=args.no_remove_bg,
        foreground_ratio=args.foreground_ratio,
    )

    with torch.no_grad():
        rgb_cond = model.image_processor([conditioning_image], model.cfg.cond_image_size)[:, None].to(device)
        input_image_tokens = model.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )
        input_image_tokens = rearrange(input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1)
        triplane_seed_tokens = model.tokenizer(rgb_cond.shape[0])
        backbone_tokens = model.backbone(
            triplane_seed_tokens,
            encoder_hidden_states=input_image_tokens,
        )
        detokenized = model.tokenizer.detokenize(backbone_tokens)
        scene_codes = model.post_processor(detokenized)

    probe_positions = build_probe_positions(
        radius=float(model.renderer.cfg.radius),
        resolution=args.probe_grid_resolution,
        device=device,
    )
    with torch.no_grad():
        query_features, probe_out = query_probe(model, scene_codes[0], probe_positions)
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=True,
            resolution=args.mc_resolution,
            threshold=args.mc_threshold,
        )

    mesh = meshes[0]
    vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
    faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64))
    vertex_colors_array = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
    vertex_colors = torch.from_numpy(vertex_colors_array)
    bbox_min, bbox_max = bbox_from_xyz(vertices) if vertices.numel() else (None, None)

    light_root = ensure_dir(repo_root / "3d" / "baselines" / "triposr" / args.sample_id)
    heavy_root = ensure_dir(repo_root / "3d" / "_generated" / "python-baseline" / "triposr" / args.sample_id)

    preview_path = light_root / "input_preview.png"
    save_preview_from_rgb(preview_rgb, preview_path)

    tensor_artifacts = []
    stored_tensors: dict[str, torch.Tensor] = {}
    heavy_paths: list[Path] = []

    if args.sample_kind == "golden":
        tensor_sources = {
            "triposr.preprocessed_image": rgb_cond.detach(),
            "triposr.image_tokens": input_image_tokens.detach(),
            "triposr.triplane_seed_tokens": triplane_seed_tokens.detach(),
            "triposr.backbone_tokens": backbone_tokens.detach(),
            "triposr.detokenized_triplanes": detokenized.detach(),
            "triposr.scene_codes": scene_codes.detach(),
            "triposr.query_positions": probe_positions.detach(),
            "triposr.query_features": query_features.detach(),
            "triposr.density_act": probe_out["density_act"].detach(),
            "triposr.color": probe_out["color"].detach(),
            "triposr.mesh_vertices": vertices.detach(),
            "triposr.mesh_faces": faces.detach(),
            "triposr.vertex_colors": vertex_colors.detach(),
        }

        for name, semantic in TRIPOSR_TAP_SEMANTICS.items():
            artifact, stored = tensor_artifact(
                name=name,
                semantic=semantic,
                tensor=tensor_sources[name],
                storage_relpath="artifacts.safetensors",
                storage_key=name,
                force_fp32=True,
            )
            tensor_artifacts.append(artifact)
            stored_tensors[name] = stored

        for name, semantic in TRIPOSR_EXTRA_SEMANTICS.items():
            artifact, stored = tensor_artifact(
                name=name,
                semantic=semantic,
                tensor=tensor_sources[name],
                storage_relpath="artifacts.safetensors",
                storage_key=name,
                force_fp32=True,
            )
            tensor_artifacts.append(artifact)
            stored_tensors[name] = stored

        artifacts_path = heavy_root / "artifacts.safetensors"
        save_safetensors(stored_tensors, artifacts_path)
        heavy_paths.append(artifacts_path)

        mesh_path = heavy_root / "mesh.obj"
        mesh.export(mesh_path)
        heavy_paths.append(mesh_path)

    summary = {
        "family": "triposr",
        "sample_id": args.sample_id,
        "sample_kind": args.sample_kind,
        "source_path": str(source_path),
        "weight_files": [
            str((weight_root / "model.ckpt").resolve()),
            str((weight_root / "config.yaml").resolve()),
        ],
        "device": device,
        "scene_codes_shape": list(scene_codes.shape),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "first_vertex_samples": sample_triplets(vertices),
        "first_color_samples": sample_triplets(vertex_colors),
        "mc_resolution": args.mc_resolution,
        "mc_threshold": float(args.mc_threshold),
    }

    geometry_summary = {
        "device": device,
        "sampled_frames": None,
        "interval": None,
        "target_size": None,
        "scene_codes_shape": list(scene_codes.shape),
        "point_count": None,
        "mask_true_count": None,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "first_point_samples": [],
        "first_color_samples": summary["first_color_samples"],
        "first_vertex_samples": summary["first_vertex_samples"],
        "mc_resolution": args.mc_resolution,
        "mc_threshold": float(args.mc_threshold),
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
        "family": "triposr",
        "sample_id": args.sample_id,
        "sample_kind": args.sample_kind,
        "source_path": str(source_path),
        "light_root": str(light_root.resolve()),
        "heavy_root": str(heavy_root.resolve()) if args.sample_kind == "golden" else None,
        "weight_files": summary["weight_files"],
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
            "Official TripoSR modules invoked without vendor edits.",
            "model.eval(), torch.no_grad(), autocast disabled, tensors stored in fp32.",
            f"Probe grid resolution fixed at {args.probe_grid_resolution}.",
            "Marching cubes executed with fixed resolution and threshold for baseline parity.",
            f"torchmcubes backend: {TORCHMCUBES_BACKEND}.",
            f"rembg backend: {REMBG_BACKEND}.",
        ],
    }
    write_json(light_root / "manifest.json", manifest)


if __name__ == "__main__":
    main()
