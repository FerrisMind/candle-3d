"""Capture TripoSG Python baseline artifacts.

Mirrors the conventions of ``tools/python_baseline/triposr_dump.py``:

- heavy artifacts (``artifacts.safetensors``, ``mesh.obj``) under the
  runtime-root generated cache: ``3d/_generated/python-baseline/triposg/<sample-id>/``
- light artifacts (``summary.json``, ``manifest.json``, ``checksums.json``,
  ``input_preview.png``) under the runtime-root baseline directory:
  ``3d/baselines/triposg/<sample-id>/``

Reference code is imported read-only from the local TripoSG checkout
(``G:/jobs/candle-based-projects/candle-3d-refs/TripoSG``); no vendor files are
modified. Constraints enforced by this script:

- no ``rembg``/``BriaRMBG``: the input must carry a valid alpha channel and the
  reference ``load_image`` alpha branch is used (hard error otherwise);
- no ``diso.DiffDMC``: the module is stubbed so the reference
  ``hierarchical_extract_geometry`` (skimage marching cubes) can be imported,
  and the stub raises if DiffDMC is ever constructed;
- no weight downloads: ``local_files_only=True`` against a local model dir;
- all baseline tensors are stored F32, CPU, contiguous (integer semantics such
  as face indices are stored as f32 values, matching the triposr baseline and
  its Rust consumption path).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path

# Support `python -I` (isolated mode), which does not put the script directory
# on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from common import (
    bbox_from_xyz,
    build_checksums,
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


DEFAULT_REF_REPO = Path(
    os.environ.get("TRIPOSG_REF_REPO", "G:/jobs/candle-based-projects/candle-3d-refs/TripoSG")
)
GEOMETRY_BOUNDS = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005)
SDF_PROBE_COUNT = 8192
SDF_PROBE_SEED = 1234

# Weight/config files recorded in the manifest, relative to the model dir.
WEIGHT_FILES = [
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

TRIPOSG_TAP_SEMANTICS = {
    "triposg.preprocessed_uint8": (
        "uint8 RGB CHW conditioning image after prepare_image plus BitImageProcessor "
        "resize/crop to 224 (stored as f32 values 0..255)"
    ),
    "triposg.pixel_values": (
        "BitImageProcessor output (rescale + ImageNet normalize) fed to the DINOv2 encoder"
    ),
    "triposg.image_embeds": (
        "DINOv2 image encoder last hidden state (conditional embeds, uncond is zeros_like)"
    ),
    "triposg.latents_init": (
        "initial latents from torch.randn with torch.Generator(device=device).manual_seed(seed)"
    ),
    "triposg.latents_step_000": "latents at step 0 (0 completed scheduler steps; equals latents_init)",
    "triposg.latents_step_001": "latents after 1 completed scheduler step",
    "triposg.latents_step_010": (
        "latents after 10 completed scheduler steps (clamped to the last available step)"
    ),
    "triposg.latents_step_025": (
        "latents after 25 completed scheduler steps (clamped to the last available step)"
    ),
    "triposg.latents_step_final": "latents after the final scheduler step",
    "triposg.velocity_step_000": "CFG-combined transformer output at step 0",
    "triposg.sdf_points": "deterministic SDF probe points in [-1.005, 1.005]^3 (rng seed 1234)",
    "triposg.sdf_values": (
        "raw VAE decoder logits at sdf_points; the reference decoder applies the *-1 negation"
    ),
    "triposg.dense_sdf": (
        "raw VAE decoder logits for the dense octree pass, f32 before the reference fp16 cast"
    ),
    "triposg.mesh_vertices": "marching cubes vertices from hierarchical_extract_geometry",
    "triposg.mesh_faces": (
        "marching cubes triangle indices from hierarchical_extract_geometry "
        "(stored as f32; round to int64 on load)"
    ),
}

# Requested step labels: name -> number of completed scheduler steps.
STEP_LABELS = {
    "latents_step_000": 0,
    "latents_step_001": 1,
    "latents_step_010": 10,
    "latents_step_025": 25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture TripoSG Python baseline artifacts.")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--light-output-dir", type=str, required=True)
    parser.add_argument("--sample-id", type=str, required=True)
    parser.add_argument("--kind", choices=["smoke", "golden"], required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-octree-depth", type=int, default=8)
    parser.add_argument("--hierarchical-octree-depth", type=int, default=9)
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--repo-root", type=str, default=None)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "f32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "f16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in table:
        raise ValueError(f"unsupported --dtype {name!r}; expected one of {sorted(table)}")
    return table[name]


def install_reference_shims() -> None:
    """Make the reference repo importable on this environment without edits.

    - ``diso`` is intentionally not installed: ``triposg.inference_utils`` only
      imports ``DiffDMC`` at module scope for the flash decoder path, which is
      never used here. The stub raises if DiffDMC is ever constructed.
    - transformers 5.x removed ``FLAX_WEIGHTS_NAME``, which diffusers 0.33
      imports unconditionally; restore the legacy constant before diffusers
      loads.
    """
    if "diso" not in sys.modules:
        module = types.ModuleType("diso")

        class _DiffDMC:  # pragma: no cover - must never be constructed
            def __init__(self, *args, **kwargs):
                raise RuntimeError("diso.DiffDMC is forbidden in candle-3d baselines")

        module.DiffDMC = _DiffDMC
        sys.modules["diso"] = module

    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "FLAX_WEIGHTS_NAME"):
        transformers_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"


def install_reference_path() -> Path:
    ref_repo = DEFAULT_REF_REPO.resolve()
    if not ref_repo.is_dir():
        raise RuntimeError(f"TripoSG reference repo not found at {ref_repo}")
    if str(ref_repo) not in sys.path:
        sys.path.insert(0, str(ref_repo))
    scripts_dir = ref_repo / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return ref_repo


def assert_alpha_branch(source: Path) -> None:
    """Fail early unless load_image will take its valid-alpha branch.

    The reference ``load_image`` falls back to the BriaRMBG network when the
    alpha channel is invalid; that path is forbidden in candle-3d baselines.
    """
    import cv2

    img = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        raise RuntimeError(
            f"{source} must be a 4-channel RGBA image with a valid alpha channel; "
            "the reference RMBG fallback path is forbidden here"
        )
    alpha = img[:, :, 3]
    hist = cv2.calcHist([alpha], [0], None, [20], [0, 256])
    min_hist_val = alpha.shape[0] * alpha.shape[1] * 0.01
    if not (hist[0] >= min_hist_val and hist[-1] >= min_hist_val):
        raise RuntimeError(
            f"{source} alpha channel is invalid per the reference is_valid_alpha check; "
            "load_image would invoke BriaRMBG, which is forbidden here"
        )


def load_pipeline(model_dir: Path, device: str, load_dtype: torch.dtype):
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    load_path = "from_pretrained(use_flash_decoder=False)"
    try:
        pipeline = TripoSGPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=load_dtype,
            local_files_only=True,
            use_flash_decoder=False,
        )
    except Exception as error:  # noqa: BLE001 - fall back per task brief
        print(f"[triposg-dump] from_pretrained rejected use_flash_decoder ({error}); retrying without it")
        load_path = "from_pretrained(default kwargs)"
        pipeline = TripoSGPipeline.from_pretrained(
            str(model_dir),
            torch_dtype=load_dtype,
            local_files_only=True,
        )
    pipeline.to(device)
    if load_dtype != torch.float32:
        pipeline.to(load_dtype)
    pipeline.transformer.eval()
    pipeline.vae.eval()
    pipeline.image_encoder_dinov2.eval()
    return pipeline, load_path


def latents_for_label(steps_completed: int, captures: dict[int, torch.Tensor], latents_init: torch.Tensor):
    if steps_completed <= 0:
        return latents_init
    index = min(steps_completed, len(captures)) - 1
    return captures[index]


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    started = time.perf_counter()
    phases: dict[str, float] = {}

    def phase_done(name: str) -> None:
        phases[name] = time.perf_counter() - started
        print(f"[triposg-dump] phase {name}: {phases[name]:.2f}s")

    device = select_device(args.device)
    load_dtype = resolve_dtype(args.dtype)
    if load_dtype != torch.float32:
        print(
            f"[triposg-dump] WARNING: --dtype {args.dtype} is not the baseline float32; "
            "parity runs must use float32"
        )
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "TripoSG reference preprocessing (scripts/image_process.py) hardcodes .cuda(); "
            "this baseline requires a CUDA device"
        )

    set_deterministic_seed(args.seed)
    # Strict F32 arithmetic for baseline parity (no TF32 shortcuts).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model_dir = Path(args.model_dir).resolve()
    source_path = Path(args.image).resolve()
    missing_weights = [name for name in WEIGHT_FILES if not (model_dir / name).is_file()]
    if missing_weights:
        raise RuntimeError(f"missing weight/config files in {model_dir}: {missing_weights}")

    install_reference_shims()
    ref_repo = install_reference_path()
    print(f"[triposg-dump] reference repo: {ref_repo}")
    phase_done("setup")

    from image_process import prepare_image
    from triposg.inference_utils import (
        hierarchical_extract_geometry,
    )

    pipeline, load_path = load_pipeline(model_dir, device, load_dtype)
    print(f"[triposg-dump] pipeline loaded via {load_path}")
    phase_done("pipeline_load")

    # ------------------------------------------------------------------
    # Conditioning image: reference preprocess + DINOv2 feature extraction
    # ------------------------------------------------------------------
    assert_alpha_branch(source_path)
    bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    conditioning_pil = prepare_image(str(source_path), bg_color, None)
    print(
        f"[triposg-dump] conditioning image (alpha branch, no RMBG): "
        f"size={conditioning_pil.size} mode={conditioning_pil.mode}"
    )

    processor = pipeline.feature_extractor_dinov2
    raw_pixels = processor(
        conditioning_pil, return_tensors="pt", do_rescale=False, do_normalize=False
    ).pixel_values
    preprocessed_uint8 = raw_pixels.clamp(0.0, 255.0).round().to(torch.uint8)[0]
    if list(preprocessed_uint8.shape) != [3, 224, 224]:
        raise RuntimeError(
            f"preprocessed_uint8 shape {list(preprocessed_uint8.shape)} != [3, 224, 224]"
        )
    pixel_values = processor(conditioning_pil, return_tensors="pt").pixel_values.to(
        device=device, dtype=load_dtype
    )
    if list(pixel_values.shape) != [1, 3, 224, 224]:
        raise RuntimeError(f"pixel_values shape {list(pixel_values.shape)} != [1, 3, 224, 224]")
    phase_done("preprocess")

    with torch.no_grad():
        image_embeds = pipeline.image_encoder_dinov2(pixel_values).last_hidden_state
    if list(image_embeds.shape) != [1, 257, 1024]:
        raise RuntimeError(f"image_embeds shape {list(image_embeds.shape)} != [1, 257, 1024]")
    # The image encoder is no longer needed; offload it to make room for the
    # transformer/VAE on the 12 GiB card. No numerical effect on later phases.
    pipeline.image_encoder_dinov2.to("cpu")
    torch.cuda.empty_cache()
    phase_done("image_encode")

    # ------------------------------------------------------------------
    # Denoising loop (reproduces TripoSGPipeline.__call__ exactly)
    # ------------------------------------------------------------------
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(args.steps, device=device)
    timesteps = scheduler.timesteps
    sigmas = [float(value) for value in scheduler.sigmas.tolist()]
    timestep_values = [float(value) for value in timesteps.tolist()]
    print(f"[triposg-dump] timesteps={timestep_values}")

    in_channels = int(pipeline.transformer.config.in_channels)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents_init = torch.randn(
        (1, args.num_tokens, in_channels),
        generator=generator,
        device=device,
        dtype=load_dtype,
    )
    print(f"[triposg-dump] latents_init shape={list(latents_init.shape)}")

    image_embeds_cfg = torch.cat([torch.zeros_like(image_embeds), image_embeds], dim=0)
    latents = latents_init
    captures: dict[int, torch.Tensor] = {}
    velocity_step_000: torch.Tensor | None = None
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = pipeline.transformer(
                latent_model_input,
                timestep,
                encoder_hidden_states=image_embeds_cfg,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
            noise_pred_combined = noise_pred_uncond + args.guidance_scale * (
                noise_pred_image - noise_pred_uncond
            )
            if i == 0:
                velocity_step_000 = noise_pred_combined.detach().clone()
            latents = pipeline.scheduler.step(
                noise_pred_combined, t, latents, return_dict=False
            )[0]
            captures[i] = latents.detach().clone()
            print(f"[triposg-dump] denoise step {i:03d}/{len(timesteps)} t={float(t):.3f}")
    latents_final = captures[len(timesteps) - 1]
    # The transformer is no longer needed; offload it so the VAE decode has
    # headroom in F32 on the 12 GiB card. No numerical effect on later phases.
    pipeline.transformer.to("cpu")
    torch.cuda.empty_cache()
    phase_done("denoise")

    # ------------------------------------------------------------------
    # SDF probes at deterministic points
    # ------------------------------------------------------------------
    probe_np = np.random.default_rng(SDF_PROBE_SEED).uniform(
        -1.005, 1.005, (SDF_PROBE_COUNT, 3)
    ).astype(np.float32)
    sdf_points = torch.from_numpy(probe_np).to(device=device, dtype=load_dtype)
    with torch.no_grad():
        sdf_values = pipeline.vae.decode(
            latents_final, sampled_points=sdf_points.unsqueeze(0)
        ).sample
    sdf_values = sdf_values.reshape(-1)
    if sdf_values.shape[0] != SDF_PROBE_COUNT:
        raise RuntimeError(f"sdf_values shape {list(sdf_values.shape)} != [{SDF_PROBE_COUNT}]")
    phase_done("sdf_probe")

    # ------------------------------------------------------------------
    # Geometry: dense pass capture + reference hierarchical marching cubes
    # ------------------------------------------------------------------
    class _GeometryRecorder:
        """Record every vae.decode call made by hierarchical_extract_geometry.

        Call 0 is the dense octree pass; its raw f32 output is captured before
        the reference fp16 cast.
        """

        def __init__(self, vae, latents_ref: torch.Tensor):
            self._vae = vae
            self._latents = latents_ref
            self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def __call__(self, points: torch.Tensor) -> torch.Tensor:
            out = self._vae.decode(self._latents, sampled_points=points).sample
            self.calls.append((points.detach().clone(), out.detach().clone()))
            return out

    recorder = _GeometryRecorder(pipeline.vae, latents_final)
    mesh_v_f = hierarchical_extract_geometry(
        recorder,
        device,
        bounds=list(GEOMETRY_BOUNDS),
        dense_octree_depth=args.dense_octree_depth,
        hierarchical_octree_depth=args.hierarchical_octree_depth,
    )[0]
    if mesh_v_f[0] is None or mesh_v_f[1] is None:
        raise RuntimeError(
            "hierarchical_extract_geometry produced no mesh "
            "(marching cubes failed on the extracted grid)"
        )
    vertices_np, faces_np = mesh_v_f
    vertices = torch.from_numpy(
        np.ascontiguousarray(vertices_np, dtype=np.float32)
    )
    faces = torch.from_numpy(np.ascontiguousarray(faces_np).astype(np.int64))
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"mesh_vertices shape {list(vertices.shape)} is not [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"mesh_faces shape {list(faces.shape)} is not [M, 3]")

    expected_dense_points = (2**args.dense_octree_depth) ** 3
    if not recorder.calls:
        raise RuntimeError("geometric_func was never invoked; dense SDF capture is missing")
    dense_points, dense_logits = recorder.calls[0]
    if dense_points.shape[1] != expected_dense_points:
        raise RuntimeError(
            f"dense pass queried {dense_points.shape[1]} points, expected {expected_dense_points} "
            f"for --dense-octree-depth {args.dense_octree_depth}"
        )
    dense_side = 2**args.dense_octree_depth
    dense_sdf = dense_logits.reshape(dense_side, dense_side, dense_side)
    print(f"[triposg-dump] dense_sdf shape={list(dense_sdf.shape)} (raw f32 logits)")
    phase_done("geometry")

    # ------------------------------------------------------------------
    # Assemble, validate, and save artifacts
    # ------------------------------------------------------------------
    bbox_min, bbox_max = bbox_from_xyz(vertices) if vertices.numel() else (None, None)

    steps_completed_mapping = {
        f"latents_step_{steps:03d}": min(steps, len(timesteps))
        for steps in STEP_LABELS.values()
    }
    steps_completed_mapping["latents_step_final"] = len(timesteps)
    tensor_sources = {
        "triposg.preprocessed_uint8": preprocessed_uint8,
        "triposg.pixel_values": pixel_values,
        "triposg.image_embeds": image_embeds,
        "triposg.latents_init": latents_init,
        "triposg.latents_step_000": latents_for_label(
            steps_completed_mapping["latents_step_000"], captures, latents_init
        ),
        "triposg.latents_step_001": latents_for_label(
            steps_completed_mapping["latents_step_001"], captures, latents_init
        ),
        "triposg.latents_step_010": latents_for_label(
            steps_completed_mapping["latents_step_010"], captures, latents_init
        ),
        "triposg.latents_step_025": latents_for_label(
            steps_completed_mapping["latents_step_025"], captures, latents_init
        ),
        "triposg.latents_step_final": latents_final,
        "triposg.velocity_step_000": velocity_step_000,
        "triposg.sdf_points": sdf_points,
        "triposg.sdf_values": sdf_values,
        "triposg.dense_sdf": dense_sdf,
        "triposg.mesh_vertices": vertices,
        "triposg.mesh_faces": faces,
    }

    problems = []
    for name in TRIPOSG_TAP_SEMANTICS:
        tensor = tensor_sources.get(name)
        if tensor is None:
            problems.append(f"missing tap {name}")
            continue
        if not torch.isfinite(tensor.to(torch.float32)).all():
            problems.append(f"tap {name} contains non-finite values")
    fixed_shapes = {
        "triposg.preprocessed_uint8": [3, 224, 224],
        "triposg.pixel_values": [1, 3, 224, 224],
        "triposg.image_embeds": [1, 257, 1024],
        "triposg.latents_init": [1, args.num_tokens, in_channels],
        "triposg.velocity_step_000": [1, args.num_tokens, in_channels],
        "triposg.sdf_points": [SDF_PROBE_COUNT, 3],
        "triposg.sdf_values": [SDF_PROBE_COUNT],
        "triposg.dense_sdf": [dense_side, dense_side, dense_side],
    }
    for name, expected_shape in fixed_shapes.items():
        actual = list(tensor_sources[name].shape)
        if actual != expected_shape:
            problems.append(f"tap {name} shape {actual} != {expected_shape}")
    if problems:
        raise RuntimeError("baseline tap validation failed:\n  " + "\n  ".join(problems))

    heavy_root = ensure_dir(Path(args.output_dir).resolve())
    light_root = ensure_dir(Path(args.light_output_dir).resolve())

    tensor_artifacts = []
    stored_tensors: dict[str, torch.Tensor] = {}
    for name, semantic in TRIPOSG_TAP_SEMANTICS.items():
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
    heavy_paths: list[Path] = [artifacts_path]

    import trimesh

    mesh = trimesh.Trimesh(
        vertices_np.astype(np.float32), np.ascontiguousarray(faces_np)
    )
    mesh_path = heavy_root / "mesh.obj"
    mesh.export(mesh_path)
    heavy_paths.append(mesh_path)
    phase_done("save_heavy")

    preview_path = light_root / "input_preview.png"
    save_preview_from_rgb(preprocessed_uint8, preview_path)

    summary = {
        "family": "triposg",
        "sample_id": args.sample_id,
        "sample_kind": args.kind,
        "source_path": str(source_path),
        "weight_files": list(WEIGHT_FILES),
        "device": device,
        "dtype": args.dtype,
        "pipeline_load_path": load_path,
        "reference_repo": str(ref_repo),
        "steps": args.steps,
        "guidance_scale": float(args.guidance_scale),
        "seed": args.seed,
        "num_tokens": args.num_tokens,
        "dense_octree_depth": args.dense_octree_depth,
        "hierarchical_octree_depth": args.hierarchical_octree_depth,
        "geometry_bounds": list(GEOMETRY_BOUNDS),
        "timesteps": timestep_values,
        "sigmas": sigmas,
        "image_embeds_shape": list(image_embeds.shape),
        "latents_shape": list(latents_init.shape),
        "step_label_mapping": steps_completed_mapping,
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensor_sources.items()},
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "first_vertex_samples": sample_triplets(vertices),
        "first_color_samples": [],
        "mc_resolution": 2**args.hierarchical_octree_depth,
        "mc_threshold": 0.0,
        "phases_seconds": {key: round(value, 3) for key, value in phases.items()},
    }

    geometry_summary = {
        "device": device,
        "sampled_frames": None,
        "interval": None,
        "target_size": None,
        "scene_codes_shape": None,
        "point_count": None,
        "mask_true_count": None,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "first_point_samples": [],
        "first_color_samples": [],
        "first_vertex_samples": summary["first_vertex_samples"],
        "mc_resolution": 2**args.hierarchical_octree_depth,
        "mc_threshold": 0.0,
    }

    manifest = {
        "family": "triposg",
        "sample_id": args.sample_id,
        "sample_kind": args.kind,
        "source_path": str(source_path),
        "light_root": str(light_root.resolve()),
        "heavy_root": str(heavy_root.resolve()),
        "weight_files": list(WEIGHT_FILES),
        "tensor_artifacts": tensor_artifacts if args.kind == "golden" else [],
        "geometry_summary": geometry_summary,
        "summary_relpath": "summary.json",
        "checksums_relpath": "checksums.json",
        "previews": [
            {
                "label": "input_preview",
                "relative_path": relative_to(preview_path, light_root),
            }
        ],
        "notes": [
            "Official TripoSG modules invoked without vendor edits (read-only reference repo).",
            "model.eval(), torch.no_grad(), TF32 disabled, all tensors stored in fp32.",
            f"Pipeline loaded via {load_path}.",
            "diso module stubbed; diso.DiffDMC is never used (skimage marching cubes only).",
            "No rembg/BriaRMBG: conditioning image used the reference valid-alpha branch.",
            "Image encoder and transformer are offloaded to CPU after their last use "
            "(no numerical effect; frees VRAM for the F32 VAE decode).",
            "Step labels count completed scheduler steps: latents_step_000 equals "
            "latents_init; labels beyond --steps are clamped to the final step "
            "(see step_label_mapping in summary.json).",
            "mesh_faces and preprocessed_uint8 are stored as f32 values (uint8/int64 "
            "semantics), matching the triposr baseline storage convention.",
        ],
    }

    summary_path = light_root / "summary.json"
    write_json(summary_path, summary)
    write_json(heavy_root / "summary.json", summary)
    checksums_path = light_root / "checksums.json"
    checksums = build_checksums(
        repo_root=repo_root,
        paths=[summary_path, preview_path, *heavy_paths],
    )
    write_json(checksums_path, checksums)
    write_json(light_root / "manifest.json", manifest)
    write_json(heavy_root / "manifest.json", manifest)
    phase_done("save_light")

    print("[triposg-dump] captured tensor shapes:")
    for name in TRIPOSG_TAP_SEMANTICS:
        stored = stored_tensors[name]
        print(
            f"[triposg-dump]   {name}: shape={list(stored.shape)} "
            f"dtype={stored.dtype} device={stored.device} contiguous={stored.is_contiguous()}"
        )
    print("[triposg-dump] step label mapping (completed steps):", steps_completed_mapping)
    print(
        f"[triposg-dump] mesh: vertices={len(vertices)} faces={len(faces)} "
        f"bbox_min={bbox_min} bbox_max={bbox_max}"
    )
    print(f"[triposg-dump] heavy artifacts: {artifacts_path} ({artifacts_path.stat().st_size} bytes)")
    print(f"[triposg-dump] light baseline: {light_root}")
    print(f"[triposg-dump] total: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
