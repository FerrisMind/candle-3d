#!/usr/bin/env python3
"""Dump TripoSG flash-decoder / DiffDMC taps for the Rust parity harness.

The reference `flash_extract_geometry` is run unmodified. `DiffDMC` is wrapped
only to capture its input SDF grid and its grid-space vertices/faces; the
returned values are passed through unchanged, and the final world-space mesh
from `flash_extract_geometry` is also dumped for an end-to-end check.

Usage:

  python -I tools/python_baseline/triposg_flash_dump.py \
      --model-dir models/TripoSG \
      --latents ../3d/_generated/python-baseline/triposg/smoke/artifacts.safetensors \
      --octree-depth 6 \
      --output ../3d/_generated/python-baseline/triposg_flash/depth6/artifacts.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from safetensors.torch import save_file


REF_ROOT = Path(r"G:/jobs/candle-based-projects/candle-3d-refs/TripoSG")
DISO_ROOT = Path(r"G:/jobs/candle-based-projects/candle-3d-refs/diso")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--latent-name", default="triposg.latents_step_final")
    parser.add_argument("--octree-depth", type=int, default=6)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def tap(name: str, value) -> tuple[str, torch.Tensor]:
    value = torch.as_tensor(value).detach().to("cpu", dtype=torch.float32).contiguous()
    print(f"  {name}: {list(value.shape)}")
    return name, value


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    latents_path = Path(args.latents).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Diso DiffDMC is CUDA-only")

    sys.path.insert(0, str(REF_ROOT))
    sys.path.insert(0, str(DISO_ROOT))
    from safetensors.torch import load_file
    from triposg.inference_utils import flash_extract_geometry
    from triposg.models.autoencoders import TripoSGVAEModel
    from triposg import inference_utils

    torch.set_grad_enabled(False)
    vae = TripoSGVAEModel.from_pretrained(
        str(model_dir), subfolder="vae"
    ).to(device="cuda", dtype=torch.float32)
    vae.eval()
    vae.set_flash_decoder()

    latents = load_file(str(latents_path))[args.latent_name].cuda().float()
    if latents.ndim == 2:
        latents = latents.unsqueeze(0)
    print("latents", list(latents.shape))

    taps: dict[str, torch.Tensor] = {}
    taps.update([tap("flash.latents", latents)])

    original = inference_utils.DiffDMC

    class CaptureDMC(original):
        def forward(self, grid, *fargs, **fkwargs):
            taps.update([tap("flash.dmc_input_sdf", grid)])
            vertices, faces = super().forward(grid, *fargs, **fkwargs)
            taps.update([tap("flash.dmc_vertices_grid", vertices)])
            taps.update([tap("flash.dmc_faces_grid", faces)])
            return vertices, faces

    inference_utils.DiffDMC = CaptureDMC
    try:
        output = flash_extract_geometry(
            latents,
            vae,
            bounds=(-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
            octree_depth=args.octree_depth,
        )
    finally:
        inference_utils.DiffDMC = original

    vertices, faces = output[0]
    if vertices is None:
        raise RuntimeError("reference flash_extract_geometry returned no mesh")
    taps.update([tap("flash.mesh_vertices", vertices)])
    taps.update([tap("flash.mesh_faces", faces.astype(np.float32))])

    save_file(taps, str(output_path))
    print(
        f"wrote {output_path} ({output_path.stat().st_size} bytes, "
        f"{len(taps)} tensors)"
    )


if __name__ == "__main__":
    main()
