#!/usr/bin/env python3
"""Dump TripoSG VAE encoder taps for the Rust parity harness.

The reference `TripoSGVAEModel.encode` path is used verbatim except for one
deterministic input selection:

  * `x` is a fixed slice of the shipped `surface_point_demo.npy` point cloud
    (positions concatenated with normals, float32);
  * the initial NumPy `Generator.choice` draw is fixed with `seed=1234`;
  * `torch_cluster.fps` runs with `random_start=False`, exactly like
    `_sample_features` does in evaluation mode.

Taps are written to safetensors:

  vae_encode.input              [1, N, 6]
  vae_encode.random_indices     [K]        (NumPy choice, no replacement)
  vae_encode.sampled_raw        [1, K, 6]
  vae_encode.fps_indices        [L]        (indices into sampled_raw)
  vae_encode.sampled_points     [1, L, 6]  (FPS output)
  vae_encode.x_kv               [1, N, 54]
  vae_encode.x_q                [1, L, 54]
  vae_encode.encoder_hidden     [1, L, 512]
  vae_encode.parameters         [1, L, 128]
  vae_encode.mean               [1, L, 64]
  vae_encode.logvar             [1, L, 64]
  vae_encode.std                [1, L, 64]
  vae_encode.noise              [1, L, 64]  (torch CPU randn)
  vae_encode.sample             [1, L, 64]  (mean + std * noise)

Usage:

  python -I tools/python_baseline/triposg_vae_encode_dump.py \
      --model-dir models/TripoSG \
      --surface ../candle-3d-refs/TripoSG/assets/example_data_point/surface_point_demo.npy \
      --output ../3d/_generated/python-baseline/triposg_vae_encode/smoke/artifacts.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from safetensors.torch import save_file
from torch_cluster import fps


REF_ROOT = Path(r"G:/jobs/candle-based-projects/candle-3d-refs/TripoSG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--num-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def tap(name: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
    value = value.detach().to("cpu", dtype=torch.float32).contiguous()
    print(f"  {name}: {list(value.shape)}")
    return name, value


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    surface_path = Path(args.surface).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REF_ROOT))
    from triposg.models.autoencoders import TripoSGVAEModel  # type: ignore
    from triposg.models.autoencoders import autoencoder_kl_triposg  # type: ignore
    from triposg.models.autoencoders.autoencoder_kl_triposg import (  # type: ignore
        TripoSGVAEModel as ConcreteVAE,
    )

    # The upstream file imports `fps` only in a commented-out line, so the
    # documented encode path raises NameError as shipped. Restore the intended
    # torch_cluster symbol before calling it; no model math is changed.
    autoencoder_kl_triposg.fps = fps

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    vae: ConcreteVAE = TripoSGVAEModel.from_pretrained(
        str(model_dir), subfolder="vae"
    ).to(device="cpu", dtype=torch.float32)
    vae.eval()

    demo = np.load(surface_path, allow_pickle=True).tolist()
    points = np.asarray(demo["surface_points"], dtype=np.float32)
    normals = np.asarray(demo["surface_normals"], dtype=np.float32)
    if args.num_points > points.shape[0]:
        raise RuntimeError(
            f"requested {args.num_points} points but the demo has {points.shape[0]}"
        )
    # Deterministic source slice; the reference load_surface also uses a
    # random subset, but a fixed slice makes the parity input reproducible.
    positions = torch.from_numpy(points[: args.num_points]).unsqueeze(0)
    features = torch.from_numpy(normals[: args.num_points]).unsqueeze(0)
    x = torch.cat([positions, features], dim=-1).contiguous()

    rng = np.random.default_rng(args.seed)
    sample_count = args.num_tokens * 4
    random_indices = rng.choice(
        x.shape[1], sample_count, replace=sample_count > x.shape[1]
    )
    sampled_raw = x[:, random_indices]
    print(f"random indices: {random_indices.shape}")

    flat = sampled_raw.view(-1, sampled_raw.shape[-1])
    batch_indices = torch.zeros(flat.shape[0], dtype=torch.long)
    fps_indices = fps(
        flat[:, :3], batch_indices, ratio=1.0 / 4, random_start=False
    )
    sampled_points = flat[fps_indices].view(1, -1, sampled_raw.shape[-1])
    print(f"fps indices: {fps_indices.shape}")

    position_channels = vae.config.in_channels
    positions_q = sampled_points[..., :position_channels]
    features_q = sampled_points[..., position_channels:]
    positions_kv = x[..., :position_channels]
    features_kv = x[..., position_channels:]
    x_q = torch.cat([vae.embedder(positions_q), features_q], dim=-1)
    x_kv = torch.cat([vae.embedder(positions_kv), features_kv], dim=-1)

    encoder_hidden = vae.encoder(x_q, x_kv)
    parameters = vae.quant(encoder_hidden)
    posterior = vae.encode(x, num_tokens=args.num_tokens, seed=args.seed).latent_dist
    # `encode` re-samples with the same seed, so its posterior must match the
    # direct _encode output exactly.
    if not torch.equal(posterior.parameters, parameters):
        delta = (posterior.parameters - parameters).abs().max().item()
        raise RuntimeError(f"encode/_encode mismatch (max delta {delta})")
    mean = posterior.mean
    logvar = posterior.logvar
    std = posterior.std
    noise = torch.randn(mean.shape, dtype=mean.dtype)
    sample = mean + std * noise

    taps = {}
    taps.update([tap("vae_encode.input", x)])
    taps.update(
        [
            tap(
                "vae_encode.random_indices",
                torch.from_numpy(np.asarray(random_indices, dtype=np.float32)),
            )
        ]
    )
    taps.update([tap("vae_encode.sampled_raw", sampled_raw)])
    taps.update(
        [
            tap(
                "vae_encode.fps_indices",
                fps_indices.to(dtype=torch.float32),
            )
        ]
    )
    taps.update([tap("vae_encode.sampled_points", sampled_points)])
    taps.update([tap("vae_encode.x_kv", x_kv)])
    taps.update([tap("vae_encode.x_q", x_q)])
    taps.update([tap("vae_encode.encoder_hidden", encoder_hidden)])
    taps.update([tap("vae_encode.parameters", parameters)])
    taps.update([tap("vae_encode.mean", mean)])
    taps.update([tap("vae_encode.logvar", logvar)])
    taps.update([tap("vae_encode.std", std)])
    taps.update([tap("vae_encode.noise", noise)])
    taps.update([tap("vae_encode.sample", sample)])

    save_file(taps, str(output_path))
    print(
        f"wrote {output_path} ({output_path.stat().st_size} bytes, "
        f"{len(taps)} tensors)"
    )


if __name__ == "__main__":
    main()
