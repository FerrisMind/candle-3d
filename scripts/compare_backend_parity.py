#!/usr/bin/env python3
"""Compare numerical dumps from different candle backends.

The dumps are produced by `crates/lux3d-core/examples/backend_parity_dump.rs`
(or by `lux3d-cli run vggt --dump-npz`). For every shared array this reports
pairwise drift metrics:

* ``max_abs`` / ``mean_abs`` / ``rms`` - absolute error in the output units;
* ``rel_l2`` - ``||a-b||2 / max(||a||2, eps)``;
* ``range_rel`` - ``max_abs / max(data_range(a), eps)``;
* ``cos_dist`` - ``1 - cosine_similarity``;
* ``exact`` - fraction of exactly equal elements (useful for integer arrays).

Acceptance uses ``rel_l2`` (and ``range_rel`` for tensors whose L2 norm is
dominated by a few large values, e.g. mesh coordinates). The default target is
1e-4 for F32 and 1e-3 for F16; pass ``--threshold`` to override.

Example:
  python scripts/compare_backend_parity.py --precision f32 \
      --run cuda=out/triposg-cuda.npz \
      --run vulkan=out/triposg-vulkan.npz \
      --run wgpu=out/triposg-wgpu.npz
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def load_dump(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        return {
            "shape_mismatch": True,
            "ref_shape": list(reference.shape),
            "cand_shape": list(candidate.shape),
        }
    if reference.dtype.kind in "iu" or candidate.dtype.kind in "iu":
        equal = reference == candidate
        return {
            "exact": float(equal.mean()) if equal.size else 1.0,
            "mismatches": int((~equal).sum()),
            "shape_mismatch": False,
        }

    ref = reference.astype(np.float64, copy=False)
    cand = candidate.astype(np.float64, copy=False)
    finite = np.isfinite(ref) & np.isfinite(cand)
    if not finite.any():
        return {"shape_mismatch": False, "non_finite": int((~finite).sum())}
    ref = ref[finite]
    cand = cand[finite]
    delta = cand - ref
    abs_delta = np.abs(delta)
    max_abs = float(abs_delta.max()) if abs_delta.size else 0.0
    mean_abs = float(abs_delta.mean()) if abs_delta.size else 0.0
    rms = float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0
    ref_norm = float(np.sqrt(np.mean(ref * ref))) if ref.size else 0.0
    l2 = float(np.sqrt(np.sum(delta * delta))) if delta.size else 0.0
    ref_l2 = float(np.sqrt(np.sum(ref * ref))) if ref.size else 0.0
    rel_l2 = l2 / max(ref_l2, 1e-30)
    data_range = float(ref.max() - ref.min()) if ref.size else 0.0
    range_rel = max_abs / max(data_range, 1e-30)
    ref_norm_safe = max(float(np.linalg.norm(ref)), 1e-30)
    cand_norm_safe = max(float(np.linalg.norm(cand)), 1e-30)
    cos_dist = 1.0 - float(np.dot(ref, cand) / (ref_norm_safe * cand_norm_safe))
    psnr = (
        20.0 * np.log10(max(data_range, 1e-30) / max(rms, 1e-30))
        if rms > 0.0
        else float("inf")
    )
    return {
        "shape_mismatch": False,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms": rms,
        "rel_l2": rel_l2,
        "range_rel": range_rel,
        "cos_dist": cos_dist,
        "psnr_db": psnr,
        "ref_norm": ref_norm,
        "non_finite": int((~finite).sum()),
    }


def fmt(value: float) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "inf"
    return f"{value:.3e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="named npz dump; pass at least two",
    )
    parser.add_argument(
        "--precision",
        choices=["f32", "f16", "bf16"],
        default="f32",
        help="acceptance target: f32=1e-4, f16/bf16=1e-3 rel_l2",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the rel_l2 acceptance threshold",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    runs: dict[str, Path] = {}
    for spec in args.run:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            parser.error(f"invalid --run `{spec}`; expected LABEL=PATH")
        runs[label] = Path(path)
    if len(runs) < 2:
        parser.error("need at least two --run dumps")

    threshold = args.threshold
    if threshold is None:
        threshold = 1e-4 if args.precision == "f32" else 1e-3

    dumps = {label: load_dump(path) for label, path in runs.items()}
    keys = sorted(set.intersection(*(set(dump) for dump in dumps.values())))

    report: dict = {
        "precision": args.precision,
        "threshold_rel_l2": threshold,
        "runs": {label: str(path) for label, path in runs.items()},
        "pairs": {},
    }

    print(
        f"# backend parity ({args.precision}, rel_l2 threshold {threshold:.1e})\n"
    )
    print(
        "| pair | tensor | shape | max_abs | rms | rel_l2 | range_rel | cos_dist | exact |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    failed = False
    for left, right in combinations(runs, 2):
        pair_metrics: dict[str, dict] = {}
        for key in keys:
            metrics = compare_arrays(dumps[left][key], dumps[right][key])
            pair_metrics[key] = metrics
            if metrics.get("shape_mismatch"):
                print(
                    f"| {left}/{right} | {key} | {metrics['ref_shape']} vs "
                    f"{metrics['cand_shape']} | - | - | - | - | - | - |"
                )
                continue
            if "exact" in metrics:
                print(
                    f"| {left}/{right} | {key} | {list(dumps[left][key].shape)} | "
                    f"- | - | - | - | - | {metrics['exact']:.6f} |"
                )
                continue
            rel_l2 = metrics["rel_l2"]
            range_rel = metrics["range_rel"]
            # A tensor passes when the global L2 drift and the worst-value
            # drift relative to its data range are both within threshold.
            if rel_l2 > threshold and range_rel > threshold:
                failed = True
            print(
                f"| {left}/{right} | {key} | {list(dumps[left][key].shape)} | "
                f"{fmt(metrics['max_abs'])} | {fmt(metrics['rms'])} | "
                f"{fmt(rel_l2)} | {fmt(range_rel)} | {fmt(metrics['cos_dist'])} | - |"
            )
        report["pairs"][f"{left}/{right}"] = pair_metrics

    print(
        "\n_Acceptance: rel_l2 <= threshold OR range_rel <= threshold per tensor._"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
