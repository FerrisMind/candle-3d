#!/usr/bin/env python3
"""Run a float reference and GGUF quantized variants, then report drift metrics.

This is the evaluation loop for `lux3d weights quantize`: it executes the same
input through the F32 canonical package and each quantized package, then feeds
the artifacts to `compare_geometry.py`.

Examples:
  python scripts/eval_quantization.py triposg \
      --binary target/release/lux3d-cli.exe \
      --reference /path/to/canonical-weights/triposg \
      --variant q8_0=/path/to/quantized/triposg-q8_0 \
      --variant q4_k=/path/to/quantized/triposg-q4_k \
      --source assets/input.png --device cuda \
      --output-dir out/eval-triposg -- \
      --steps 2 --num-tokens 1024 --flash-octree-depth 5

  python scripts/eval_quantization.py vggt \
      --binary target/release/lux3d-cli.exe \
      --reference /path/to/canonical-weights/vggt \
      --variant q8_0=/path/to/quantized/vggt-q8_0 \
      --source assets/frames --device vulkan \
      --output-dir out/eval-vggt

  python scripts/eval_quantization.py pi3 \
      --binary target/release/lux3d-cli.exe \
      --reference /path/to/canonical-weights/pi3 \
      --variant q8_0=/path/to/quantized/pi3-q8_0 \
      --source assets/frames --device cuda \
      --output-dir out/eval-pi3

  python scripts/eval_quantization.py triposr \
      --binary target/release/lux3d-cli.exe \
      --reference /path/to/canonical-weights/triposr \
      --variant q4_k=/path/to/quantized/triposr-q4_k \
      --source assets/input.png --device cuda \
      --output-dir out/eval-triposr
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_geometry import (  # noqa: E402
    geometry_metrics,
    load_npz,
    load_obj,
    load_ply,
    normal_consistency,
    vggt_report,
    voxel_iou,
)

MESH_FAMILIES = {"triposg", "triposr"}


def run_one(
    binary: str,
    family: str,
    model_path: Path,
    source: Path,
    device: str,
    output: Path,
    dump_npz: Path | None,
    extra: list[str],
) -> None:
    cmd = [
        binary,
        "run",
        family,
        "--device",
        device,
        "--model-path",
        str(model_path),
        "--source",
        str(source),
        "--output",
        str(output),
        *extra,
    ]
    if dump_npz is not None:
        cmd += ["--dump-npz", str(dump_npz)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _lookup(metrics: dict, key: str):
    # Keys such as `fscore@0.005` contain a dot themselves, so try the whole
    # remainder before splitting on the next nested-dict separator.
    node = metrics
    rest = key
    while True:
        if isinstance(node, dict) and rest in node:
            return node[rest]
        head, sep, tail = rest.partition(".")
        if not sep or not isinstance(node, dict) or head not in node:
            return None
        node = node[head]
        rest = tail


def fmt_row(name: str, metrics: dict) -> str:
    def g(*keys: str, spec: str = ".4g") -> str:
        for key in keys:
            value = _lookup(metrics, key)
            if isinstance(value, (int, float)):
                return format(value, spec)
        return "-"

    return (
        f"| {name} "
        f"| {g('chamfer_l1_mean', 'pointmap_geometry.chamfer_l1_mean')} "
        f"| {g('chamfer_l1_rel', 'pointmap_geometry.chamfer_l1_rel')} "
        f"| {g('fscore@0.005', 'pointmap_geometry.fscore@0.005')} "
        f"| {g('fscore@0.01', 'pointmap_geometry.fscore@0.01')} "
        f"| {g('voxel_iou')} | {g('normal_consistency')} "
        f"| {g('pointmap_psnr_db')} | {g('depth_psnr_db')} "
        f"| {g('camera_pose.auc@30')} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=sorted(MESH_FAMILIES | {"vggt", "pi3", "pi3x"}))
    parser.add_argument("--binary", required=True)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="quantized package directory, e.g. q8_0=/models/triposg-q8_0",
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    # Model-specific `run` flags (steps, tokens, octree depth, RMBG path, ...)
    # are unknown to this parser; collect them via parse_known_args instead of
    # a REMAINDER positional, which would swallow the options above.
    args, extra = parser.parse_known_args()

    extra = [item for item in extra if item != "--"]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    is_mesh = args.family in MESH_FAMILIES
    suffix = "obj" if is_mesh else "ply"

    reference_artifact = out_dir / f"reference.{suffix}"
    reference_npz = out_dir / "reference.npz" if args.family == "vggt" else None
    run_one(
        args.binary,
        args.family,
        args.reference,
        args.source,
        args.device,
        reference_artifact,
        reference_npz,
        extra,
    )

    if is_mesh:
        ref_v, ref_f = load_obj(str(reference_artifact))
    elif args.family == "vggt":
        ref_npz = load_npz(str(reference_npz))
    else:
        ref_cloud = load_ply(str(reference_artifact))

    reports: dict[str, dict] = {}
    for spec in args.variant:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--variant expects NAME=PATH, got {spec!r}")
        artifact = out_dir / f"{name}.{suffix}"
        npz_path = out_dir / f"{name}.npz" if args.family == "vggt" else None
        run_one(
            args.binary,
            args.family,
            Path(path),
            args.source,
            args.device,
            artifact,
            npz_path,
            extra,
        )
        if is_mesh:
            cand_v, cand_f = load_obj(str(artifact))
            metrics = geometry_metrics(ref_v, cand_v)
            metrics["voxel_iou"] = voxel_iou(ref_v, cand_v)
            if len(ref_f) and len(cand_f):
                metrics["normal_consistency"] = normal_consistency(
                    ref_v, ref_f, cand_v, cand_f
                )
        elif args.family == "vggt":
            metrics = vggt_report(ref_npz, load_npz(str(npz_path)))
        else:
            metrics = geometry_metrics(ref_cloud, load_ply(str(artifact)))
        reports[name] = metrics

    print()
    print(
        "| variant | chamfer_l1 | rel | F@0.5% | F@1% | voxel IoU | normal "
        "| pointmap PSNR dB | depth PSNR dB | pose AUC@30 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, metrics in reports.items():
        print(fmt_row(name, metrics))

    if args.json_path:
        args.json_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
