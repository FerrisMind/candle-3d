#!/usr/bin/env python3
"""Quantization-fidelity metrics for Lux3D outputs.

Compares a quantized (GGUF) run against the same run with float weights.
The reference is the F32/F16 model, not a ground-truth asset, so the report
measures *drift*: how far the quantized prediction moved away from the
float prediction.

Metrics (all implemented here, no training-set assumptions):
  geometry (mesh / point cloud): Chamfer-L1/L2, F-score@tau, normal
    consistency (mesh), voxel IoU (mesh surface occupancy);
  VGGT tensors (.npz): pointmap PSNR / Chamfer / F-score, depth PSNR and
    AbsRel, camera-pose AUC@{5,15,30} degrees on relative poses.

Usage:
  python scripts/compare_geometry.py mesh   ref.obj cand.obj [--json out.json]
  python scripts/compare_geometry.py cloud  ref.ply cand.ply [--voxel 0.002]
  python scripts/compare_geometry.py vggt   ref.npz cand.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- loaders
def load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    # v/vt/vn -> zero-based vertex index
                    idx.append(int(token.split("/")[0]) - 1)
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64)


def load_ply(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header = []
        for line in f:
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        if not any(line.startswith("format ascii") for line in header):
            raise SystemExit(f"{path}: only ascii PLY is supported")
        data = np.loadtxt(f, usecols=(0, 1, 2), dtype=np.float64)
    return np.atleast_2d(data)


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


# ---------------------------------------------------------------- helpers
def _nn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dist, _ = cKDTree(b).query(a, workers=-1)
    return dist


def _diag(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.ptp(points, axis=0)))


def geometry_metrics(
    ref: np.ndarray, cand: np.ndarray, tau_rel: tuple[float, ...] = (0.002, 0.005, 0.01, 0.02)
) -> dict:
    """Symmetric Chamfer + F-score at thresholds relative to the ref bbox diagonal."""
    d_ref = _nn(ref, cand)
    d_cand = _nn(cand, ref)
    diag = max(_diag(ref), 1e-12)
    out: dict = {
        "ref_points": int(len(ref)),
        "cand_points": int(len(cand)),
        "chamfer_l1_mean": float((d_ref.mean() + d_cand.mean()) / 2),
        "chamfer_l2_mean": float(
            (np.sqrt((d_ref**2).mean()) + np.sqrt((d_cand**2).mean())) / 2
        ),
        "chamfer_l1_rms": float(np.sqrt(((d_ref**2).mean() + (d_cand**2).mean()) / 2)),
        "p95": float((np.percentile(d_ref, 95) + np.percentile(d_cand, 95)) / 2),
        "max": float(max(d_ref.max(), d_cand.max())),
        "bbox_diag": diag,
        "chamfer_l1_rel": float(
            ((d_ref.mean() + d_cand.mean()) / 2) / diag
        ),
    }
    for rel in tau_rel:
        tau = rel * diag
        fscore = (
            float(np.mean(d_ref < tau)) + float(np.mean(d_cand < tau))
        ) / 2
        out[f"fscore@{rel:g}"] = fscore
        out[f"fscore_abs@{tau:.6g}"] = fscore
    return out


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(verts)
    if len(faces) == 0:
        return normals
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    face_n = np.cross(b - a, c - a)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_n)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norm, 1e-12)


def normal_consistency(
    ref_v: np.ndarray, ref_f: np.ndarray, cand_v: np.ndarray, cand_f: np.ndarray
) -> float:
    nr = vertex_normals(ref_v, ref_f)
    nc = vertex_normals(cand_v, cand_f)
    _, idx = cKDTree(cand_v).query(ref_v, workers=-1)
    dots = np.abs(np.sum(nr * nc[idx], axis=1))
    return float(dots.mean())


def voxel_iou(ref: np.ndarray, cand: np.ndarray, resolution: int = 128) -> float:
    lo = np.minimum(ref.min(0), cand.min(0))
    hi = np.maximum(ref.max(0), cand.max(0))
    extent = np.maximum(hi - lo, 1e-12)
    cell = float(extent.max() / resolution)
    occupied = []
    for points in (ref, cand):
        keys = np.floor((points - lo) / cell).astype(np.int64)
        occupied.append(set(map(tuple, np.unique(keys, axis=0))))
    inter = len(occupied[0] & occupied[1])
    union = len(occupied[0] | occupied[1])
    return float(inter / union) if union else 1.0


def psnr(ref: np.ndarray, cand: np.ndarray, valid: np.ndarray | None = None) -> float:
    diff = (ref - cand) ** 2
    if valid is not None:
        diff = diff[valid]
    mse = float(diff.mean())
    peak = float(np.abs(ref).max())
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10((peak * peak) / mse))


# ---------------------------------------------------------------- VGGT
def _as_4x4(extr: np.ndarray) -> np.ndarray:
    """Accept [.,3,4] or [.,4,4] extrinsics and return [.,4,4]."""
    extr = np.asarray(extr, np.float64)
    if extr.shape[-2:] == (4, 4):
        return extr
    rot, trans = extr[..., :3, :3], extr[..., :3, 3:4]
    bottom = np.zeros(extr.shape[:-2] + (1, 4), np.float64)
    bottom[..., 3] = 1.0
    return np.concatenate([np.concatenate([rot, trans], axis=-1), bottom], axis=-2)


def _rot_angle_deg(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    cos = (np.trace(ra.swapaxes(-1, -2) @ rb, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _vec_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    if np.all(na < 1e-9) and np.all(nb < 1e-9):
        return np.zeros_like(na)
    cos = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def pose_auc(ref: np.ndarray, cand: np.ndarray, thresholds=(5.0, 15.0, 30.0)) -> dict:
    """Drift AUC: relative poses from the quantized run vs the float run.

    Uses consecutive frame pairs. Error = max(rotation, translation-direction)
    angular error in degrees, matching the VGGT camera-AUC convention.
    """
    ea = _as_4x4(ref)
    eb = _as_4x4(cand)
    if ea.shape != eb.shape:
        return {"error": f"shape mismatch {ea.shape} vs {eb.shape}"}
    ea = ea.reshape(-1, *ea.shape[-2:])
    eb = eb.reshape(-1, *eb.shape[-2:])
    rel_a = np.linalg.inv(ea[:-1]) @ ea[1:]
    rel_b = np.linalg.inv(eb[:-1]) @ eb[1:]
    rot = _rot_angle_deg(rel_a[..., :3, :3], rel_b[..., :3, :3])
    trans = _vec_angle_deg(rel_a[..., :3, 3], rel_b[..., :3, 3])
    err = np.maximum(rot, trans)
    out = {
        "pairs": int(err.size),
        "rot_err_deg_mean": float(rot.mean()),
        "trans_err_deg_mean": float(trans.mean()),
        "err_deg_mean": float(err.mean()),
        "err_deg_p95": float(np.percentile(err, 95)),
    }
    for t in thresholds:
        out[f"auc@{t:g}"] = float(np.mean(err < t))
    return out


def vggt_report(ref: dict[str, np.ndarray], cand: dict[str, np.ndarray]) -> dict:
    out: dict = {}
    if "world_points" in ref and "world_points" in cand:
        a = ref["world_points"].astype(np.float64).reshape(-1, 3)
        b = cand["world_points"].astype(np.float64).reshape(-1, 3)
        if a.shape == b.shape:
            out["pointmap_psnr_db"] = psnr(a, b)
        out["pointmap_geometry"] = geometry_metrics(
            a[:: max(1, len(a) // 200_000)], b[:: max(1, len(b) // 200_000)]
        )
    if "depth" in ref and "depth" in cand:
        dref = ref["depth"].astype(np.float64)
        dcan = cand["depth"].astype(np.float64)
        if dref.shape == dcan.shape:
            valid = None
            if "depth_conf" in ref:
                valid = ref["depth_conf"] > 1.0
            out["depth_psnr_db"] = psnr(dref, dcan, valid)
            if valid is not None:
                abs_rel = np.abs(dref[valid] - dcan[valid]) / np.maximum(
                    np.abs(dref[valid]), 1e-9
                )
                out["depth_abs_rel"] = float(abs_rel.mean())
    if "extrinsics" in ref and "extrinsics" in cand:
        out["camera_pose"] = pose_auc(ref["extrinsics"], cand["extrinsics"])
    return out


# ---------------------------------------------------------------- CLI
def _fmt(report: dict, indent: int = 0) -> str:
    lines = []
    pad = "  " * indent
    for key, value in report.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_fmt(value, indent + 1))
        elif isinstance(value, float):
            lines.append(f"{pad}{key}: {value:.6g}")
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    mesh = sub.add_parser("mesh", help="compare two OBJ meshes")
    mesh.add_argument("reference")
    mesh.add_argument("candidate")
    mesh.add_argument("--json", dest="json_path")
    mesh.add_argument("--no-normals", action="store_true")
    mesh.add_argument("--iou-resolution", type=int, default=128)

    cloud = sub.add_parser("cloud", help="compare two ascii PLY point clouds")
    cloud.add_argument("reference")
    cloud.add_argument("candidate")
    cloud.add_argument("--json", dest="json_path")

    vggt = sub.add_parser("vggt", help="compare two VGGT tensor dumps (.npz)")
    vggt.add_argument("reference")
    vggt.add_argument("candidate")
    vggt.add_argument("--json", dest="json_path")

    args = parser.parse_args()
    if args.mode == "mesh":
        ref_v, ref_f = load_obj(args.reference)
        cand_v, cand_f = load_obj(args.candidate)
        report = geometry_metrics(ref_v, cand_v)
        report["voxel_iou"] = voxel_iou(ref_v, cand_v, args.iou_resolution)
        if not args.no_normals and len(ref_f) and len(cand_f):
            report["normal_consistency"] = normal_consistency(
                ref_v, ref_f, cand_v, cand_f
            )
    elif args.mode == "cloud":
        report = geometry_metrics(load_ply(args.reference), load_ply(args.candidate))
    else:
        report = vggt_report(load_npz(args.reference), load_npz(args.candidate))

    print(_fmt(report))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
