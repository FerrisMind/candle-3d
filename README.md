<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <img src=".github/assets/lux3d-logo.svg" alt="Lux3D logo" width="512" height="512">
</p>

<p align="center">
  <b>Rust workspace for clean-room 3D inference, contract inspection, and weight canonicalization.</b><br>
  FerrisMind's CUDA-first runtime for Pi3, Pi3X, and TripoSR inside LuxRT.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-FerrisMind-232323" alt="Author FerrisMind">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D</h1>

<p align="center">
  <img src=".github/assets/lux3d-overview.svg" alt="Lux3D workspace overview" width="900">
</p>

## 📚 Table of Contents

- [What is this?](#-what-is-this)
- [Demo](#-demo)
- [Key Features](#-key-features)
- [Repository Layout](#-repository-layout)
- [Installation & Setup](#️-installation--setup)
- [How to Start Using](#-how-to-start-using)
- [System Requirements](#️-system-requirements)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

## ✨ What is this?

Lux3D is the Rust 2024 workspace in `3d-rs` that implements a Candle-first runtime, contract inspection, and export tooling for Pi3, Pi3X, and TripoSR. It is authored by FerrisMind and is intended to run inside the wider `LuxRT` repository, where vendor sources, raw model weights, and canonicalized weights live next to this workspace.

## 🎬 Demo

No public demo asset is checked in yet. The verified local entrypoints are the CLI commands below, which emit `.ply` outputs for Pi3/Pi3X and `.obj` outputs for TripoSR.

## 🚀 Key Features

- `lux3d-core` implements contracts, runtime loading, geometry export, and canonical weight validation.
- `lux3d-cli` exposes `inspect`, `weights normalize`, and `run`.
- Supports three model families: `pi3`, `pi3x`, and `triposr`.
- Keeps canonical safetensors under `3d/canonical-weights/<family>` instead of mutating vendor trees.
- Ships Python baseline tooling in [`tools/python_baseline/README.md`](tools/python_baseline/README.md) for parity capture and weight normalization.
- Separates repository licensing from upstream vendor/model licenses and surfaces per-family license policy through `inspect`.

## 🗂️ Repository Layout

| Path | Purpose |
|------|---------|
| `crates/lux3d-core` | Core runtime, contracts, geometry/export code, and weight validation |
| `crates/lux3d-cli` | CLI front-end for inspection, normalization, and inference runs |
| `tools/python_baseline` | Python parity tooling and canonical-weight normalization scripts |
| `../3d/models` | Raw upstream weights expected by the runtime |
| `../3d/canonical-weights` | Normalized safetensors, manifests, and checksums |
| `../tp/3d` | Vendor source-of-truth repositories used for contract inspection |

`--repo-root` always refers to the parent LuxRT directory that contains `3d-rs`, `3d/`, and `tp/`.

## 🛠️ Installation & Setup

### Prerequisites

- Rust 1.85 or newer
- Cargo with support for `edition = "2024"`
- Python 3.x for normalization and baseline tooling
- NVIDIA GPU plus CUDA runtime for verified `run` commands
- A LuxRT repo root that contains `3d-rs`, `3d/`, and `tp/`

### Expected Parent Layout

```text
<repo-root>/
  3d-rs/
  3d/
    models/
    canonical-weights/
    _generated/
  tp/
    3d/
      Pi3/
      TripoSR/
```

### Workspace Checks

```powershell
cargo metadata --no-deps
cargo run -p lux3d-cli -- --help
```

### Python Baseline Tooling

See [`tools/python_baseline/README.md`](tools/python_baseline/README.md) for environment creation plus golden/smoke dump commands.

## 📖 How to Start Using

### Inspect Contracts And License Notes

```powershell
cargo run -p lux3d-cli -- inspect --repo-root H:\GitHub\LuxRT pi3
cargo run -p lux3d-cli -- inspect --repo-root H:\GitHub\LuxRT pi3x
cargo run -p lux3d-cli -- inspect --repo-root H:\GitHub\LuxRT triposr
```

### Normalize Canonical Weights

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root H:\GitHub\LuxRT pi3
cargo run -p lux3d-cli -- weights normalize --repo-root H:\GitHub\LuxRT pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root H:\GitHub\LuxRT triposr
```

### Run Inference Exports

```powershell
# Pi3 -> PLY
cargo run -p lux3d-cli -- run --repo-root H:\GitHub\LuxRT pi3 --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\house --output H:\GitHub\LuxRT\3d\_generated\pi3.ply

# Pi3X core -> PLY
cargo run -p lux3d-cli -- run --repo-root H:\GitHub\LuxRT pi3x --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\rgb --conditions H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\condition.npz --output H:\GitHub\LuxRT\3d\_generated\pi3x.ply

# Pi3X VO -> PLY
cargo run -p lux3d-cli -- run --repo-root H:\GitHub\LuxRT pi3x --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\skating.mp4 --vo --chunk-size 8 --overlap 4 --conf-threshold 0.05 --inject-condition pose,depth,ray --output H:\GitHub\LuxRT\3d\_generated\pi3x-vo.ply

# TripoSR -> OBJ
cargo run -p lux3d-cli -- run --repo-root H:\GitHub\LuxRT triposr --source H:\GitHub\LuxRT\tp\3d\TripoSR\examples\horse.png --mc-resolution 256 --mc-threshold 25.0 --output H:\GitHub\LuxRT\3d\_generated\triposr.obj
```

## 🖥️ System Requirements

- Windows PowerShell examples are the reference workflow used in this repository.
- `inspect` and `cargo metadata` do not require GPU inference support.
- `run` currently initializes `candle_core::Device::new_cuda(0)`, so verified inference runs require CUDA-capable NVIDIA hardware.
- Raw weights must exist under:
  - `3d/models/yyfz233-Pi3/`
  - `3d/models/yyfz233-Pi3X/`
  - `3d/models/stabilityai-TripoSR/`
- Canonical outputs are expected under `3d/canonical-weights/<family>/`.

## 🙏 Acknowledgments

This workspace stands on top of:

- Pi3 and Pi3X upstream source trees under `tp/3d/Pi3`
- TripoSR under `tp/3d/TripoSR`
- Candle, Clap, nalgebra, mcubes, and safetensors
- FerrisMind as the author of the Rust workspace and repository packaging

## 📄 License

The code and documentation inside `3d-rs` are licensed under Apache 2.0. See [LICENSE](LICENSE).

Vendor source trees, raw model weights, and canonicalized model artifacts keep their upstream licenses and usage restrictions. Use `cargo run -p lux3d-cli -- inspect --repo-root <repo-root> <family>` to review the per-family license policy before redistribution.

Copyright (c) 2026 FerrisMind
