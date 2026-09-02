<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>Rust workspace for clean-room 3D inference, contract inspection, and weight canonicalization.</b><br>
  CUDA-first runtime for Pi3, Pi3X, and TripoSR.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/Rust-1.85+-93450a?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/Runtime-CUDA%20first-76B900" alt="CUDA first runtime">
</p>

<h1 align="center">Lux3D / candle-3d</h1>

## Table of Contents

- [What is this?](#what-is-this)
- [Key Features](#key-features)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [License](#license)

## What is this?

Lux3D / candle-3d is a Rust 2024 workspace that provides:

- a Candle-first inference runtime
- contract inspection for supported model families
- canonical weight normalization and validation
- geometry export tooling for point clouds and meshes

Supported model families:

- `pi3`
- `pi3x`
- `triposr`

## Key Features

- `lux3d-core` implements contracts, runtime loading, export logic, and weight validation.
- `lux3d-cli` exposes `inspect`, `weights normalize`, and `run`.
- `lux3d-server-core` exposes an OpenAI-style async HTTP API for Pi3, Pi3X, and TripoSR generation.
- Python baseline tooling lives in [`tools/python_baseline/README.md`](https://github.com/oxide-lab/Lux3d/blob/main/tools/python_baseline/README.md).
- Model-family-specific licensing can be inspected through the CLI before redistribution.

## Repository Layout

| Path | Purpose |
|------|---------|
| `crates/lux3d-core` | Core runtime, contracts, geometry/export code, and weight validation |
| `crates/lux3d-cli` | CLI front-end for inspection, normalization, and inference runs |
| `crates/lux3d-server-core` | Embeddable Axum router and HTTP API for external applications |
| `crates/lux3d-server` | Standalone HTTP server binary |
| `tools/python_baseline` | Python parity tooling and canonical-weight normalization scripts |
| `.github/assets` | Project documentation assets |

## Quick Start

### Workspace Checks

```powershell
cargo metadata --no-deps
cargo run -p lux3d-cli -- --help
```

### Inspect Contracts

```powershell
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> pi3
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> pi3x
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposr
```

### Normalize Canonical Weights

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --output-dir <canonical-pi3-dir> pi3
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --output-dir <canonical-pi3x-dir> pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --output-dir <canonical-triposr-dir> triposr
```

### Run Inference

```powershell
# Pi3 -> PLY with an explicit canonical package directory (layout matches 3d/canonical-weights/pi3)
cargo run -p lux3d-cli -- run pi3 --model-path <canonical-pi3-dir> --source <input-sequence> --output <output-file.ply>

# Pi3X core -> PLY with an explicit canonical package directory (layout matches 3d/canonical-weights/pi3x)
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-sequence> --conditions <conditions-file> --output <output-file.ply>

# Pi3X VO -> PLY with an explicit canonical package directory (layout matches 3d/canonical-weights/pi3x)
cargo run -p lux3d-cli -- run pi3x --model-path <canonical-pi3x-dir> --source <input-video> --vo --chunk-size 8 --overlap 4 --conf-threshold 0.05 --inject-condition pose,depth,ray --output <output-file.ply>

# TripoSR -> OBJ with an explicit canonical package directory (layout matches 3d/canonical-weights/triposr)
cargo run -p lux3d-cli -- run triposr --model-path <canonical-triposr-dir> --source <input-image> --mc-resolution 256 --mc-threshold 25.0 --output <output-file.obj>

# Auto-download canonical packages from Hugging Face into the user cache
cargo run -p lux3d-cli -- run pi3 --source <input-sequence> --output <output-file.ply>
```

### Run the HTTP Server

```powershell
cargo run -p lux3d-server -- --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

Generation endpoints accept `multipart/form-data` and return `202 Accepted` with an async job object:

| Endpoint | Models | Output |
|----------|--------|--------|
| `POST /v1/point-clouds/generations` | `pi3`, `pi3x` | PLY |
| `POST /v1/meshes/generations` | `triposr` | OBJ |

Common fields:

- `model` — required for point clouds (`pi3` or `pi3x`), optional for meshes (`triposr` default)
- `source` — required upload (`.zip` frame archive, `.mp4` video, or image)
- `conditions` — optional Pi3X conditions file
- `options` — optional JSON string with family-specific parameters

Workflow:

```powershell
# 1. Create a mesh generation job
curl -X POST http://127.0.0.1:8080/v1/meshes/generations `
  -F "model=triposr" `
  -F "source=@input.png" `
  -F "options={\"mc_resolution\":256,\"mc_threshold\":25.0}"

# 2. Poll job status
curl http://127.0.0.1:8080/v1/generations/gen_...

# 3. Download completed asset
curl http://127.0.0.1:8080/v1/generations/gen_.../content --output mesh.obj
```

Embed the router in another Axum app:

```rust
use lux3d_server_core::{
    lux3d_for_server_builder::Lux3dForServerBuilder,
    lux3d_server_router_builder::Lux3dServerRouterBuilder,
};

let lux3d = Lux3dForServerBuilder::new().build()?;
let routes = Lux3dServerRouterBuilder::new()
    .with_lux3d(lux3d)
    .build()
    .await?;
let app = axum::Router::new().nest("/api/lux3d", routes);
```

## System Requirements

- Rust 1.85 or newer
- Cargo with support for `edition = "2024"`
- Python 3.x for baseline tooling and normalization
- CUDA-capable NVIDIA hardware for verified runtime inference
- Metal backend on macOS is not tested in this repository, but may be supported theoretically.
- Canonical model package directories supplied with `--model-path`, or downloadable from Hugging Face into the user cache

## License

The code and documentation in this repository are licensed under Apache 2.0. See [LICENSE](https://github.com/oxide-lab/Lux3d/blob/main/LICENSE).

Upstream model assets and canonicalized model artifacts keep their original licenses and usage restrictions. Review model-family-specific terms before redistribution.
