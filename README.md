<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-5B7CFA" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>Rust workspace for clean-room 3D inference, contract inspection, and weight canonicalization.</b><br>
  CUDA-first runtime for Pi3, Pi3X, TripoSR, VGGT, and TripoSG.
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
- [Demo](#demo)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
- [Performance](#performance)
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
- `vggt` — VGGT-1B multi-view point/depth/camera inference. Original weights are
  CC-BY-NC-4.0 (non-commercial); the commercial checkpoint is distributed separately.
- `triposg` — TripoSG image-to-mesh inference (DINOv2-large encoder, 21-block rectified-flow
  DiT, SDF VAE decoder/encoder, reference-default flash decoder + clean-room DiffDMC). MIT
  licensed; the BriaRMBG 1.4 fallback for RGB input ships as the separate `rmbg-1.4` package
  under the BRIA license.

## Key Features

- `lux3d-core` implements contracts, runtime loading, export logic, and weight validation.
- `lux3d-cli` exposes `inspect`, `weights normalize`, and `run`.
- `lux3d-server-core` exposes an OpenAI-style async HTTP API for Pi3, Pi3X, TripoSR, VGGT, and TripoSG generation.
- Python baseline tooling lives in [`tools/python_baseline/README.md`](https://github.com/oxide-lab/Lux3d/blob/main/tools/python_baseline/README.md).
- Model-family-specific licensing can be inspected through the CLI before redistribution.
- **Experimental:** Vulkan and WGPU inference via `--device vulkan` / `--device wgpu` (build with `--features vulkan` / `--features wgpu`). CUDA is the primary backend; the Pi3/Pi3X/TripoSR and VGGT paths have Vulkan correctness checks, and TripoSG has Vulkan/WGPU smoke coverage (see [Performance](#performance)). WGPU remains experimental — may be unstable, produce wrong results, or not work at all depending on hardware and drivers.

## Demo

https://github.com/user-attachments/assets/0a9f766f-68d3-482a-9c94-c55d0c355808

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
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> vggt
cargo run -p lux3d-cli -- inspect --repo-root <runtime-root> triposg
```

### Normalize Canonical Weights

```powershell
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --output-dir <canonical-pi3-dir> pi3
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --output-dir <canonical-pi3x-dir> pi3x
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --output-dir <canonical-triposr-dir> triposr
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-vggt-dir> --output-dir <canonical-vggt-dir> vggt
cargo run -p lux3d-cli -- weights normalize --repo-root <runtime-root> --raw-model-dir <raw-triposg-dir> --output-dir <canonical-triposg-dir> triposg
```

### Quantize Weights to GGUF

`weights quantize` packs the canonical `model.safetensors` into `model.gguf`.
Only rank-2 `*.weight` matrices become GGUF blocks; norms, biases, embeddings,
convolutions, and rows that do not divide into the dtype block stay in the
fallback dtype. Quantized linears run through `QMatMul` (integer dot-product
kernels), everything else is dequantized on demand.

The converter embeds the sibling `resolved_config.json` in GGUF metadata, so
`model.gguf` can be used as a **standalone package**: put it alone in a
directory and pass that directory as `--model-path`. The loader synthesizes
the manifest/checksums and reads the config from the GGUF. If `model.gguf`
sits next to a full canonical package, the package files are still validated
and the GGUF supplies the weights. Rank-0 tensors (training-only buffers such
as BatchNorm `num_batches_tracked`) are omitted; they are not used at
inference. `--model-path` also accepts a standalone `.gguf` file directly, so
quantized files can be distributed flat (`TripoSG-1.5B-Q8_0.gguf`,
`VGGT-1B-Q8_0.gguf`, ...).

`general.architecture` is derived from the resolved config: top-level
`architecture`, then `family`, then the model name (`triposg`, `vggt`,
`rmbg`). For example RMBG writes `BriaRMBG`, TripoSG writes `triposg`, and
VGGT (whose config has neither key) writes `vggt`. `lux3d.family` always
records the runtime package family.

```powershell
# Q8_0 weights, F16 fallback (reference-quality default)
cargo run -p lux3d-cli -- weights quantize triposg `
  --source <canonical-triposg-dir>/model.safetensors `
  --output <gguf-triposg-dir>/model.gguf --dtype q8_0 --fallback-dtype f16

# K-quants: smaller files, 256-wide blocks
cargo run -p lux3d-cli -- weights quantize vggt `
  --source <canonical-vggt-dir>/model.safetensors `
  --output <gguf-vggt-dir>/model.gguf --dtype q4_k --fallback-dtype f16

# Pi3, Pi3X and TripoSR use the same converter.
cargo run -p lux3d-cli -- weights quantize pi3 `
  --source <canonical-pi3-dir>/model.safetensors `
  --output <gguf-pi3-dir>/model.gguf --dtype q8_0 --fallback-dtype f16
cargo run -p lux3d-cli -- weights quantize pi3x `
  --source <canonical-pi3x-dir>/model.safetensors `
  --output <gguf-pi3x-dir>/model.gguf --dtype q8_0 --fallback-dtype f16
cargo run -p lux3d-cli -- weights quantize triposr `
  --source <canonical-triposr-dir>/model.safetensors `
  --output <gguf-triposr-dir>/model.gguf --dtype q8_0 --fallback-dtype f16

# RMBG-1.4 is an auxiliary package: separate GGUF, F16, same standalone rules.
cargo run -p lux3d-cli -- weights quantize --aux-family rmbg `
  --source <canonical-rmbg-dir>/model.safetensors `
  --output <gguf-rmbg-dir>/model.gguf --dtype f16 --fallback-dtype f16

# Run with only the GGUF directories; no safetensors/manifest/checksums needed.
cargo run -p lux3d-cli -- run triposg --device cuda `
  --model-path <gguf-triposg-dir> --rmbg-model-path <gguf-rmbg-dir> `
  --source <image> --output out.obj

# Optional: `--all-2d` also quantizes non-`*.weight` rank-2 tensors;
# `--prefix transformer.` (repeatable) restricts quantization for bisection.
```

Supported `--dtype` values: `f16`, `f32`, `bf16`, `q8_0`, `q4_0`, `q4_1`,
`q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`. `q8_1`/`q8_k` are
activation-side GGML types and are rejected as weight targets. A GGUF created
without an embedded config (no sibling `resolved_config.json` at conversion
time) can still sit next to a full canonical package, but is rejected as a
standalone package.

Measured on the bundled canonical packages (`RTX 3060 12 GiB`; TripoSG
`--steps 2 --num-tokens 1024 --flash-octree-depth 5`, VGGT/Pi3/Pi3X 3 frames,
TripoSR 1 image at `--mc-resolution 256`):

| model | variant | GGUF size | weights kept |
|---|---|---|---|
| TripoSG | canonical F32 | 7.40 GiB | — |
| TripoSG | q8_0 | 1.97 GiB | 527 matrices |
| TripoSG | q6_k | 1.52 GiB | 525 |
| TripoSG | q5_k | 1.28 GiB | 525 |
| TripoSG | q4_k | 1.05 GiB | 525 |
| VGGT | canonical F32 | 4.68 GiB | — |
| VGGT | q8_0 | 1.33 GiB | 384 |
| VGGT | q6_k | 1.09 GiB | 331 |
| VGGT | q5_k | 0.95 GiB | 331 |
| VGGT | q4_k | 0.82 GiB | 331 |
| Pi3 | canonical F32 | 3.57 GiB | — |
| Pi3 | q8_0 | 0.95 GiB | 318 |
| Pi3 | q6_k | 0.74 GiB | 318 |
| Pi3 | q5_k | 0.62 GiB | 318 |
| Pi3 | q4_k | 0.51 GiB | 318 |
| Pi3X | canonical F32 | 5.07 GiB | — |
| Pi3X | q8_0 | 1.37 GiB | 466 |
| Pi3X | q6_k | 1.06 GiB | 466 |
| Pi3X | q5_k | 0.90 GiB | 466 |
| Pi3X | q4_k | 0.74 GiB | 466 |
| TripoSR | canonical F32 | 1.56 GiB | — |
| TripoSR | q8_0 | 0.42 GiB | 244 |
| TripoSR | q6_k | 0.33 GiB | 235 |
| TripoSR | q5_k | 0.27 GiB | 235 |
| TripoSR | q4_k | 0.23 GiB | 235 |

TripoSR Q8_0 quantizes 244 matrices; the K-quants use 256-wide blocks, so nine
matrices whose row length is not divisible by 256 stay in F16.

RMBG-1.4 (auxiliary): F32 0.16 GiB → F16 0.08 GiB; 686 tensors written,
112 rank-0 training buffers omitted.

### Quantization Fidelity

`scripts/compare_geometry.py` computes drift against the float reference:
symmetric Chamfer-L1 (relative to the reference bbox diagonal), F-score@tau,
surface voxel IoU, normal consistency for meshes; pointmap/depth PSNR,
AbsRel, Chamfer and camera-pose AUC for VGGT tensor dumps.
`scripts/eval_quantization.py` runs the reference plus every quantized package
and prints the table.

```powershell
# Mesh drift (TripoSG/TripoSR)
python scripts/compare_geometry.py mesh <reference.obj> <quantized.obj>

# VGGT: dump raw tensors next to the PLY, then compare
cargo run -p lux3d-cli -- run vggt --model-path <pkg> --source <frames> --output <out.ply> --dump-npz <out.npz>
python scripts/compare_geometry.py vggt <reference.npz> <quantized.npz>

# One-shot sweep over several quantized packages
python scripts/eval_quantization.py triposg --binary target/release/lux3d-cli.exe `
  --reference <canonical-triposg-dir> --variant q8_0=<dir-q8_0> --variant q4_k=<dir-q4_k> `
  --source <input-image> --device cuda --output-dir out/eval -- --steps 2 --num-tokens 1024 --flash-octree-depth 5

# The same loop covers the point-cloud and TripoSR families.
python scripts/eval_quantization.py pi3x --binary target/release/lux3d-cli.exe `
  --reference <canonical-pi3x-dir> --variant q8_0=<dir-q8_0> `
  --source <frames-dir> --device cuda --output-dir out/eval-pi3x
python scripts/eval_quantization.py triposr --binary target/release/lux3d-cli.exe `
  --reference <canonical-triposr-dir> --variant q4_k=<dir-q4_k> `
  --source <input-image> --device cuda --output-dir out/eval-triposr `
  --mc-resolution 256 --mc-threshold 25.0
```

TripoSG mesh drift (2 steps, 1024 tokens, depth 5; reference is the same
backend's F32 mesh):

| variant | backend | CD rel | F@1% | voxel IoU | VRAM (delta) | wall |
|---|---|---|---|---|---|---|
| q8_0 | CUDA | 0.026% | 0.996 | 0.994 | 3.1 GiB | 10.4 s |
| q8_0 | Vulkan | 0.184% | 0.978 | 0.970 | 3.9 GiB | 13.3 s |
| q6_k | CUDA | 0.135% | 0.988 | 0.988 | — | — |
| q6_k | Vulkan | 0.134% | 0.995 | 0.995 | — | — |
| q5_k | CUDA | 0.155% | 0.993 | 0.994 | — | — |
| q5_k | Vulkan | 0.224% | 0.980 | 0.984 | — | — |
| q4_k | CUDA | 0.253% | 0.973 | 0.953 | 2.2 GiB | 10.0 s |
| q4_k | Vulkan | 0.180% | 0.982 | 0.972 | 2.3 GiB | 12.7 s |
| F32 | CUDA | — | — | — | 8.6 GiB | 14.9 s |
| F32 | Vulkan | — | — | — | 8.8 GiB | 17.1 s |

Two-step sampling is input-sensitive. On a deliberately harsh RGB input
(`mesh-input-rgb.png`, exercising the RMBG path) Q5_K and Q4_0 drifted to
~1.4% Chamfer while Q4_K stayed at 0.23% and Q8_0 at 0.06%. The outliers are
repeatable on that input but do not reproduce on the alpha input above, so
treat Q5_K/Q4_0 as input-dependent and prefer Q8_0/Q6_K/Q4_K for deployment.

VGGT (3 frames; reference is the same backend's F32 tensors; pose AUC on the
2 relative frame pairs):

| variant | backend | pointmap PSNR | depth PSNR | depth AbsRel | CD rel | pose AUC@5 | VRAM (delta) |
|---|---|---|---|---|---|---|---|
| q8_0 | CUDA | 53.1 dB | 48.1 dB | 0.0038 | 0.274% | 1.0 | 4.7 GiB |
| q8_0 | Vulkan | 55.9 dB | 53.5 dB | 0.0020 | 0.182% | 1.0 | 4.4 GiB |
| q8_0 | WGPU | 55.4 dB | 51.0 dB | 0.0027 | 0.193% | 1.0 | — |
| q6_k | CUDA | 45.7 dB | 44.8 dB | 0.0053 | 0.571% | 1.0 | — |
| q6_k | Vulkan | 46.5 dB | 49.5 dB | 0.0026 | 0.413% | 0.5 | — |
| q5_k | CUDA | 34.7 dB | 35.9 dB | 0.0144 | 1.54% | 0.0 | — |
| q5_k | Vulkan | 36.4 dB | 35.0 dB | 0.0180 | 1.39% | 1.0 | — |
| q4_k | CUDA | 24.7 dB | 23.5 dB | 0.0683 | 3.50% | 0.0 | 4.2 GiB |
| q4_k | Vulkan | 26.9 dB | 25.8 dB | 0.0511 | 2.84% | 0.0 | 3.7 GiB |
| F32 | CUDA | — | — | — | — | — | 6.7 GiB |
| F32 | Vulkan | — | — | — | — | — | 7.3 GiB |

Pi3/Pi3X point clouds and TripoSR mesh (CUDA; reference is the same backend's
F32 run; Pi3/Pi3X 3 frames, TripoSR `mesh-input.png` at 256):

| family | variant | CD rel | F@0.5% | F@1% | voxel IoU | normal |
|---|---|---|---|---|---|---|
| Pi3 | q8_0 | 0.166% | 0.990 | 0.99995 | — | — |
| Pi3 | q6_k | 0.139% | 0.985 | 0.99907 | — | — |
| Pi3 | q5_k | 0.851% | 0.469 | 0.706 | — | — |
| Pi3 | q4_k | 0.813% | 0.475 | 0.696 | — | — |
| Pi3X | q8_0 | 0.081% | 0.99999 | 1.000 | — | — |
| Pi3X | q6_k | 0.873% | 0.129 | 0.598 | — | — |
| Pi3X | q5_k | 2.279% | 0.000 | 0.000 | — | — |
| Pi3X | q4_k | 0.789% | 0.274 | 0.719 | — | — |
| TripoSR | q8_0 | 0.036% | 0.9997 | 0.99994 | 0.920 | 0.993 |
| TripoSR | q6_k | 0.050% | 0.9998 | 0.99995 | 0.886 | 0.991 |
| TripoSR | q5_k | 0.058% | 0.9996 | 0.99985 | 0.863 | 0.990 |
| TripoSR | q4_k | 0.129% | 0.9964 | 0.99985 | 0.682 | 0.978 |

Q8_0 is stable across backends: Pi3 F@1% is 0.99995 on CUDA and 0.99993 on
WGPU; Pi3X is 1.000 on CUDA/Vulkan/WGPU and 1.000 on CPU; TripoSR is 0.99994
on CUDA and 0.99996 on WGPU.

Recommendations: Pi3 is safe at Q8_0/Q6_K; Q5_K/Q4_K lose ~30% F-score, so keep
them out of production. Pi3X needs Q8_0 — Q6_K/Q5_K/Q4_K collapse despite the
usual monotonic storage error. The image `encoder.` is the sensitive module:
keeping it F16 restores Q6_K F@1% from 0.598 to 0.99996, but the resulting
package is 1.40 GiB versus 1.37 GiB for all-Q8_0, so Q8_0 is strictly better.
TripoSR is the most quantization-tolerant family here: Q8_0 through Q4_K keep
F@1% ≥ 0.9998. Q4_K still loses surface topology (voxel IoU 0.68 versus 0.92
at Q8_0), so prefer Q6_K/Q8_0 when topology fidelity matters.

TripoSG and VGGT: TripoSG stays within ~0.25% Chamfer even at Q4_K, so
Q4_K/Q5_K are practical there. VGGT is pose-sensitive: Q8_0 is near-lossless,
Q6_K is an acceptable fallback, Q5_K is marginal, and Q4_K is not viable
(pointmap and camera pose collapse), matching the published finding that naive
4-bit quantization destroys VGGT-style reconstruction. Use `--dump-npz` and the
script above on your own validation set before shipping a lower-bit variant.
RMBG-1.4 is a small matting network and is kept as a separate F16 GGUF, which
halves its footprint with no measurable mask drift.

### Run Inference

GPU backends are Cargo features (`cuda` / `vulkan` / `wgpu` / `metal`). Default builds are CPU-only.

```powershell
# Pi3 -> PLY with an explicit canonical package directory (layout matches 3d/canonical-weights/pi3)
cargo run -p lux3d-cli --features cuda -- run pi3 --model-path <canonical-pi3-dir> --source <input-sequence> --output <output-file.ply>

# Pi3X core -> PLY with an explicit canonical package directory (layout matches 3d/canonical-weights/pi3x)
cargo run -p lux3d-cli --features cuda -- run pi3x --model-path <canonical-pi3x-dir> --source <input-sequence> --conditions <conditions-file> --output <output-file.ply>

# Pi3X VO -> PLY with an explicit canonical package directory (layout matches 3d/canonical-weights/pi3x)
cargo run -p lux3d-cli --features cuda -- run pi3x --model-path <canonical-pi3x-dir> --source <input-video> --vo --chunk-size 8 --overlap 4 --conf-threshold 0.05 --inject-condition pose,depth,ray --output <output-file.ply>

# TripoSR -> GLB with an explicit canonical package directory (layout matches 3d/canonical-weights/triposr)
cargo run -p lux3d-cli --features cuda -- run triposr --model-path <canonical-triposr-dir> --source <input-image> --mc-resolution 256 --mc-threshold 25.0 --output <output-file.glb>

# VGGT -> PLY (world points filtered by confidence > 1.0 by default)
cargo run -p lux3d-cli -- run vggt --model-path <canonical-vggt-dir> --source <input-sequence> --output <output-file.ply>

# VGGT canonical package auto-download (oxide-lab/VGGT-1B)
cargo run -p lux3d-cli -- run vggt --source <input-sequence> --output <output-file.ply>

# TripoSG -> GLB (defaults: 50 rectified-flow steps, CFG 7.0, seed 42, flash decoder depth 9)
cargo run -p lux3d-cli --features cuda -- run triposg --rmbg-model-path <canonical-rmbg-dir> --model-path <canonical-triposg-dir> --source <input-image> --output <output-file.glb>
# Flash decoder is the reference default; disable it for dense/hierarchical marching cubes:
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --flash-decoder false --dense-octree-depth 8 --hierarchical-octree-depth 9 --output <output-file.glb>
# Faster preview: 2 steps, flash depth 5
cargo run -p lux3d-cli --features cuda -- run triposg --model-path <canonical-triposg-dir> --source <input-image> --steps 2 --flash-decoder true --flash-octree-depth 5 --output <preview.glb>

# TripoSG canonical package auto-download (oxide-lab/TripoSG); RMBG 1.4 is a separate package
cargo run -p lux3d-cli --features cuda -- run triposg --source <input-image> --output <output-file.glb>

# Auto-download canonical packages from Hugging Face into the user cache
cargo run -p lux3d-cli --features cuda -- run pi3 --source <input-sequence> --output <output-file.ply>
```

**Output formats:** mesh commands (`triposr`, `triposg`) select the writer from the output extension: `.glb` (primary/default), `.obj`, `.ply`, `.stl`. Point-cloud commands (`pi3`, `pi3x`, `vggt`) use `.ply` (primary) or `.xyz`. The HTTP mesh API defaults to GLB and accepts `options.output_format`; the point-cloud API defaults to PLY and accepts `options.output_format` (`ply`/`xyz`).

> **Experimental backends (Vulkan / WGPU):** select a non-CUDA device with `--device`:
>
> ```powershell
> cargo run -p lux3d-cli --features vulkan -- run triposr --device vulkan --source <input-image> --output <output-file.glb>
> cargo run -p lux3d-cli --features wgpu -- run pi3 --device wgpu --source <input-sequence> --output <output-file.ply>
> cargo run -p lux3d-cli --features vulkan -- run triposg --device vulkan --source <input-image> --steps 2 --num-tokens 1024 --flash-octree-depth 5 --output <preview.glb>
> ```
>
> These paths are experimental and not verified in CI. Inference may be unstable, produce incorrect output, or fail entirely depending on your GPU, driver, and OS. Use CUDA for production workloads.

### Run the HTTP Server

```powershell
cargo run -p lux3d-server --features cuda -- --host 127.0.0.1 --port 8080
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models
```

Prebuilt releases (`.github/workflows/release.yml`): Windows + Ubuntu × cpu/cuda/vulkan.
Names follow llama.cpp (`lux3d-<tag>-bin-…`); CUDA libs ship inside the archive (mistral.rs style).

Generation endpoints accept `multipart/form-data` and return `202 Accepted` with an async job object:

| Endpoint | Models | Output |
|----------|--------|--------|
| `POST /v1/point-clouds/generations` | `pi3`, `pi3x`, `vggt` | PLY |
| `POST /v1/meshes/generations` | `triposr`, `triposg` | GLB |

Common fields:

- `model` — required for point clouds (`pi3`, `pi3x`, or `vggt`), optional for meshes (`triposr` default; set `triposg` for TripoSG)
- `source` — required upload (`.zip` frame archive, `.mp4` video, or image)
- `conditions` — optional Pi3X conditions file
- `options` — optional JSON string with family-specific parameters (TripoSG: `steps`, `guidance_scale`, `seed`, `num_tokens`, `use_flash_decoder`, `flash_octree_depth`, `dense_octree_depth`, `hierarchical_octree_depth`, `query_chunk_size`, `output_format` (`glb` default, `obj`, `ply`, `stl`); point clouds: `output_format` (`ply` default, `xyz`), plus `interval`, `vo`, `chunk_size`, `overlap`, `conf_threshold`, `inject_condition`)

Workflow:

```powershell
# 1. Create a mesh generation job
curl -X POST http://127.0.0.1:8080/v1/meshes/generations `
  -F "model=triposr" `
  -F "source=@input.png" `
  -F "options={\"mc_resolution\":256,\"mc_threshold\":25.0}"

# TripoSG job (completed downloads as triposg.glb)
curl -X POST http://127.0.0.1:8080/v1/meshes/generations `
  -F "model=triposg" `
  -F "source=@input.png" `
  -F "options={\"steps\":50,\"use_flash_decoder\":true,\"flash_octree_depth\":9}"

# 2. Poll job status
curl http://127.0.0.1:8080/v1/generations/gen_...

# 3. Download completed asset
curl http://127.0.0.1:8080/v1/generations/gen_.../content --output mesh.glb
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

## Performance

Measured inference time (`[stage] infer`: preprocessing + neural pass, single iteration per process, weights excluded) on an NVIDIA GeForce RTX 3060 12 GiB, Windows (WDDM), candle `wgpu/vulkan` fork rev `efb506c9`, candle-3d (2026-09-06):

| model | CUDA | Vulkan | WGPU | Vulkan/CUDA | WGPU/CUDA |
|---|---:|---:|---:|---:|---:|
| pi3 (5 frames, 518×518) | **4.82 s** | 7.83 s | 13.96 s | 1.62× | 2.90× |
| pi3x (6 frames, 518×518) | **7.00 s** | 11.10 s | 31.84 s | 1.59× | 4.55× |
| triposr (single image) | **1.23 s** | 3.36 s | 4.89 s | 2.73× | 3.98× |

VGGT-1B was added later and is measured separately (fork rev `f94d26eb`, 2026-11-11): 3 frames at
518×518, including camera, DPT depth, and DPT point heads. RTX 3060 12 GiB: CUDA **6.76 s**,
Vulkan 10.96 s (1.62×), WGPU 19.34 s, CPU 107.66 s. The CUDA peak is 6.52 GiB at 3 frames
(Vulkan 6.1–7.4 GiB and WGPU 6.8 GiB across repeated runs); on an NVIDIA L4 23 GiB
the peak is 7.98 GiB at 3 frames (4.19 s infer) and 8.30 GiB at 12 frames (26.21 s infer).

TripoSG is measured per stage on an RTX 3060 12 GiB (Windows/WDDM, release build, single image
`test-assets/mesh-input.png`):

| Stage | CUDA | Vulkan | WGPU |
|---|---:|---:|---:|
| load | 7.12 s | 6.68 s | 6.60 s |
| infer (preprocess + DINO + denoise) | 6.84 s | 8.56 s | 13.13 s |
| mesh (flash SDF + DiffDMC + trimesh processing) | 0.73 s | 0.26 s | 0.45 s |
| export | 0.00 s | 0.00 s | 0.01 s |
| peak VRAM | 8.29 GiB | 8.36 GiB | 8.65 GiB |

All three columns use the same command (`--steps 2 --num-tokens 1024 --flash-octree-depth 5`).
TripoSG keeps all three submodels (DINOv2, DiT, VAE) resident — 7.4 GiB of weights — so memory is
dominated by the checkpoint. After the memory-default fixes (Vulkan transient pool 0 / inflight
grace 2, WGPU 2 MiB allocator blocks) Vulkan peaks within 0.1 GiB of CUDA and WGPU within ~0.4 GiB
(occasional ±0.5 GiB run-to-run variance from async staging).
12 GiB is enough for the 2048-token preview (Vulkan: load 6.56 s, infer 12.86 s, peak 9.13 GiB),
but 16 GiB is still recommended for the 50-step / depth-9 reference run.

Warm-loop criterion (10 samples, steady state — the candle-bench protocol, one back-to-back window): vulkan 3.14 s / 4.90 s per iter (1.16×/1.20× of CUDA 2.72 s / 4.09 s), wgpu 8.21 s / 24.0 s (3.0×/5.9×). The biggest single lever was a fused LayerNorm+RoPE kernel (`rope_layernorm.comp`/`.wgsl` in the fork): one dispatch per q/k replaces the slow-path LayerNorm, both surrounding copies, and the 8-op rope chain at every rope site, cutting wgpu pi3 from 54.9 s to 8.21 s warm.

Correctness: outputs of every backend × model combination were verified against the CUDA reference meshes (bounding box / center / mean delta within 1% of max extent, vertex/face counts within 0.5% — all PASS); the fused kernel is additionally unit-tested exact to 0.0 diff against the composed form on both backends.

Memory behavior under repeated inference (10-iteration loop, same process):

- **CUDA** — flat; the decoder attention scores buffer is bounded to 768 MiB by query-axis chunking (`LUX3D_MAX_SDPA_SCORES_BYTES`), which removes a ~4 GiB transient that previously fragmented the CUDA memory pool and stalled a 12 GiB card ~100×. VGGT peaks at 6.5 GiB for 3 frames on the 12 GiB card, with DPT frames processed one at a time (`LUX3D_VGGT_DPT_CHUNK`, default 1).
- **Vulkan** — flat; retention is bounded by the inflight byte budget (256 MiB × grace band 2) and transients are not pooled by default (`CANDLE_VK_INFLIGHT_GRACE`, `CANDLE_VK_POOL_MAX_BYTES=0`); the pool can be re-enabled for latency-first runs. Batch caps are tuned from flush-reason profiling: transfer bytes 512 MiB (`CANDLE_VK_MAX_BATCH_TRANSFER_BYTES`), descriptor sets 8× dispatches — closing the batch per big activation copy cost ~4-9 ms of WDDM fence-signal latency per submission.
- **WGPU** — 10-iteration pi3x benchmark completes with a 10.4 GiB VRAM peak and zero errors (previously OOM'd by iteration ~3); free pool / recycle backlog / in-flight retention are each byte-capped (`CANDLE_WGPU_POOL_MAX_BYTES`, `CANDLE_WGPU_INFLIGHT_MAX_BYTES`) and 2 MiB allocator blocks remove most checkpoint slack (`CANDLE_WGPU_BLOCK_BYTES`).

Known gaps: the remaining Vulkan/CUDA warm gap (1.16–1.20×) is the structural WDDM submit tax (~4-9 ms per submission on a strictly dependent op chain) — a platform property code cannot reach; Linux without WDDM would close it. WGPU remains 3.0–5.9× behind warm (down from 13.9–19.5×), dominated by WGSL GEMM quality plus the same inter-pass WDDM gaps. A criterion harness for whole-model benches lives in `crates/lux3d-core/benches/` (run one `(backend, model)` pair per process: `LUX3D_BENCH_DEVICE=… LUX3D_BENCH_MODEL=… cargo bench -p lux3d-core --bench bench_main --features vulkan,wgpu`).

## System Requirements

- Rust 1.85 or newer
- Cargo with support for `edition = "2024"`
- Python 3.x for baseline tooling and normalization
- CUDA-capable NVIDIA hardware for verified runtime inference
- **Vulkan and WGPU backends** (via [FerrisMind/candle](https://github.com/FerrisMind/candle) `wgpu/vulkan` fork): verified for correctness and memory stability on RTX 3060 (see [Performance](#performance)); slower than CUDA, WGPU substantially so. TripoSG peaks within ~0.4 GiB of CUDA on both backends, so the 7.9 GB weight set leaves ~3 GiB of headroom on a 12 GiB card.
- Metal backend on macOS is not tested in this repository, but may be supported theoretically.
- Canonical model package directories supplied with `--model-path`, or downloadable from Hugging Face into the user cache
- TripoSG needs roughly 8.3–8.7 GiB at the 1024-token preview (9.1 GiB at 2048 tokens) on all three backends; 16 GiB+ is recommended for the 50-step reference run.

## License

The code and documentation in this repository are licensed under Apache 2.0. See [LICENSE](https://github.com/oxide-lab/Lux3d/blob/main/LICENSE).

Upstream model assets and canonicalized model artifacts keep their original licenses and usage restrictions. Review model-family-specific terms before redistribution.

**VGGT-1B:** the original code and `facebook/VGGT-1B` checkpoint are CC-BY-NC-4.0 (non-commercial). Commercial use requires the separately distributed `facebook/VGGT-1B-Commercial` checkpoint. The canonical package used by `--model-path`/auto-download is published as `oxide-lab/VGGT-1B`; the raw upstream checkpoint stays at `facebook/VGGT-1B`. The native track head is intentionally not implemented in this inference slice; the contract records that exclusion.

**TripoSG:** the upstream code and weights are MIT licensed. The raw checkpoint lives at `VAST-AI/TripoSG`; the canonical package used by `--model-path`/auto-download is published as `oxide-lab/TripoSG`. Upstream `load_image` uses BriaRMBG 1.4 when a valid alpha channel is missing; Lux3D implements that fallback fully. The RMBG weights do not join the MIT TripoSG package: they are normalized into the separate `rmbg-1.4` canonical package under the BRIA source-available/non-commercial license (commercial use requires a BRIA license), resolved via `--rmbg-model-path` or a sibling `rmbg-1.4` directory. Upstream's pipeline defaults to the flash decoder (`use_flash_decoder=True`, `flash_octree_depth=9`); Lux3D implements the same flash SDF schedule plus a clean-room Rust DiffDMC. Diso (CC BY-NC 4.0) source and tables are not copied, so ambiguous checkerboard cells may resolve differently; the non-flash dense/hierarchical path remains available via `--flash-decoder false`. The full VAE encode API (FPS sampling, `vae.quant`, mean/logvar/std distribution) is also implemented even though the image-to-3D pipeline only calls `vae.decode`.
