# Golden Python Baseline Tooling (Lux3D / candle-3d)

These scripts capture source-truth baseline artifacts for Pi3, Pi3X, Pi3X VO, TripoSR, and TripoSG without modifying your external vendor or model sources.

Note: the macOS Metal backend is not tested in this repository, but may be supported theoretically.

## Environments

Use separate Python environments per family.

### Pi3

```powershell
python -m venv --system-site-packages <pi3-env>
<pi3-env>/Scripts/python.exe -m pip install --upgrade pip
<pi3-env>/Scripts/python.exe -m pip install -r <pi3-requirements>
```

### Pi3X

```powershell
python -m venv --system-site-packages <pi3x-env>
<pi3x-env>/Scripts/python.exe -m pip install --upgrade pip
<pi3x-env>/Scripts/python.exe -m pip install -r <pi3x-requirements> safetensors plyfile
```

### TripoSR

Install PyTorch first according to the official upstream setup for your CUDA or CPU environment, then:

```powershell
python -m venv --system-site-packages <triposr-env>
<triposr-env>/Scripts/python.exe -m pip install --upgrade pip setuptools
<triposr-env>/Scripts/python.exe -m pip install -r <triposr-requirements>
<triposr-env>/Scripts/python.exe -m pip install safetensors
```

### TripoSG

Uses the system Python 3.11 environment (torch 2.5.1+cu124, transformers 5.17, diffusers 0.33). The reference checkout (`candle-3d-refs/TripoSG`) is put on `sys.path` by the script itself, so `python -I` works; set the `TRIPOSG_REF_REPO` environment variable to override the reference location. Install the missing packages into the system Python:

```powershell
python -m pip install trimesh scipy scikit-image jaxtyping typeguard
```

The main `triposg_dump.py` harness is deliberately hermetic: the input image must be RGBA with a valid alpha channel (the reference `load_image` alpha branch is used; the rembg/BriaRMBG fallback is forbidden and hard-errors), and `diso`/`DiffDMC` is stubbed (geometry comes from the reference hierarchical skimage marching-cubes path). RMBG has its own parity script, `rmbg_dump.py`, and the flash/DiffDMC path has `triposg_flash_dump.py` (requires CUDA and a locally built Diso extension).

Smoke:

```powershell
python -I <repo-root>/tools/python_baseline/triposg_dump.py --model-dir <triposg-weights-dir> --image <rgba-input> --output-dir <runtime-root>/3d/_generated/python-baseline/triposg/<sample-id> --light-output-dir <runtime-root>/3d/baselines/triposg/<sample-id> --sample-id smoke --kind smoke --steps 2 --guidance-scale 7.0 --seed 42 --dense-octree-depth 5 --hierarchical-octree-depth 5 --device cuda --dtype float32
```

Golden: same command with `--kind golden`, more `--steps`, and deeper octree depths (for example `--dense-octree-depth 8 --hierarchical-octree-depth 9`).

```powershell
python -I <repo-root>/tools/python_baseline/triposg_dump.py --model-dir <triposg-weights-dir> --image <rgba-input> --output-dir <runtime-root>/3d/_generated/python-baseline/triposg/mesh-input-golden --light-output-dir <runtime-root>/3d/baselines/triposg/mesh-input-golden --sample-id mesh-input-golden --kind golden --steps 50 --guidance-scale 7.0 --seed 42 --dense-octree-depth 8 --hierarchical-octree-depth 9 --device cuda --dtype float32
```

The hermetic runner supplies `diso`/`DiffDMC` stubs and does not modify the reference checkout. Set `TRIPOSG_REF_REPO` when the reference lives outside `candle-3d-refs/TripoSG`; set `TRIPOSG_MODEL_DIR` to override the raw model directory. Golden runs also write the canonical `artifacts.safetensors` taps listed in the contract plus the reference `mesh.obj`.

## Commands

### Pi3

Golden:

```powershell
<pi3-env>/Scripts/python.exe <repo-root>/tools/python_baseline/pi3_dump.py --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --sample-id house-golden --sample-kind golden --source <pi3-input>
```

Smoke:

```powershell
<pi3-env>/Scripts/python.exe <repo-root>/tools/python_baseline/pi3_dump.py --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --sample-id skating-smoke --sample-kind smoke --source <pi3-video>
```

### Pi3X

Golden core:

```powershell
python <repo-root>/tools/python_baseline/pi3x_dump.py --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --sample-id room-golden --sample-kind golden --source <pi3x-input> --conditions <pi3x-conditions> --device auto
```

Golden VO:

```powershell
python <repo-root>/tools/python_baseline/pi3x_vo_dump.py --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --sample-id skating-vo-golden --sample-kind golden --source <pi3x-video> --chunk-size 8 --overlap 4 --device auto
```

### TripoSR

Golden:

```powershell
<triposr-env>/Scripts/python.exe <repo-root>/tools/python_baseline/triposr_dump.py --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --sample-id horse-golden --sample-kind golden --source <triposr-input>
```

Smoke:

```powershell
<triposr-env>/Scripts/python.exe <repo-root>/tools/python_baseline/triposr_dump.py --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --sample-id chair-smoke --sample-kind smoke --source <triposr-smoke-input>
```

## Output Layout

- Light repo-resident artifacts: repository baseline output directory
- Heavy local cache: external generated cache directory

Light artifacts:

- `manifest.json`
- `summary.json`
- `checksums.json`
- `input_preview.png`

Heavy artifacts for golden runs:

- `artifacts.safetensors`
- `point_cloud.ply` for Pi3
- `point_cloud.ply` for Pi3X / Pi3X VO
- `mesh.obj` for TripoSR
- `artifacts.safetensors` and `mesh.obj` for TripoSG (written for both smoke and golden runs; smoke manifests stay summary-only)
