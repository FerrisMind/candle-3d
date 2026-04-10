# Golden Python Baseline Tooling

These scripts capture source-truth baseline artifacts for Pi3, Pi3X, Pi3X VO, and TripoSR without modifying your external vendor or model sources.

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

## Commands

### Pi3

Golden:

```powershell
<pi3-env>/Scripts/python.exe tools/python_baseline/pi3_dump.py --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --sample-id house-golden --sample-kind golden --source <pi3-input>
```

Smoke:

```powershell
<pi3-env>/Scripts/python.exe tools/python_baseline/pi3_dump.py --repo-root <runtime-root> --raw-model-dir <raw-pi3-dir> --sample-id skating-smoke --sample-kind smoke --source <pi3-video>
```

### Pi3X

Golden core:

```powershell
python tools/python_baseline/pi3x_dump.py --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --sample-id room-golden --sample-kind golden --source <pi3x-input> --conditions <pi3x-conditions> --device auto
```

Golden VO:

```powershell
python tools/python_baseline/pi3x_vo_dump.py --repo-root <runtime-root> --raw-model-dir <raw-pi3x-dir> --sample-id skating-vo-golden --sample-kind golden --source <pi3x-video> --chunk-size 8 --overlap 4 --device auto
```

### TripoSR

Golden:

```powershell
<triposr-env>/Scripts/python.exe tools/python_baseline/triposr_dump.py --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --sample-id horse-golden --sample-kind golden --source <triposr-input>
```

Smoke:

```powershell
<triposr-env>/Scripts/python.exe tools/python_baseline/triposr_dump.py --repo-root <runtime-root> --raw-model-dir <raw-triposr-dir> --sample-id chair-smoke --sample-kind smoke --source <triposr-smoke-input>
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
