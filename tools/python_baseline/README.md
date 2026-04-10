# Golden Python Baseline Tooling

These scripts capture source-truth baseline artifacts for Pi3, Pi3X, Pi3X VO, and TripoSR without modifying the vendor trees in [`H:\GitHub\LuxRT\tp\3d\Pi3`](H:\GitHub\LuxRT\tp\3d\Pi3) and [`H:\GitHub\LuxRT\tp\3d\TripoSR`](H:\GitHub\LuxRT\tp\3d\TripoSR).

## Environments

Use separate Python environments per family.

### Pi3

```powershell
python -m venv --system-site-packages H:\GitHub\LuxRT\3d\_generated\python-envs\pi3
H:\GitHub\LuxRT\3d\_generated\python-envs\pi3\Scripts\python.exe -m pip install --upgrade pip
H:\GitHub\LuxRT\3d\_generated\python-envs\pi3\Scripts\python.exe -m pip install -r H:\GitHub\LuxRT\tp\3d\Pi3\requirements.txt
```

### Pi3X

```powershell
python -m venv --system-site-packages H:\GitHub\LuxRT\3d\_generated\python-envs\pi3x
H:\GitHub\LuxRT\3d\_generated\python-envs\pi3x\Scripts\python.exe -m pip install --upgrade pip
H:\GitHub\LuxRT\3d\_generated\python-envs\pi3x\Scripts\python.exe -m pip install -r H:\GitHub\LuxRT\tp\3d\Pi3\requirements.txt safetensors plyfile
```

### TripoSR

Install PyTorch first according to the official README for your local CUDA/CPU setup, then:

```powershell
python -m venv --system-site-packages H:\GitHub\LuxRT\3d\_generated\python-envs\triposr
H:\GitHub\LuxRT\3d\_generated\python-envs\triposr\Scripts\python.exe -m pip install --upgrade pip setuptools
H:\GitHub\LuxRT\3d\_generated\python-envs\triposr\Scripts\python.exe -m pip install -r H:\GitHub\LuxRT\tp\3d\TripoSR\requirements.txt
H:\GitHub\LuxRT\3d\_generated\python-envs\triposr\Scripts\python.exe -m pip install safetensors
```

## Commands

### Pi3

Golden:

```powershell
H:\GitHub\LuxRT\3d\_generated\python-envs\pi3\Scripts\python.exe H:\GitHub\LuxRT\3d-rs\tools\python_baseline\pi3_dump.py --repo-root H:\GitHub\LuxRT --sample-id house-golden --sample-kind golden --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\house
```

Smoke:

```powershell
H:\GitHub\LuxRT\3d\_generated\python-envs\pi3\Scripts\python.exe H:\GitHub\LuxRT\3d-rs\tools\python_baseline\pi3_dump.py --repo-root H:\GitHub\LuxRT --sample-id skating-smoke --sample-kind smoke --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\skating.mp4
```

### Pi3X

Golden core:

```powershell
python H:\GitHub\LuxRT\3d-rs\tools\python_baseline\pi3x_dump.py --repo-root H:\GitHub\LuxRT --sample-id room-golden --sample-kind golden --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\rgb --conditions H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\condition.npz --device auto
```

Golden VO:

```powershell
python H:\GitHub\LuxRT\3d-rs\tools\python_baseline\pi3x_vo_dump.py --repo-root H:\GitHub\LuxRT --sample-id skating-vo-golden --sample-kind golden --source H:\GitHub\LuxRT\tp\3d\Pi3\examples\skating.mp4 --chunk-size 8 --overlap 4 --device auto
```

### TripoSR

Golden:

```powershell
H:\GitHub\LuxRT\3d\_generated\python-envs\triposr\Scripts\python.exe H:\GitHub\LuxRT\3d-rs\tools\python_baseline\triposr_dump.py --repo-root H:\GitHub\LuxRT --sample-id horse-golden --sample-kind golden --source H:\GitHub\LuxRT\tp\3d\TripoSR\examples\horse.png
```

Smoke:

```powershell
H:\GitHub\LuxRT\3d\_generated\python-envs\triposr\Scripts\python.exe H:\GitHub\LuxRT\3d-rs\tools\python_baseline\triposr_dump.py --repo-root H:\GitHub\LuxRT --sample-id chair-smoke --sample-kind smoke --source H:\GitHub\LuxRT\tp\3d\TripoSR\examples\chair.png
```

## Output Layout

- Light repo-resident artifacts: `H:\GitHub\LuxRT\3d\baselines\<family>\<sample-id>\`
- Heavy local cache: `H:\GitHub\LuxRT\3d\_generated\python-baseline\<family>\<sample-id>\`

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
