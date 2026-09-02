from huggingface_hub import hf_hub_download, list_repo_files
from pathlib import Path

base = Path(r"G:\candle-3d\models")
repos = [
    ("oxide-lab/Pi3", "pi3"),
    ("oxide-lab/Pi3X", "pi3x"),
    ("oxide-lab/TripoSR", "triposr"),
]
required = [
    "model.safetensors",
    "resolved_config.json",
    "manifest.json",
    "checksums.json",
]

for repo_id, name in repos:
    dest = base / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"=== {repo_id} -> {dest} ===")
    try:
        files = list_repo_files(repo_id)
        print(f"repo has {len(files)} files")
    except Exception as error:
        print(f"list_repo_files error: {error}")
    for filename in required:
        target = dest / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"  skip existing {filename} ({target.stat().st_size} bytes)")
            continue
        print(f"  downloading {filename}...")
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest))
        print(f"  -> {path}")

print("done")
