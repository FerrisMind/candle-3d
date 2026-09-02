from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

assets = Path(r"G:\candle-3d\test-assets")
assets.mkdir(parents=True, exist_ok=True)

# TripoSR sample image from the upstream repo if available
triposr_candidates = [
    ("oxide-lab/TripoSR", "examples/chair.png"),
    ("oxide-lab/TripoSR", "examples/input.png"),
    ("stabilityai/TripoSR", "figures/teaser.png"),
]

mesh_image = assets / "mesh-input.png"
for repo_id, filename in triposr_candidates:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(path, mesh_image)
        print(f"mesh input: {mesh_image} from {repo_id}/{filename}")
        break
    except Exception as error:
        print(f"skip {repo_id}/{filename}: {error}")

# Pi3 room rgb frames if published
pi3_frame_dir = assets / "pi3-frames"
pi3_frame_dir.mkdir(exist_ok=True)
frame_candidates = [
    ("oxide-lab/Pi3", "examples/room/rgb/000000.png"),
    ("oxide-lab/Pi3", "examples/room/rgb/000001.png"),
    ("oxide-lab/Pi3", "examples/room/rgb/000002.png"),
]
downloaded = 0
for repo_id, filename in frame_candidates:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        target = pi3_frame_dir / Path(filename).name
        shutil.copy2(path, target)
        downloaded += 1
        print(f"frame: {target}")
    except Exception as error:
        print(f"skip {repo_id}/{filename}: {error}")

conditions = assets / "pi3x-conditions.npz"
for repo_id, filename in [
    ("oxide-lab/Pi3X", "examples/room/condition.npz"),
    ("oxide-lab/Pi3", "examples/room/condition.npz"),
]:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.copy2(path, conditions)
        print(f"conditions: {conditions}")
        break
    except Exception as error:
        print(f"skip {repo_id}/{filename}: {error}")

if downloaded == 0 and mesh_image.exists():
    # fallback: duplicate mesh image as a single-frame source for pi3/pi3x smoke tests
    fallback = pi3_frame_dir / "000000.png"
    shutil.copy2(mesh_image, fallback)
    print(f"fallback pi3 frame: {fallback}")

print("assets ready")
