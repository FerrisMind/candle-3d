from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "-q"])
    from PIL import Image, ImageDraw

assets = Path(r"G:\candle-3d\test-assets")
assets.mkdir(parents=True, exist_ok=True)

mesh_image = assets / "mesh-input.png"
img = Image.new("RGB", (512, 512), color=(240, 240, 245))
draw = ImageDraw.Draw(img)
draw.rectangle((96, 96, 416, 416), fill=(120, 160, 220), outline=(40, 60, 120), width=8)
draw.ellipse((176, 120, 336, 220), fill=(220, 180, 120))
img.save(mesh_image)
print(f"created {mesh_image}")

frames_dir = assets / "pi3-frames"
frames_dir.mkdir(exist_ok=True)
for index in range(3):
    frame = Image.new("RGB", (512, 512), color=(230, 230, 235))
    frame_draw = ImageDraw.Draw(frame)
    offset = index * 24
    frame_draw.rectangle(
        (80 + offset, 120, 420 + offset, 420),
        fill=(100 + index * 20, 140, 200),
        outline=(30, 50, 90),
        width=6,
    )
    path = frames_dir / f"{index:06d}.png"
    frame.save(path)
    print(f"created {path}")

print("test assets ready")
