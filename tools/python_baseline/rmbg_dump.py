#!/usr/bin/env python3
"""Dump RMBG-1.4 / TripoSG preprocessing taps for the Rust parity harness.

This script mirrors the exact reference path used by TripoSG when the input
image has no valid alpha channel:

  scripts/image_process.py::load_image
    -> torchvision Resize((1024, 1024), antialias=True)
    -> local rmbg() wrapper: TF.normalize(image, [0.5, 0.5, 0.5], [1, 1, 1])
    -> briaai BriaRMBG
    -> resize sigmoid mask back to the capped source size (antialias=True)
    -> min-max normalize, uint8, cv2 Otsu threshold
    -> skimage label + remove_small_objects(min_size=200)
    -> bounding box, white composite, 10% square padding

The final `rmbg.final` tensor is also cross-checked against the unmodified
reference `load_image` to guarantee that the tap reconstruction did not drift.

Usage (must run with CUDA because image_process.py hardcodes .cuda()):

  python -I tools/python_baseline/rmbg_dump.py \
      --image test-assets/mesh-input-rgb.png \
      --weights models/RMBG-1.4 \
      --triposg-model-dir models/TripoSG \
      --output ../3d/_generated/python-baseline/rmbg/smoke/artifacts.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support `python -I` (isolated mode), which does not put the script directory
# on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from skimage.measure import label
from skimage.morphology import remove_small_objects
from torchvision import transforms
import torchvision.transforms.functional as TF


REF_SCRIPTS = Path(
    r"G:/jobs/candle-based-projects/candle-3d-refs/TripoSG/scripts"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--triposg-model-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def tensor(name: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
    value = value.detach().to("cpu", dtype=torch.float32).contiguous()
    print(f"  {name}: {list(value.shape)}")
    return name, value


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("reference image_process.py hardcodes .cuda()")

    source_path = Path(args.image).resolve()
    weights_dir = Path(args.weights).resolve()
    triposg_dir = Path(args.triposg_model_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REF_SCRIPTS))
    from briarmbg import BriaRMBG  # type: ignore
    from image_process import load_image, prepare_image  # type: ignore

    from safetensors.torch import load_file

    net = BriaRMBG({"in_ch": 3, "out_ch": 1})
    missing, unexpected = net.load_state_dict(
        load_file(str(weights_dir / "model.safetensors")), strict=True
    )
    if missing or unexpected:
        raise RuntimeError(f"RMBG state dict mismatch: missing={missing}, unexpected={unexpected}")
    net = net.cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Decode exactly like image_process.py (BGR unchanged), then cap >2000 px.
    img = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed to decode {source_path}")
    if img.ndim == 2:
        channels = 1
    else:
        channels = img.shape[2]
    if channels != 3:
        raise RuntimeError(f"RMBG baseline requires an RGB source, got {channels} channels")
    height, width = img.shape[:2]
    scale = 2000.0 / max(height, width)
    if scale < 1:
        img = cv2.resize(
            img,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    if img.dtype != "uint8":
        img = (img * (255.0 / np.iinfo(img.dtype).max)).astype(np.uint8)

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_image_gpu = torch.from_numpy(rgb_image).cuda().float().permute(2, 0, 1) / 255.0

    taps: dict[str, torch.Tensor] = {}
    taps.update([tensor("rmbg.source_rgb", rgb_image_gpu)])

    resize_1024 = transforms.Resize((1024, 1024), antialias=True)
    rgb_image_resized = resize_1024(rgb_image_gpu)
    taps.update([tensor("rmbg.rgb_1024", rgb_image_resized)])

    # Reference local wrapper in image_process.py.
    model_input = TF.normalize(
        rgb_image_resized, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]
    ).unsqueeze(0)
    taps.update([tensor("rmbg.model_input", model_input)])

    with torch.no_grad():
        result = net(model_input)
    sigmoid = result[0][0]
    taps.update([tensor("rmbg.sigmoid", sigmoid)])

    alpha_gpu_rmbg = sigmoid.squeeze(0)
    resize_back = transforms.Resize(
        (rgb_image_gpu.shape[1], rgb_image_gpu.shape[2]), antialias=True
    )
    alpha_gpu_rmbg = resize_back(alpha_gpu_rmbg)
    taps.update([tensor("rmbg.alpha_resized", alpha_gpu_rmbg)])

    ma, mi = alpha_gpu_rmbg.max(), alpha_gpu_rmbg.min()
    alpha_gpu_rmbg = (alpha_gpu_rmbg - mi) / (ma - mi)
    taps.update([tensor("rmbg.alpha_minmax", alpha_gpu_rmbg)])

    alpha_gpu_tmp = alpha_gpu_rmbg * 255
    alpha_u8 = alpha_gpu_tmp.to(torch.uint8).squeeze().cpu().numpy()
    taps.update([tensor("rmbg.alpha_u8", torch.from_numpy(alpha_u8.astype(np.float32)))])

    _, alpha_otsu = cv2.threshold(
        alpha_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    taps.update([tensor("rmbg.alpha_otsu", torch.from_numpy(alpha_otsu.astype(np.float32)))])

    labeled_alpha = label(alpha_otsu)
    cleaned_alpha = remove_small_objects(labeled_alpha, min_size=200)
    cleaned_alpha = (cleaned_alpha > 0).astype(np.uint8)
    taps.update([tensor("rmbg.alpha_cleaned", torch.from_numpy(cleaned_alpha.astype(np.float32)))])

    alpha = cleaned_alpha * 255
    alpha_gpu = torch.from_numpy(cleaned_alpha).cuda().float().unsqueeze(0)
    taps.update([tensor("rmbg.alpha_final", alpha_gpu)])

    # find_bounding_box uses cv2 contours; for a binary mask this is the
    # inclusive bbox of foreground pixels. Dump it for cross-checking.
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    taps.update([tensor("rmbg.bbox", torch.tensor([x, y, w, h], dtype=torch.float32))])

    bg_gray = bg_color[0]
    bg_tensor = (
        torch.from_numpy(bg_color)
        .float()
        .cuda()
        .repeat(alpha_gpu.shape[1], alpha_gpu.shape[2], 1)
        .permute(2, 0, 1)
    )
    composited = rgb_image_gpu * alpha_gpu + bg_tensor * (1 - alpha_gpu)
    taps.update([tensor("rmbg.composited", composited)])

    padding_size = [0] * 6
    if w > h:
        padding_size[0] = int(w * 0.1)
        padding_size[2] = int(padding_size[0] + (w - h) / 2)
    else:
        padding_size[2] = int(h * 0.1)
        padding_size[0] = int(padding_size[2] + (h - w) / 2)
    padding_size[1] = padding_size[0]
    padding_size[3] = padding_size[2]
    final = F.pad(
        composited[:, y : (y + h), x : (x + w)],
        pad=tuple(padding_size),
        mode="constant",
        value=bg_gray,
    )
    taps.update([tensor("rmbg.final", final)])

    # Cross-check the reconstructed final tensor against the unmodified
    # reference function. This runs the network a second time.
    reference_final = load_image(
        str(source_path), bg_color=bg_color, rmbg_net=net, padding_ratio=0.1
    )
    if not torch.equal(final, reference_final):
        delta = (final - reference_final).abs().max().item()
        raise RuntimeError(
            f"tap reconstruction drifted from reference load_image (max delta {delta})"
        )
    print("reference load_image cross-check: exact")

    conditioning_pil = prepare_image(str(source_path), bg_color, rmbg_net=net)
    from transformers import BitImageProcessor

    processor = BitImageProcessor.from_pretrained(
        str(triposg_dir / "feature_extractor_dinov2")
    )
    raw_pixels = processor(
        conditioning_pil, return_tensors="pt", do_rescale=False, do_normalize=False
    ).pixel_values
    preprocessed_uint8 = raw_pixels.clamp(0.0, 255.0).round().to(torch.uint8)[0]
    taps.update([tensor("rmbg.preprocessed_uint8", preprocessed_uint8)])
    pixel_values = processor(conditioning_pil, return_tensors="pt").pixel_values
    taps.update([tensor("rmbg.pixel_values", pixel_values)])

    save_file(taps, str(output_path))
    print(f"wrote {output_path} ({output_path.stat().st_size} bytes, {len(taps)} tensors)")


if __name__ == "__main__":
    main()
