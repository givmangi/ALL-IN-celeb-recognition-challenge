"""
Competition-day script: apply GFPGAN restoration to gallery images.

Setup (run once before competition day):
    pip install basicsr facexlib gfpgan insightface==0.2.1 onnxruntime

Download pretrained model (run once):
    wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth
    mkdir -p experiments/pretrained_models
    mv GFPGANCleanv1-NoCE-C2.pth experiments/pretrained_models/

Usage:
    python gfpgan_augment.py --input gallery/ --output gallery_hq/
"""

import os
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--input',  type=str, required=True,
                    help='Input folder of gallery images')
parser.add_argument('--output', type=str, required=True,
                    help='Output folder for restored images')
parser.add_argument('--model_path', type=str,
                    default='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth',
                    help='Path to GFPGAN pretrained weights')
parser.add_argument('--upscale', type=int, default=2,
                    help='Upscale factor (default: 2)')
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# ── Load GFPGAN ───────────────────────────────────────────────────────────────
from gfpgan import GFPGANer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

restorer = GFPGANer(
    model_path=args.model_path,
    upscale=args.upscale,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None   # no background upsampling — faster
)

# ── Process images ────────────────────────────────────────────────────────────
exts = ('.jpg', '.jpeg', '.png')
filenames = [f for f in os.listdir(args.input) if f.lower().endswith(exts)]
print(f"Found {len(filenames)} images to process")

for fname in tqdm(filenames, desc="Restoring"):
    img_path = os.path.join(args.input, fname)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"  Warning: could not read {fname}, skipping")
        continue

    try:
        # GFPGAN returns: cropped_faces, restored_faces, restored_img
        _, restored_faces, restored_img = restorer.enhance(
            img,
            has_aligned=False,      # let GFPGAN detect and align faces
            only_center_face=True,  # process only the largest/center face
            paste_back=True         # paste restored face back into original
        )

        out_path = os.path.join(args.output, fname)
        cv2.imwrite(out_path, restored_img)

    except Exception as e:
        print(f"  Warning: failed on {fname}: {e}, copying original")
        import shutil
        shutil.copy(img_path, os.path.join(args.output, fname))

print(f"\nDone. Restored images saved to: {args.output}")
print("You can now run the retrieval pipeline with gallery_folder = args.output")
