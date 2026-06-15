"""
Test script for the GFPGAN augmentation pipeline.

Tests:
  1. Model loads without errors
  2. A single image is processed correctly
  3. Output image has expected properties
  4. Batch processing works and preserves filenames
  5. Graceful fallback on unreadable images

Run on the Azure VM (GFPGAN is hard to install on Windows + Python 3.13):
    pip install gfpgan
    python test_gfpgan.py \
        --model_path experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth \
        --sample_image path/to/some_face.jpg \
        --extra_test_images path/to/folder_with_a_few_faces/

Note: GFPGAN internally uses BGR numpy arrays (OpenCV convention).
We use PIL to read/write files but convert to/from BGR numpy arrays
when interacting with GFPGAN.
"""

import os
import argparse
import shutil
import tempfile
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth')
parser.add_argument('--sample_image', type=str, required=True,
                    help='Path to a real face image used in Test 2')
parser.add_argument('--extra_test_images', type=str, default=None,
                    help='Optional folder of additional face images for Test 3 batch processing. '
                         'If not provided, Test 3 reuses the sample image 3 times.')
args = parser.parse_args()

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def print_result(name, passed, detail=""):
    status = PASS if passed else FAIL
    detail_str = f" — {detail}" if detail else ""
    print(f"  {status} {name}{detail_str}")


# ── PIL ↔ BGR-numpy helpers ───────────────────────────────────────────────────
def pil_to_bgr(pil_img):
    """Convert PIL Image (RGB) to BGR numpy array, the format GFPGAN expects."""
    rgb = np.array(pil_img.convert('RGB'))
    bgr = rgb[:, :, ::-1].copy()  # reverse last axis: RGB → BGR
    return bgr


def bgr_to_pil(bgr_array):
    """Convert BGR numpy array back to PIL Image (RGB)."""
    rgb = bgr_array[:, :, ::-1]  # BGR → RGB
    return Image.fromarray(rgb)


def safe_read_image(path):
    """Read an image file, returning a BGR numpy array or None if it fails."""
    try:
        pil_img = Image.open(path)
        pil_img.load()  # force read — raises if file is corrupt
        return pil_to_bgr(pil_img)
    except Exception:
        return None


def load_restorer(model_path):
    from gfpgan import GFPGANer
    return GFPGANer(
        model_path=model_path,
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None
    )


# ── Test 1: Model loading ─────────────────────────────────────────────────────
print("\nTest 1: Model loading")
try:
    restorer = load_restorer(args.model_path)
    print_result("GFPGANer loads successfully", True)
except FileNotFoundError:
    print_result("Model file found", False,
                 f"not found at {args.model_path} — download it first")
    print("  Skipping remaining tests.")
    exit(1)
except ImportError as e:
    print_result("gfpgan library installed", False, str(e))
    print("  Install with: pip install gfpgan")
    exit(1)
except Exception as e:
    print_result("GFPGANer loads successfully", False, str(e))
    exit(1)


# ── Test 2: Single image processing ──────────────────────────────────────────
print("\nTest 2: Single image processing")
if not os.path.exists(args.sample_image):
    print_result("Sample image exists", False, f"not found at {args.sample_image}")
    exit(1)

img = safe_read_image(args.sample_image)
print_result("Loaded sample image", img is not None,
             f"shape {img.shape}" if img is not None else "failed to load")

if img is None:
    exit(1)

try:
    _, restored_faces, restored_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=True,
        paste_back=True
    )
    print_result("enhance() runs without error", True)
except Exception as e:
    print_result("enhance() runs without error", False, str(e))
    exit(1)

# Check output properties
print_result("Output image is not None", restored_img is not None)

if restored_img is not None:
    h_in, w_in = img.shape[:2]
    h_out, w_out = restored_img.shape[:2]
    expected_h, expected_w = h_in * 2, w_in * 2  # upscale=2
    print_result(
        f"Output size is upscaled (expected {expected_w}×{expected_h})",
        h_out == expected_h and w_out == expected_w,
        f"got {w_out}×{h_out}"
    )
    print_result("Output has 3 channels", restored_img.shape[2] == 3,
                 f"got {restored_img.shape[2]}")
    print_result("Output dtype is uint8", restored_img.dtype == np.uint8,
                 f"got {restored_img.dtype}")
    print_result("Detected and restored at least one face",
                 restored_faces is not None and len(restored_faces) > 0,
                 f"{len(restored_faces) if restored_faces else 0} face(s) restored")


# ── Test 3: File I/O round-trip ───────────────────────────────────────────────
print("\nTest 3: File I/O round-trip")
with tempfile.TemporaryDirectory() as tmpdir:
    in_dir  = os.path.join(tmpdir, "input")
    out_dir = os.path.join(tmpdir, "output")
    os.makedirs(in_dir)
    os.makedirs(out_dir)

    # Gather test images: either from --extra_test_images folder or reuse sample
    test_sources = []
    if args.extra_test_images and os.path.isdir(args.extra_test_images):
        candidates = [
            f for f in sorted(os.listdir(args.extra_test_images))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ][:3]
        test_sources = [os.path.join(args.extra_test_images, f) for f in candidates]
    if not test_sources:
        # Fallback: copy the sample image three times under different names
        test_sources = [args.sample_image] * 3

    # Write test inputs with identity__name.jpg convention
    fnames = []
    for i, src in enumerate(test_sources):
        fname = f"identity_{chr(65+i)}__img{i:03d}.jpg"
        shutil.copy(src, os.path.join(in_dir, fname))
        fnames.append(fname)

    print_result(f"Prepared {len(fnames)} test images", True)

    # Run augmentation on all of them
    processed = 0
    for fname in fnames:
        in_path = os.path.join(in_dir, fname)
        img = safe_read_image(in_path)
        if img is None:
            print(f"    Warning: could not read {fname}")
            continue
        try:
            _, _, restored = restorer.enhance(img, has_aligned=False,
                                              only_center_face=True, paste_back=True)
            bgr_to_pil(restored).save(os.path.join(out_dir, fname))
            processed += 1
        except Exception as e:
            print(f"    Warning: {fname} failed: {e}")

    print_result(f"Processed {processed}/{len(fnames)} images", processed == len(fnames))

    # Verify output filenames match input
    out_files = sorted(os.listdir(out_dir))
    print_result("Output filenames match input filenames", sorted(fnames) == out_files,
                 f"got {out_files}")

    # Verify output files are valid images
    valid = all(
        safe_read_image(os.path.join(out_dir, f)) is not None
        for f in out_files
    )
    print_result("All output files are readable images", valid)


# ── Test 4: Fallback on corrupt image ─────────────────────────────────────────
print("\nTest 4: Graceful fallback on unreadable file")
with tempfile.TemporaryDirectory() as tmpdir:
    bad_path = os.path.join(tmpdir, "corrupt.jpg")
    with open(bad_path, 'w') as f:
        f.write("this is not an image")

    img = safe_read_image(bad_path)
    print_result("safe_read_image returns None for corrupt file", img is None)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\nAll tests completed.")
print("If Tests 1–3 passed, your augmentation pipeline is ready for competition day.")
