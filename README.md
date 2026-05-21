# Celebrity Retrieval Across Domains — Branch Documentation

This branch contains the data augmentation pipeline and several baseline retrieval scripts for the *Celebrity Retrieval Across Domains* competition. The central component is `gfpgan_augment.py`, which prepares the competition gallery images on competition day so they match the distribution our team's models were fine-tuned on.

---

## Task Recap

Given a real photograph of a celebrity (query), retrieve the 10 most similar AI-generated images of that same celebrity from a synthetic gallery. The challenge is the **domain shift** between real photos and synthetic images. See the course handout for full task and submission format.

---

## Scripts Overview

| Script | Purpose |
|---|---|
| `gfpgan_augment.py` | **Main augmentation script.** Applies GFPGAN face restoration to a folder of images. Used on competition day to bring competition data closer to our training distribution. |
| `test_gfpgan.py` | Sanity-check script for `gfpgan_augment.py`. Verifies that GFPGAN is installed correctly, the model weights load, and the augmentation pipeline produces valid output on real face images. |
| `resnet_50.py` | Pretrained ResNet-50 baseline. Extracts embeddings from images, runs cosine-similarity retrieval, evaluates Top-1/5/10 locally on an LFW-based split. Used as a reference baseline. |
| `siglip2_retrieval.py` | SigLIP2 baseline with optional MTCNN face cropping and query expansion. Used as a zero-shot alternative to the fine-tuned CLIP pipeline. |
| `requirements.txt` | Python dependencies. |

---

## `gfpgan_augment.py` — Detailed Documentation

### Why we need this script

Our team's primary pipeline relies on a CLIP model that has been fine-tuned on **VGGFace2-HQ** — a version of VGGFace2 where every real photo has been processed by GFPGAN for face restoration and enhancement. This means our fine-tuned model has learned to map images that *look like GFPGAN-restored faces* to identity-discriminative embeddings.

On competition day we receive:
- **Query images**: real photographs of celebrities (natural domain).
- **Gallery images**: synthetic AI-generated images of celebrities (synthetic domain).

Our fine-tuned models expect images that resemble GFPGAN-restored faces. The competition data does not naturally match this expectation, so we apply GFPGAN at test time to close the gap.

### Two strategies to consider

There are two reasonable ways to apply GFPGAN at test time, and which one works best is an empirical question that should be tested rather than assumed:

**Strategy 1 — Augment gallery only.** Real queries already roughly match the "real" side of our training pairs, so we leave them as-is. The synthetic gallery is augmented to push it toward the "restored" side. This relies on the model's *learned cross-domain mapping* to bring query and gallery embeddings together.

**Strategy 2 — Augment both query and gallery.** Both folders are processed by GFPGAN, putting them into a shared "GFPGAN-output" domain. This makes the inputs more pixel-level similar to each other at the cost of moving queries away from the training "real" distribution.

Each strategy has plausible arguments in its favor:
- Strategy 1 preserves the alignment between queries and the training "real" side, but assumes GFPGAN on synthetic input produces something close enough to the training "restored" side.
- Strategy 2 maximizes the similarity between query and gallery inputs at inference, but assumes the fine-tuned model is robust to a domain shift on the query side as well.

**Recommendation: prepare and benchmark both pipelines.** During the dry run (and on competition day if time permits), produce results for both strategies on the same data and submit whichever scores higher. The augmentation script doesn't care which folder it's pointed at, so producing both versions is just two invocations of the same script.

### What GFPGAN does to an image

GFPGAN is a generative face restoration model. Given an input image, it:
1. Detects faces using a built-in face detector (RetinaFace).
2. Aligns and crops each detected face.
3. Restores the face using a pretrained GAN, producing a sharper, higher-resolution version with smoother skin texture and more defined facial features.
4. Pastes the restored face back into the original image.

The output is an image where faces have a characteristic "GFPGAN signature" — slightly smoothed skin, enhanced eye/lip details, doubled resolution by default.

### Inputs and outputs

**Inputs**
- A folder of images (the competition gallery).
- A path to the GFPGAN pretrained model weights (`GFPGANCleanv1-NoCE-C2.pth`).

**Outputs**
- A new folder containing the same filenames as the input folder, but with each image GFPGAN-restored.

Filenames are preserved exactly. This is critical because the submission format requires matching gallery filenames to query filenames.

### Failure handling

If GFPGAN fails on a particular image (no face detected, corrupt file, etc.), the script copies the original image to the output folder unchanged. This ensures the output folder always contains exactly as many files as the input folder, with the same names. Otherwise the retrieval step downstream would fail.

### Setup (once, before competition day)

```bash
# Install GFPGAN and dependencies
pip install gfpgan
pip install opencv-python-headless

# Fix the known basicsr compatibility issue (see Troubleshooting below)

# Download the pretrained model weights
mkdir -p experiments/pretrained_models
wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth \
     -O experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth
```

After setup, run the test script to verify everything works:

```bash
python test_gfpgan.py \
    --model_path experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth \
    --sample_image path/to/some_face.jpg \
    --extra_test_images path/to/folder_with_face_images/
```

All four tests should pass before competition day.

### Usage on competition day

Assume the competition data lives in `COMPETITION_DATA_FOLDER/` with the structure described in the handout:

```
COMPETITION_DATA_FOLDER/
├── train/
│   ├── identity1/
│   │   ├── aaa.jpg
│   │   └── ...
│   └── identity2/
│       └── ...
└── test/
    ├── query/
    │   ├── aaa.jpg
    │   └── ...
    └── gallery/
        ├── aaa.jpg
        └── ...
```

**Strategy 1 — augment gallery only:**

```bash
python gfpgan_augment.py \
    --input COMPETITION_DATA_FOLDER/test/gallery/ \
    --output COMPETITION_DATA_FOLDER/test/gallery_hq/ \
    --model_path experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth
```

The downstream retrieval script then uses `COMPETITION_DATA_FOLDER/test/query/` and `COMPETITION_DATA_FOLDER/test/gallery_hq/`.

**Strategy 2 — augment both:**

```bash
# Same as above, plus:
python gfpgan_augment.py \
    --input COMPETITION_DATA_FOLDER/test/query/ \
    --output COMPETITION_DATA_FOLDER/test/query_hq/ \
    --model_path experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth
```

The downstream retrieval script then uses `COMPETITION_DATA_FOLDER/test/query_hq/` and `COMPETITION_DATA_FOLDER/test/gallery_hq/`.

**Do not augment the training data on competition day** — fine-tuning has already been done during the prep phase. If any test-time refinement using the competition training set is planned, augmentation of training images would need to be considered separately.

### Timing considerations

GFPGAN processes images one at a time and is moderately compute-intensive. On a single GPU expect roughly 0.5–1 second per image, so ~3000 gallery images should take 25–50 minutes. Plan accordingly given the 2-hour competition window:

- Start gallery augmentation **immediately** after receiving the test data.
- If running Strategy 2, the query folder (~1500 images) also needs augmenting — that's another 15–25 minutes.
- Embedding extraction on un-augmented images can happen in parallel with augmentation if your hardware allows.
- Plan to have **both** the un-augmented baseline submission and the augmented submission ready before the deadline, so you have a fallback if augmentation produces unexpected issues.

If timing is very tight, consider augmenting only the gallery (Strategy 1) — it's the higher-priority side because the model's training pairs had real images on the query side and restored images on the gallery side.

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--input` | Folder of input images | required |
| `--output` | Folder for output images | required |
| `--model_path` | Path to GFPGAN weights | `experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth` |
| `--upscale` | Upscale factor (output is `upscale × input` resolution) | 2 |

**Note on `--upscale`**: at upscale=2 a 224×224 input becomes 448×448. This is usually fine, but if your downstream embedding model expects a specific input size, the resize will happen anyway during preprocessing. The upscale affects the quality of the restoration more than the final embedding size.

---

## Other Scripts (Brief)

### `test_gfpgan.py`

Pre-competition sanity check for the GFPGAN pipeline. Tests model loading, single-image processing, batch file I/O, and graceful handling of corrupt files. Use this once after setting up GFPGAN to confirm everything works. See the `gfpgan_augment.py` setup section above for usage.

### `resnet_50.py`

Loads pretrained ResNet-50 (ImageNet), removes the classification head, and uses the resulting 2048-d feature vectors as image embeddings. Includes a data preparation step that downloads LFW and splits it into a query/gallery format mimicking the competition. Reports Top-1/5/10 locally.

Used as the simplest possible baseline — ResNet-50 was not trained on faces specifically, so its performance establishes a floor that any face-aware model should clearly exceed.

```bash
python resnet_50.py
```

### `siglip2_retrieval.py`

Loads pretrained SigLIP2 from HuggingFace, optionally applies MTCNN face cropping as preprocessing, and runs the retrieval pipeline. Includes a query-expansion post-processing option that averages each query embedding with its top-k nearest gallery neighbors before re-ranking.

Useful as a strong zero-shot baseline. SigLIP2 was trained on a very diverse image-text corpus including illustrations and generated images, so it may handle the real-vs-synthetic gap better than face-recognition-specific models out of the box.

```bash
python siglip2_retrieval.py
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`

This appears when importing `basicsr` (a GFPGAN dependency) with recent torchvision versions. The function was moved between modules. Fix it by editing the offending line:

```bash
# Find the file
python -c "import basicsr, os; print(os.path.dirname(basicsr.__file__))"

# Edit <that_path>/data/degradations.py, change line 8:
#   from torchvision.transforms.functional_tensor import rgb_to_grayscale
# to:
#   from torchvision.transforms.functional import rgb_to_grayscale
```

Or with a one-liner (adjust path for your Python version):

```bash
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    $(python -c "import basicsr, os; print(os.path.dirname(basicsr.__file__))")/data/degradations.py
```

### `FileNotFoundError: ... GFPGANCleanv1-NoCE-C2.pth`

The pretrained model weights aren't included with the pip install. Download them as shown in the setup section.

### `ModuleNotFoundError: No module named 'gfpgan'`

Install with `pip install gfpgan`. If the install fails on Python 3.13 (Windows), use the Linux VM instead — GFPGAN doesn't currently support Python 3.13.

### Output image identical to input

GFPGAN's face detector failed to find a face. In this case the script falls back to copying the original. Check the input image manually — extreme angles, very low resolution, or non-face images cause this. For competition gallery images this should be rare since they are explicitly face images, but worth monitoring during the augmentation step.

---

## Requirements

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

For PyTorch with the correct CUDA version for the Azure VM, install it first separately:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
