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

Our team's primary pipeline relies on models that has been fine-tuned on **VGGFace2-HQ** — a version of VGGFace2 where every real photo has been processed by GFPGAN for face restoration and enhancement. This means our fine-tuned model has learned to map images that *look like GFPGAN-restored faces* to identity-discriminative embeddings.

On competition day we receive:
- **Query images**: real photographs of celebrities (natural domain).
- **Gallery images**: synthetic AI-generated images of celebrities (synthetic domain).

Our model expects images in the GFPGAN-restored style. Real query photos already roughly match the "original" side of our training pairs. The synthetic gallery images, however, do not match anything the model saw during fine-tuning. By applying GFPGAN to the gallery before extracting embeddings, we push the gallery distribution closer to what our model expects, narrowing the test-time domain gap.

In short: **GFPGAN is applied to the gallery, not the query.** Queries are left as-is.

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

Augment the gallery:

```bash
python gfpgan_augment.py \
    --input COMPETITION_DATA_FOLDER/test/gallery/ \
    --output COMPETITION_DATA_FOLDER/test/gallery_hq/ \
    --model_path experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth
```

This produces `COMPETITION_DATA_FOLDER/test/gallery_hq/` containing the GFPGAN-processed gallery images. The downstream retrieval script should then point at `gallery_hq/` instead of `gallery/`.

**Do not augment the query folder.** Queries are real photos and should remain as-is.

**Do not augment the training data on competition day.** If the team plans to use the competition training set for any test-time refinement, augmentation of training images should also be considered — but for the primary retrieval pipeline, only the gallery is augmented.

### Timing considerations

GFPGAN processes images one at a time and is moderately compute-intensive. On a single GPU expect roughly 0.5–1 second per image, so ~3000 gallery images should take 25–50 minutes. Plan accordingly given the 2-hour competition window:

- Start gallery augmentation **immediately** after receiving the test data.
- The query embeddings can be extracted in parallel while augmentation runs (queries don't need GFPGAN).
- Gallery embeddings are extracted *after* augmentation completes.

If timing is tight, consider running the embedding extraction on the original (un-augmented) gallery as a fallback in case augmentation has issues — both pipelines should be ready to submit.

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
