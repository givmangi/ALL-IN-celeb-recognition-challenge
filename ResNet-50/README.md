# ALL-IN — Celebrity Recognition Challenge (ResNet-50 Baseline)

This branch contains the **ResNet-50 baseline** for the *Celebrity Retrieval Across Domains* competition.

## Task Recap

Given a real photograph of a celebrity (query), retrieve the 10 most similar images of that same celebrity from a gallery. The main challenge is the **domain shift** between query and gallery image distributions. See the course handout for the full task and submission format.

## Approach

A pretrained **ResNet-50** (ImageNet weights) is used as a fixed feature extractor:

1. The classification head (`fc`) is replaced with `nn.Identity()`, so each image is mapped to a 2048-d embedding.
2. Embeddings are L2-normalized.
3. Retrieval is done via cosine similarity: for each query, the top-10 most similar gallery images are returned.

ResNet-50 was not trained for face recognition, so this baseline establishes a performance floor that any face-aware model should clearly beat.

## Scripts

| Script | Purpose |
|---|---|
| `resnet_50.py` | Main script. Supports two modes: `local` (downloads LFW via `kagglehub`, splits it into query/gallery, and reports Top-1/5/10 accuracy) and `competition` (extracts embeddings for the competition query/gallery folders, runs retrieval, and submits results). |
| `resnet_50_with_submit.py` | Variant of `resnet_50.py` with the submission and logging helper integrated directly into the script. |
| `submit.py` | Standalone helper (`submit_and_log`) for posting retrieval results to the competition server and appending the response to a log file. |
| `test-model-resnet50.py` | Minimal end-to-end script: loads query/gallery images from a local folder, extracts ResNet-50 features, runs cosine-similarity retrieval, and submits the results. |
| `resnet-50.txt` | Logged response from a past submission run (accuracy ≈ 80.67%). |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

For PyTorch with the correct CUDA version (e.g. on the Azure VM):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Local evaluation (LFW)

In `resnet_50.py`, set:

```python
MODE = "local"
```

Then run:

```bash
python resnet_50.py
```

This downloads the LFW dataset, splits it into query/gallery folders (one image per identity as query, the rest as gallery), runs retrieval, and prints Top-1/5/10 accuracy.

### Competition submission

In `resnet_50.py` (or `resnet_50_with_submit.py`), set:

```python
MODE = "competition"
QUERY_FOLDER   = "path/to/test/query"
GALLERY_FOLDER = "path/to/test/gallery"
SUBMIT_URL     = "..."   # provided by the professor
GROUP_NAME     = "ALL-IN"
MODEL_TAG      = "ResNet50_ImageNet_baseline"
```

Then run:

```bash
python resnet_50.py
```

The script extracts embeddings for both folders, retrieves the top-10 gallery matches per query, submits the results to the server, and appends the response to `submission_log.txt`.

## Results

Latest logged submission (`resnet-50.txt`): **accuracy score ≈ 80.67** (out of 1000).
