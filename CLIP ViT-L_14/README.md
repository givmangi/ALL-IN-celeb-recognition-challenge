# CLIP ViT-L/14 — Celebrity Retrieval

CLIP ViT-L/14 pipeline for the Celebrity Retrieval Across Domains competition (real celebrity photos → synthetic/AI-generated gallery images).

Best result: **734.00 / 1000** (Top-1: 69.00%, Top-5: 79.00%, Top-10: 83.00%), achieved by exactly 2 epochs on the competition training set.

Full experiment log: see [`CLIP_submission_log.txt`].

> **Note:** `.pt` files are **not included in this repo** (too large for GitHub).

---

## 1. Environment Setup

```bash
conda create -n ML_comp_env python=3.12
conda activate ML_comp_env
pip install -r requirements.txt

```

Requires a CUDA-capable GPU (developed/tested on a Tesla V100 16GB). Due to the memory footprint of ViT-L/14, batch sizes during training and extraction are constrained to 64.

---

## 2. Dataset Download

### Competition data (for domain adaptation)

```bash
gdown 1gWtHBZvDWtDyFPnV8fMX789xO-ugqplL  # train: 250 identities, 5,000 images
gdown 1gPdQoLsuaMIxobZreNpsJLEdsb9flsy7  # test: 300 query, 3,000 gallery

```

*(Note: VGGFace2-HQ was used for preliminary ablation experiments but is not required to reproduce the final best result).*

---

## 3. Reproducing the Best Result (734.00)

Fine-tuning utilizes a Triplet Margin Loss (margin = 1.0) and Adam optimizer (LR = 1e-6). To prevent catastrophic forgetting, all parameters are frozen except for the final two transformer blocks (`resblocks[-2:]`).

### Fine-tuning

```bash
python 00train.py

```

* Trains directly from the raw CLIP weights (`STARTING_WEIGHTS = ""`).


* Processes 10,000 triplets per epoch.


* **Exactly 2 epochs is optimal.** The script saves a checkpoint for every epoch (e.g., `clip_competition_epoch2.pt`) and tracks the best overall loss.



### Run, evaluate, and submit

Open `00runPREPARED.py` and ensure the following are set:

```python
MODE = "competition"
WEIGHTS_PATH = "clip_competition_epoch2.pt"
GROUP_NAME = "ALL-IN"

```

```bash
python 00runPREPARED.py

```

This preprocesses images (224x224 center crop, standard CLIP normalization), computes L2-normalized embeddings, utilizes Scikit-Learn `NearestNeighbors` (brute-force cosine similarity) for ranking, and submits the JSON payload directly to the server.

---

## 4. File Overview

| File | Purpose |
| --- | --- |
| `00train.py` | Domain adaptation script: fine-tunes the raw ViT-L/14 model on competition triplets.
 |
| `python 00runPREPARED.py` | Inference script: extracts features, builds KNN index, evaluates local splits, and submits to the competition server.
 |
| `submission_log.txt` | Automatically generated log of all server submissions and HTTP responses.
 |
 
---

## 5. Key Findings

* **Foundation models excel at domain shift:** Even without any fine-tuning, the zero-shot CLIP model scored 664.33, outperforming baseline.
* **Direct Domain Adaptation:** Fine-tuning the raw CLIP model directly on the competition data (`STARTING_WEIGHTS = ""`) proved highly effective. The base model already possessed sufficient facial awareness without needing intermediate VGGFace2-HQ pre-training.


* **Strict early stopping is critical:** Training on the competition dataset peaks quickly. The `clip_competition_epoch2.pt` weights yielded the highest score on the test server.



---
