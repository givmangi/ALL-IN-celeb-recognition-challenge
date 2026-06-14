# DINOv2 — Celebrity Retrieval

DINOv2 ViT-B/14 pipeline for the Celebrity Retrieval Across Domains competition
(real celebrity photos → synthetic/AI-generated gallery images).

Best result: **272.67 / 1000** (Top-1: 18.67%, Top-5: 38.00%, Top-10: 46.67%),
achieved by fine-tuning on VGGFace2-HQ (100%) followed by 3 epochs on the
competition training set.

Full experiment log, ablations, and findings: see [`DINOv2_experiment_log.md`](DINOv2_experiment_log.md).

> **Note on checkpoints:** `.pth` checkpoint files are **not included in this
> repo** (too large for GitHub). They live on the project VM under
> `checkpoints/`. If a checkpoint path in `DINOv2_run_models.py` doesn't exist,
> the script automatically falls back to the pretrained model and prints a
> warning — it won't crash, but you won't get fine-tuned results without
> access to the VM checkpoints.

---

## 1. Environment Setup

```bash
conda create -n ML_comp_env python=3.12
conda activate ML_comp_env
pip install -r requirements.txt
```

Requires a CUDA-capable GPU (developed/tested on a Tesla V100 16GB).

---

## 2. Dataset Download

### LFW (preliminary experiments)

LFW (Labeled Faces in the Wild) was used for the earliest experiments —
comparing ViT-B/14 vs ViT-L/14, resolution 224 vs 336, and with/without
MTCNN face cropping — before moving to VGGFace2-HQ. Results are in
`DINOv2_experiment_log.md`.

Expected layout (query/gallery split, 1,680 queries, 7,484 gallery):
```
/home/disi/data/lfw_split/
├── query/
└── gallery/
```

No training was done on LFW — pretrained DINOv2 only, used for ablations.
Not required to reproduce the final competition result.

### VGGFace2-HQ (for pre-training)

Download `vgg_face_parts/1.tar` from HuggingFace
[`RichardErkhov/VGGFace2-HQ`](https://huggingface.co/RichardErkhov/VGGFace2-HQ)
and extract to:

```
/home/disi/data/vggface2_raw/1/
```

Then create the train/val/test split:

```bash
python create_train_val_test_split.py
```

This produces `/home/disi/data/vggface2_train_split/` with:
- `train/` — 1,380 identities
- `val/` — 172 identities
- `test/query/` + `test/gallery/` — 174 queries, 24,495 gallery images

### Competition data

```bash
gdown 1gWtHBZvDWtDyFPnV8fMX789xO-ugqplL  # train: 250 identities, 5,000 images
gdown 1gPdQoLsuaMIxobZreNpsJLEdsb9flsy7  # test: 300 query, 3,000 gallery
```

Extract to `/home/disi/data/competition_train/` and `/home/disi/data/competition_test/`.

---

## 3. Reproducing the Best Result (272.67)

Two-stage fine-tuning, then run on the competition test set.

### Stage 1 — VGGFace2 fine-tuning (15 epochs, ~3-5 hrs on V100)

```bash
python train_dinov2.py
```

- Trains on `/home/disi/data/vggface2_train_split/train`
- Validates each epoch on `val/`, saves best checkpoint by Top-1 accuracy
- Best checkpoint: **epoch 3, val Top-1 79.07%** → `checkpoints/best_model_split.pth`
- To use the 100% (no held-out split) variant for the best result, retrain on
  the full VGGFace2 set (1,726 identities) — point `DATA_FOLDER` at the full
  dataset and skip validation, or adapt the script accordingly. Save the
  result as `checkpoints/best_model_100pct.pth`.

### Stage 2 — Competition fine-tuning (3 epochs, ~20-30 min on V100)

Edit `train_competition.py`:
```python
CHECKPOINT_IN  = "checkpoints/best_model_100pct.pth"   # stage 1 output
CHECKPOINT_OUT = "checkpoints/best_model_competition.pth"
```

```bash
python train_competition.py
```

- 3 epochs is optimal — overfits past that (15 epochs drops score to 250.67)

### Stage 3 — Run, evaluate, and submit

Open `DINOv2_run_models.py` and set:
```python
CHECKPOINT = "checkpoints/best_model_competition.pth"
SUBMISSION_NAME = "DINOv2_100pct_competition"
DATA_FOLDER = "/home/disi/data/competition_test"
```

```bash
python DINOv2_run_models.py
```

This computes embeddings, ranks the gallery by cosine similarity, prints
local Top-1/5/10 accuracy, saves results to `results/`, and submits to the
competition server (set `SUBMIT_RESULTS = False` to skip submission).

---

## 4. Running Other Experiments

`DINOv2_run_models.py` is config-driven — change `CHECKPOINT` at the top to
switch between models:

| `CHECKPOINT` value | Description | Score |
|---|---|---|
| `None` | Pretrained baseline | 123.67 |
| `None` + `RESOLUTION = 336` | Pretrained, 336px | 136.00 |
| `"checkpoints/best_model_split_v3_epoch3.pth"` | VGGFace2 80% split | 218.33 |
| `"checkpoints/best_model_100pct.pth"` | VGGFace2 100% | 232.67 |
| `"checkpoints/best_model_pretrained_competition.pth"` | Pretrained → competition (no VGGFace2) | 244.33 |
| `"checkpoints/best_model_competition.pth"` (15 epochs) | VGGFace2 + competition, 15 ep | 250.67 |
| **`"checkpoints/best_model_competition.pth"` (3 epochs, 100% base)** | **Best** | **272.67** |

For configurable ablations (face crop, resolution, etc.), see `experiments/dinov2_experiment.py` —
same on-the-fly extraction + local eval as `DINOv2_run_models.py`, but no
checkpoint/submission logic, and caches embeddings per (model, resolution, dataset)
combination for quick reruns.

---

## 5. File Overview

| File | Purpose |
|---|---|
| `DINOv2_run_models.py` | Run any checkpoint: extract features, evaluate, submit |
| `train_dinov2.py` | Stage 1: fine-tune on VGGFace2-HQ with triplet loss + validation |
| `train_competition.py` | Stage 2: fine-tune from a checkpoint on competition data (3 epochs) |
| `create_train_val_test_split.py` | Builds the VGGFace2 train/val/test split (seed=1) |
| `experiments/dinov2_experiment.py` | Configurable ablation script (resolution, face crop, dataset) |
| `results/` | Saved evaluation results per run |
| `submission_log.txt` | Log of all server submissions |
| `DINOv2_experiment_log.md` | Full experiment log, ablation tables, key findings |

---

## 6. Key Findings (see `DINOv2_experiment_log.md` for details)

- Two-stage fine-tuning (VGGFace2 → competition) is necessary — skipping
  VGGFace2 pre-training loses ~28 points
- 3 epochs is optimal for competition fine-tuning (overfits on 250 identities by 15 epochs)
- Face detection (MTCNN) consistently hurts performance
- ViT-L/14 does not outperform ViT-B/14, and exceeds 16GB GPU memory at 336px