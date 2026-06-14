# DINOv2 Experiments

## Models Tested
DINOv2 is a self-supervised vision transformer by Meta AI. Two variants tested:

- **ViT-B/14** — Base model, 86M parameters, 768-dimensional embeddings
- **ViT-L/14** — Large model, 307M parameters, 1024-dimensional embeddings

## Approach
- Load pretrained DINOv2 as a frozen feature extractor
- Preprocess images (resize, center crop, normalize)
- Extract embeddings on-the-fly (no RAM overload)
- Normalize embeddings to unit length (L2 normalization)
- Rank gallery images by cosine similarity to each query
- Return top 10 matches per query
- Local evaluation using identity prefix in filenames

## Results on LFW (real images only, no fine-tuning)

| Model | Resolution | Face Crop | Top-1 | Top-5 | Top-10 |
|-------|-----------|-----------|-------|-------|--------|
| DINOv2 ViT-B/14 | 224 | No | 17.74% | 26.73% | 31.73% |
| DINOv2 ViT-B/14 | 336 | No | 18.63% | 27.62% | 31.55% |
| DINOv2 ViT-L/14 | 224 | No | 16.49% | 27.62% | 32.14% |
| DINOv2 ViT-B/14 | 224 | Yes | 11.79% | 20.18% | 25.54% |
| DINOv2 ViT-B/14 | 336 | Yes | 10.48% | 19.52% | 24.17% |

## Results on VGGFace2-HQ full split (no fine-tuning, no face crop)

| Model | Resolution | Top-1 | Top-5 | Top-10 |
|-------|-----------|-------|-------|--------|
| DINOv2 ViT-B/14 | 224 | 31.81% | 45.83% | 52.14% |
| DINOv2 ViT-B/14 | 336 | 32.44% | 46.58% | 53.59% |

## Results on VGGFace2-HQ test split (174 queries, 24,495 gallery)

| Model | Resolution | Face Crop | Fine-tuned | Top-1 | Top-5 | Top-10 |
|-------|-----------|-----------|------------|-------|-------|--------|
| DINOv2 ViT-B/14 | 224 | No | No | 53.45% | 72.41% | 77.01% |
| DINOv2 ViT-B/14 | 336 | No | No | 50.57% | 73.56% | 76.44% |
| DINOv2 ViT-B/14 | 336 | Yes | No | 39.08% | 65.52% | 74.14% |
| DINOv2 ViT-B/14 | 224 | No | Yes (split, val-guided) | 72.41% | 86.21% | 89.08% |
| DINOv2 ViT-B/14 | 224 | No | Yes (100%) | N/A* | N/A* | N/A* |

*100% model trained on all identities — cannot be fairly evaluated on same dataset.

## Results on Competition Dataset (300 queries, 3,000 gallery)

| Model | Training Path | Comp. Epochs | Score | Top-1 | Top-5 | Top-10 |
|-------|--------------|-------------|-------|-------|-------|--------|
| DINOv2 pretrained | None | - | 123.67 | 8.67% | 16.00% | 23.67% |
| DINOv2 pretrained 336 | None | - | 136.00 | 9.00% | 19.00% | 25.00% |
| DINOv2 finetuned split | VGGFace2 80% | - | 218.33 | 13.67% | 31.67% | 41.33% |
| DINOv2 finetuned 100pct | VGGFace2 100% | - | 232.67 | 15.67% | 32.67% | 40.67% |
| DINOv2 pretrained → competition | Competition only | 3 | 244.33 | 15.67% | 35.67% | 43.33% |
| DINOv2 100pct → competition | VGGFace2 100% + Competition | 15 | 250.67 | 18.33% | 32.67% | 42.67% |
| DINOv2 split → competition | VGGFace2 80% + Competition | 3 | 271.67 | 19.33% | 37.33% | 43.67% |
| **DINOv2 100pct → competition** | **VGGFace2 100% + Competition** | **3** | **272.67** | **18.67%** | **38.00%** | **46.67%** |

## Key Findings

**Fine-tuning with triplet loss is the biggest improvement** — on VGGFace2-HQ,
Top-1 jumped from 53.45% to 72.41% after fine-tuning. On the competition
dataset, fine-tuning improved from 123.67 (pretrained) to 272.67 (best model).

**VGGFace2 pre-training is necessary before competition fine-tuning** —
pretrained → competition (244.33) scores lower than VGGFace2 → competition
(271-272). DINOv2 needs face identity training before adapting to celebrities.

**3 epochs optimal for competition fine-tuning** — 15 epochs (250.67) scores
worse than 3 epochs (272.67). With only 250 training identities the model
overfits quickly. Early stopping is critical on small datasets.

**100pct VGGFace2 training slightly better as base** — 272.67 vs 271.67.
Marginal but consistent advantage from training on all available data.

**Face detection consistently hurts performance** — MTCNN face cropping reduced
scores on both LFW and VGGFace2-HQ. DINOv2 uses contextual information beyond
the face region (hair, background) for identity matching.

**ViT-L does not outperform ViT-B** — despite 3.5x more parameters, ViT-L/14
scored lower on LFW. ViT-L at 336 exceeded GPU memory (16GB V100).

**Resolution 336 marginally better than 224 for pretrained** — 136.00 vs
123.67 on competition data. Difference disappears after fine-tuning.

## Training Details
- Model: DINOv2 ViT-B/14
- Loss: Triplet Margin Loss (margin=0.3)
- Optimizer: Adam (lr=1e-5)
- VGGFace2 training: 15 epochs, best at epoch 3 (val Top-1: 79.07%)
- Competition fine-tuning: 3 epochs (optimal), batch size 8
- Triplets per epoch: 5,000
- Frozen: all layers except last 2 transformer blocks
- Augmentation: random flip, color jitter, random grayscale

## Limitations and Domain Shift Analysis
DINOv2 was pretrained on LVD-142M, a curated dataset of 142 million diverse
internet images. The specific domain gap in our competition:

- Query: real celebrity photographs
- Gallery: AI-generated/synthetic images from diffusion models

Synthetic images have specific characteristics (smooth textures, perfect
lighting, generation artifacts) not in DINOv2's pretraining distribution.
Unlike CLIP which learned celebrity identity from internet captions, DINOv2
relies purely on visual features making the real→synthetic gap harder to bridge.

The two-stage fine-tuning pipeline (VGGFace2 → competition data) partially
addresses this by first learning face identity features, then adapting to the
specific celebrities and image style of the competition.

## Papers Read
- Oquab et al., "DINOv2: Learning Robust Visual Features without
  Supervision" (Meta AI, 2023)
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers
  for Image Recognition at Scale" (2020)
- Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
  Recognition" (2019)
- Hu & Lee, "Feature Representation Learning for Unsupervised
  Cross-domain Image Retrieval" (2022)