# Kristine's DINOv2 Experiments

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
| DINOv2 ViT-B/14 | 224 | No | Yes (split) | **75.29%** | **87.36%** | **89.08%** |
| DINOv2 ViT-B/14 | 224 | No | Yes (100%) | N/A* | N/A* | N/A* |

*100% model cannot be fairly evaluated on VGGFace2-HQ as it was trained on all identities including the test set. Will be evaluated on competition data.

## Key Findings

**Fine-tuning with triplet loss is the biggest improvement** — Top-1 jumped from 
53.45% to 75.29% (+21.84%) after 15 epochs of triplet loss fine-tuning on 
VGGFace2-HQ. This is the most impactful finding of all experiments.

**Resolution 224 slightly outperforms 336 on Top-1** — on the VGGFace2-HQ test 
split, 224 scored 53.45% vs 336's 50.57% Top-1. The difference is within 
statistical noise (5 images out of 174 queries) but 224 is also faster.

**Face detection consistently hurts performance** — MTCNN face cropping reduced 
performance on both LFW and VGGFace2-HQ. DINOv2 appears to use contextual 
information beyond just the face region (hair, clothing, background) for 
identity matching. Face detection may help on the actual competition dataset 
where images have more varied backgrounds.

**ViT-L does not outperform ViT-B** — despite having 3.5x more parameters, 
ViT-L/14 at 224 scored lower than ViT-B/14 at 224 on LFW. ViT-L at 336 
exceeded GPU memory limits.

## Competition Day Strategy
Three scripts ready to run in parallel:

| Script | Model | Top-1 (VGGFace2) |
|--------|-------|------------------|
| `dinov2_vitb14_res336.py` | Pretrained, no fine-tuning | 53.45% |
| `dinov2_finetuned_split.py` | Fine-tuned on 80% split | **75.29%** |
| `dinov2_finetuned_100pct.py` | Fine-tuned on 100% data | TBD (competition) |

## Training Details
- Model: DINOv2 ViT-B/14
- Loss: Triplet Margin Loss (margin=0.3)
- Optimizer: Adam (lr=1e-5)
- Epochs: 15 (best at epoch 13)
- Batch size: 16 triplets
- Triplets per epoch: 5,000
- Frozen layers: all except last 2 transformer blocks
- Augmentation: random flip, color jitter, random grayscale
- Training data: VGGFace2-HQ part 1 (1,380 identities for split, 1,726 for 100%)

## Limitations and Domain Shift Analysis
DINOv2 was pretrained on LVD-142M, a curated dataset of 142 million diverse 
internet images. While this gives robustness to general domain shifts, the 
specific domain gap in our competition is different:

- Query: real celebrity photographs
- Gallery: AI-generated/synthetic images from diffusion models

Synthetic images have specific characteristics (smooth textures, perfect 
lighting, generation artifacts) not directly addressed by DINOv2's pretraining 
distribution. However, DINOv2's deep structural features (face shape, 
proportions, feature relationships) are preserved even in synthetic images.

The triplet loss fine-tuning on VGGFace2-HQ helps the model learn better 
identity-discriminative features, but the train/test domain gap (real photos 
vs synthetic images) remains the key challenge for competition day.

## Papers Read
- Oquab et al., "DINOv2: Learning Robust Visual Features without 
  Supervision" (Meta AI, 2023)
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers 
  for Image Recognition at Scale" (2020)
- Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face 
  Recognition" (2019)
- Hu & Lee, "Feature Representation Learning for Unsupervised 
  Cross-domain Image Retrieval" (2022)