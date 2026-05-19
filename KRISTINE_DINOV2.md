# Kristine's DINOv2 Experiments

## Models Tested
DINOv2 is a self-supervised vision transformer by Meta AI, used as a 
pretrained feature extractor without any fine-tuning. Two variants were tested:

- **ViT-B/14** — Base model, 86M parameters, 768-dimensional embeddings
- **ViT-L/14** — Large model, 307M parameters, 1024-dimensional embeddings

## Approach
- Load pretrained DINOv2 as a frozen feature extractor
- Preprocess images (resize, center crop, normalize)
- Extract embeddings for all query and gallery images on-the-fly (no RAM overload)
- Normalize embeddings to unit length (L2 normalization)
- Rank gallery images by cosine similarity to each query
- Return top 10 matches per query
- Local evaluation using identity prefix in filenames (no server needed)

## Results on LFW (real images only, no fine-tuning)

| Model | Resolution | Face Crop | Top-1 | Top-5 | Top-10 |
|-------|-----------|-----------|-------|-------|--------|
| DINOv2 ViT-B/14 | 224 | No | 17.74% | 26.73% | 31.73% |
| DINOv2 ViT-B/14 | 336 | No | 18.63% | 27.62% | 31.55% |
| DINOv2 ViT-L/14 | 224 | No | 16.49% | 27.62% | 32.14% |
| DINOv2 ViT-B/14 | 224 | Yes | 11.79% | 20.18% | 25.54% |
| DINOv2 ViT-B/14 | 336 | Yes | 10.48% | 19.52% | 24.17% |

## Results on VGGFace2-HQ (no fine-tuning, no face crop)

| Model | Resolution | Top-1 | Top-5 | Top-10 |
|-------|-----------|-------|-------|--------|
| DINOv2 ViT-B/14 | 224 | 31.81% | 45.83% | 52.14% |
| DINOv2 ViT-B/14 | 336 | 32.44% | 46.58% | 53.59% |
| DINOv2 ViT-L/14 | 224 | running... | | |

## Key Findings

**Resolution matters more than model size** — ViT-B/14 at 336x336 consistently 
outperforms both ViT-B/14 at 224 and ViT-L/14 at 224. Higher resolution gives 
DINOv2 finer patch detail (14x14 pixel patches over a larger canvas) which 
helps capture subtle facial features.

**Face detection hurts on LFW** — adding MTCNN face cropping reduced performance 
on LFW (e.g. Top-1 dropped from 17.74% to 11.79% at 224 resolution). This is 
because LFW images are already tightly cropped around faces — MTCNN over-crops 
them, losing important facial context like hair, ears, and jaw line that DINOv2 
uses for identity matching. Face detection is expected to help on the competition 
dataset where images have full backgrounds.

**VGGFace2-HQ scores are higher than LFW** — this is expected since VGGFace2 
has ~141 images per identity in the gallery vs LFW's average of 4-5, making 
it easier to find a correct match.

## Best Configuration
**DINOv2 ViT-B/14 at 336x336 resolution, batch_size=8, no face crop for large datasets**

## Limitations and Domain Shift Analysis
DINOv2 was pretrained on LVD-142M, a curated dataset of 142 million 
diverse internet images. While this gives robustness to general domain 
shifts (photos vs illustrations, different lighting, styles), the specific 
domain gap in our competition is different:

- Query: real celebrity photographs
- Gallery: AI-generated/synthetic images from diffusion models

Synthetic images have specific characteristics (smooth textures, perfect 
lighting, generation artifacts) not directly addressed by DINOv2's 
pretraining distribution. However, DINOv2's deep structural features 
(face shape, proportions, feature relationships) are preserved even in 
synthetic images, which may still allow meaningful cross-domain matching.

The ideal solution would be fine-tuning on paired real+synthetic images 
of the same identity — exactly the approach explored in the related work 
(George & Marcel, 2024; Hu & Lee, 2022).

## Papers Read
- Oquab et al., "DINOv2: Learning Robust Visual Features without 
  Supervision" (Meta AI, 2023)
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers 
  for Image Recognition at Scale" (2020)
- Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face 
  Recognition" (2019)
- Hu & Lee, "Feature Representation Learning for Unsupervised 
  Cross-domain Image Retrieval" (2022)

## Next Steps
- Complete ViT-L/14 @ 224 experiment on VGGFace2-HQ
- Fine-tune best configuration (ViT-B/14 @ 336) with triplet loss overnight
- Compare fine-tuned vs baseline on VGGFace2-HQ
- On competition day: run with face detection enabled (small dataset, manageable)