# Kristine's DINOv2 Experiments

## Models Tested
DINOv2 is a self-supervised vision transformer by Meta AI, used as a 
pretrained feature extractor without any fine-tuning. Two variants were tested:

- **ViT-B/14** — Base model, 86M parameters, 768-dimensional embeddings
- **ViT-L/14** — Large model, 307M parameters, 1024-dimensional embeddings

## Approach
- Load pretrained DINOv2 as a frozen feature extractor
- Preprocess images (resize, center crop, normalize)
- Extract embeddings for all query and gallery images in batches
- Normalize embeddings to unit length (L2 normalization)
- Rank gallery images by cosine similarity to each query
- Return top 10 matches per query

## Results on LFW (real images only, no fine-tuning)

| Model | Resolution | Batch Size | Top-1 | Top-5 | Top-10 |
|-------|-----------|------------|-------|-------|--------|
| DINOv2 ViT-B/14 | 224 | 16 | 17.74% | 26.73% | 31.73% |
| DINOv2 ViT-B/14 | 336 | 8 | 18.63% | 27.62% | 31.55% |
| DINOv2 ViT-L/14 | 224 | 8 | 16.49% | 27.62% | 32.14% |
| DINOv2 ViT-L/14 | 336 | 2 | OOM — exceeded 16GB V100 GPU memory | | |

## Key Findings
- Higher resolution (336 vs 224) improves Top-1 and Top-5 accuracy
- Larger model (ViT-L) does NOT consistently outperform ViT-B — likely 
  because the task benefits more from resolution than model capacity
- ViT-L at 336x336 exceeds available GPU memory even at batch_size=2

## Best Configuration
**DINOv2 ViT-B/14 at 336x336 resolution, batch_size=8**

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
- Wang & Lee, "Feature Representation Learning for Unsupervised 
  Cross-domain Image Retrieval" (2022)

## Next Steps
- Add face detection cropping (facenet-pytorch) to improve identity focus
- Add EXIF rotation fix for robustness
- Retest best configuration on VGGFace2-HQ when available
- Compare with CLIP, ArcFace and ResNet50 baseline on competition day