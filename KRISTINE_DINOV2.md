# Kristine's DINOv2 Experiment

## Model
DINOv2 (ViT-B/14) — self-supervised vision transformer by Meta AI, used as a pretrained feature extractor without any fine-tuning.

## Approach
- Extract 768-dimensional embeddings from DINOv2 for all query and gallery images
- Normalize embeddings to unit length
- Rank gallery images by cosine similarity to each query

## Results on LFW (first run, no fine-tuning)
| Metric | Score |
|--------|-------|
| Top-1  | 17.74% |
| Top-5  | 26.73% |
| Top-10 | 31.73% |

## Papers Read
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (Meta AI, 2023)
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020)

## Next Steps
- Try larger DINOv2 model (vitl14)
- Try larger input resolution
- Compare with competition dataset on May 21st
