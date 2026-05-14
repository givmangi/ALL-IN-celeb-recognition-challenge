# Kristine's DINOv2 Experiment

## Model
DINOv2 (ViT-B/14) — self-supervised vision transformer by Meta AI, used as a pretrained feature extractor without any fine-tuning.

## Approach
- Extract 768-dimensional embeddings from DINOv2 for all query and gallery images
- Normalize embeddings to unit length
- Rank gallery images by cosine similarity to each query

## Results on LFW
| Model | Resolution | Top-1 | Top-5 | Top-10 |
|-------|-----------|-------|-------|--------|
| DINOv2 ViT-B/14 | 224 | 17.74% | 26.73% | 31.73% |
| DINOv2 ViT-B/14 | 336 | 18.63% | 27.62% | 31.55% |
| DINOv2 ViT-L/14 | 224 | 16.49% | 27.62% | 32.14% |
| DINOv2 ViT-L/14 | 336 | OOM - exceeded 16GB GPU memory | | |

## Conclusion
Best configuration: DINOv2 ViT-B/14 at 336x336 resolution.
ViT-L exceeded GPU memory at 336x336, confirming ViT-B/14 as the 
practical choice. Higher resolution helps more than model size.

## Papers Read
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (Meta AI, 2023)
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020)

## Next Steps
- Add face detection cropping (facenet-pytorch)
- Add EXIF rotation fix
- Retest best configuration on VGGFace2-HQ when available
