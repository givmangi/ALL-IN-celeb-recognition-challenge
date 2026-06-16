# ALL-IN — Celeb Recognition Challenge

Repository for the **Celebrity Retrieval Across Domains** competition
(Introduction to Machine Learning, University of Trento, Spring 2026).

Task: match real celebrity photos (query) against a gallery of
synthetic/AI-generated images, using image retrieval. Submissions are scored
on Top-1, Top-5, and Top-10 accuracy.

## Structure

Each model lives in its own self-contained folder, with its own scripts,
requirements, results, and README. To reproduce any model's results, go into
that folder and follow its README — no cross-folder dependencies.

| Folder | Approach | Best Score |
|---|---|---|
| [`ResNet-50/`](ResNet-50/) | ResNet-50 Baseline with Random Init | 80.67 |
| [`DINOv2/`](DINOv2/) | DINOv2 ViT-B/14, two-stage fine-tuning (VGGFace2 → competition) | 272.67 |
| [`ArcFace/`](ArcFace/) | Custom CNN Backbone + ArcMargin Product Classification Head (Competition only, 30 epochs) | 278.00 |
| [`CLIP/`](CLIP/) | CLIP ViT-L/14, direct domain adaptation (competition data only, 2 epochs) | 734.00 |

## Report

The full written report (methodology, results, and discussion across all models) was submitted separately as the course deliverable and is not included in this repository.
