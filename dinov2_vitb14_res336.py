"""
DINOv2 ViT-B/14 - Best Configuration (336x336 resolution)
===========================================================
Author: Kristine
Branch: kristine-dinov2

Model: DINOv2 ViT-B/14 — self-supervised vision transformer by Meta AI,
used as a pretrained feature extractor without any fine-tuning.

Key improvements over baseline (dinov2_model.py):
- Higher resolution: 336x336 instead of 224x224
- [NEW] EXIF rotation fix: corrects images rotated by phone cameras
- [NEW] Face detection cropping: uses MTCNN to detect and crop just the
  face region before feature extraction, removing distracting backgrounds.
  Falls back to full image if no face is detected.

Approach:
- Detect and crop face region using MTCNN (facenet-pytorch)
- Fix image rotation using EXIF metadata
- Extract 768-dimensional embeddings from DINOv2 ViT-B/14
- Normalize embeddings to unit length (L2 normalization)
- Rank gallery images by cosine similarity to each query
- Evaluate locally using identity prefix in filenames

Results on LFW (no fine-tuning):
- DINOv2 ViT-B/14 @ 224: Top-1: 17.74%, Top-5: 26.73%, Top-10: 31.73%
- DINOv2 ViT-B/14 @ 336: Top-1: 18.63%, Top-5: 27.62%, Top-10: 31.55%
- This version (336 + face crop + rotation): TBD
"""

import os
import json
import datetime
from PIL import Image, ImageOps
import requests
import torch
import torchvision.transforms as T

# [NEW] Face detection
from facenet_pytorch import MTCNN

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Submit function ───────────────────────────────────────────────────────────
def submit(results, groupname, url):
    res = {}
    res['groupname'] = groupname
    res['images'] = results
    res = json.dumps(res)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

# ── Load DINOv2 ───────────────────────────────────────────────────────────────
print("Loading DINOv2 model...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model = model.to(device)
model.eval()

# ── [NEW] Load MTCNN face detector ────────────────────────────────────────────
print("Loading face detector...")
mtcnn = MTCNN(
    image_size=336,   # output face crop size matches our resolution
    margin=20,        # add a small margin around the detected face
    device=device,
    keep_all=False    # only keep the most prominent face
)

# ── Preprocessing pipeline ────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize(336),
    T.CenterCrop(336),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ── [NEW] Face detection and cropping function ────────────────────────────────
def detect_and_crop_face(img):
    """
    Detect and crop the face region from an image using MTCNN.
    Falls back to the original image if no face is detected.
    """
    face_tensor = mtcnn(img)
    if face_tensor is not None:
        # Convert tensor back to PIL image for preprocessing pipeline
        face_img = T.ToPILImage()(face_tensor.cpu().clamp(0, 1))
        return face_img
    else:
        # No face detected — use original image as fallback
        return img

# ── Batching function ─────────────────────────────────────────────────────────
def batching(images, batch_size=8):
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = torch.stack(
            [preprocess(img.convert("RGB")) for img in batch]
        ).to(device)
        with torch.no_grad():
            feats = model(inputs)
            features.append(feats)
    return torch.cat(features, dim=0)

# ── Paths ─────────────────────────────────────────────────────────────────────
data_folder = "/home/disi/data/lfw_split"
query_folder = os.path.join(data_folder, "query")
gallery_folder = os.path.join(data_folder, "gallery")

# ── Load images with rotation fix and face cropping ──────────────────────────
query_images, query_filenames = [], []
gallery_images, gallery_filenames = [], []

print("Loading query images...")
for filename in os.listdir(query_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        query_filenames.append(filename)
        with Image.open(os.path.join(query_folder, filename)) as img:
            img = img.convert("RGB")
            img = ImageOps.exif_transpose(img)  # [NEW] rotation fix
            img = detect_and_crop_face(img)      # [NEW] face crop
            query_images.append(img.copy())

print("Loading gallery images...")
for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        gallery_filenames.append(filename)
        with Image.open(os.path.join(gallery_folder, filename)) as img:
            img = img.convert("RGB")
            img = ImageOps.exif_transpose(img)  # [NEW] rotation fix
            img = detect_and_crop_face(img)      # [NEW] face crop
            gallery_images.append(img.copy())

print(f"Query images: {len(query_images)}")
print(f"Gallery images: {len(gallery_images)}")

# ── Extract features ──────────────────────────────────────────────────────────
print("Processing query images...")
query_features = batching(query_images, batch_size=8)
print("Processing gallery images...")
gallery_features = batching(gallery_images, batch_size=8)

# ── Normalize ─────────────────────────────────────────────────────────────────
print("Normalizing features...")
query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)

# ── Compute similarity ────────────────────────────────────────────────────────
print("Computing similarity matrix...")
similarity_matrix = torch.matmul(query_features, gallery_features.T)

# ── Get top 10 matches ────────────────────────────────────────────────────────
top_k = 10
_, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)

# ── Build results dictionary ──────────────────────────────────────────────────
results = {}
for i, query_filename in enumerate(query_filenames):
    results[query_filename] = [
        gallery_filenames[idx] for idx in top_k_indices[i]
    ]

# ── Local evaluation ──────────────────────────────────────────────────────────
print("\nEvaluating...")
top1, top5, top10, total = 0, 0, 0, 0

for i, query_filename in enumerate(query_filenames):
    query_identity = query_filename.split("__")[0]
    retrieved = [gallery_filenames[idx].split("__")[0] for idx in top_k_indices[i]]

    total += 1
    if query_identity == retrieved[0]:
        top1 += 1
    if query_identity in retrieved[:5]:
        top5 += 1
    if query_identity in retrieved[:10]:
        top10 += 1

print(f"Top-1  accuracy: {top1  / total:.2%}")
print(f"Top-5  accuracy: {top5  / total:.2%}")
print(f"Top-10 accuracy: {top10 / total:.2%}")

# ── Save results ──────────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
with open(f"results/results_{timestamp}.txt", "w") as f:
    f.write(f"Model: DINOv2 ViT-B/14\n")
    f.write(f"Resolution: 336x336\n")
    f.write(f"Face detection: MTCNN\n")
    f.write(f"Rotation fix: EXIF transpose\n")
    f.write(f"Dataset: LFW split\n")
    f.write(f"Top-1:  {top1/total:.2%}\n")
    f.write(f"Top-5:  {top5/total:.2%}\n")
    f.write(f"Top-10: {top10/total:.2%}\n")
print(f"Results saved to results/results_{timestamp}.txt")

# ── Submit (uncomment on competition day) ─────────────────────────────────────
# submit(
#     results=results,
#     groupname="ALL-IN-dinov2",
#     url="http://competition-server-url/retrieval/"
# )