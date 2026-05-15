"""
DINOv2 Experiment Script
=========================
Author: Kristine
Branch: kristine-dinov2

Configurable script for testing different DINOv2 configurations.
Change the parameters in the Configuration section below to test
different model sizes and resolutions.

All experiments include:
- EXIF rotation fix
- MTCNN face detection and cropping
- L2 normalized cosine similarity retrieval
- Local evaluation (no server needed)

Experiments run so far:
-------------------------------------------------------
Model           Resolution  Top-1    Top-5    Top-10
-------------------------------------------------------
DINOv2 ViT-B/14   224      17.74%   26.73%   31.73%  (no face crop)
DINOv2 ViT-B/14   336      18.63%   27.62%   31.55%  (no face crop)
DINOv2 ViT-L/14   224      16.49%   27.62%   32.14%  (no face crop)
DINOv2 ViT-L/14   336      OOM - exceeded 16GB GPU memory
-------------------------------------------------------
Results with face crop and rotation fix: TBD
"""

import os
import json
import datetime
from PIL import Image, ImageOps
import requests
import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN

# ── Configuration ─────────────────────────────────────────────────────────────
# Change these to test different configurations
MODEL_NAME  = "dinov2_vitb14"          # "dinov2_vitb14" or "dinov2_vitl14"
RESOLUTION  = 336                       # 224 or 336
BATCH_SIZE  = 8                         # reduce if out of memory (try 4 or 2)
DATA_FOLDER = "/home/disi/data/lfw_split"  # change to vggface2 when available
GROUP_NAME  = "ALL-IN-dinov2"
# ─────────────────────────────────────────────────────────────────────────────

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
print(f"Model: {MODEL_NAME} | Resolution: {RESOLUTION}x{RESOLUTION} | Batch: {BATCH_SIZE}")

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
print(f"Loading {MODEL_NAME}...")
model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
model = model.to(device)
model.eval()

# ── Load MTCNN face detector ──────────────────────────────────────────────────
print("Loading face detector...")
mtcnn = MTCNN(
    image_size=RESOLUTION,
    margin=20,
    device=device,
    keep_all=False
)

# ── Preprocessing pipeline ────────────────────────────────────────────────────
preprocess = T.Compose([
    T.Resize(RESOLUTION),
    T.CenterCrop(RESOLUTION),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ── Face detection and cropping ───────────────────────────────────────────────
def detect_and_crop_face(img):
    """
    Detect and crop face using MTCNN.
    Falls back to original image if no face detected.
    """
    face_tensor = mtcnn(img)
    if face_tensor is not None:
        face_img = T.ToPILImage()(face_tensor.cpu().clamp(0, 1))
        return face_img
    return img

# ── Batching function ─────────────────────────────────────────────────────────
def batching(images, batch_size=BATCH_SIZE):
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

# ── Load images ───────────────────────────────────────────────────────────────
query_folder   = os.path.join(DATA_FOLDER, "query")
gallery_folder = os.path.join(DATA_FOLDER, "gallery")

query_images,   query_filenames   = [], []
gallery_images, gallery_filenames = [], []

print("Loading query images...")
for filename in os.listdir(query_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        query_filenames.append(filename)
        with Image.open(os.path.join(query_folder, filename)) as img:
            img = img.convert("RGB")
            img = ImageOps.exif_transpose(img)  # rotation fix
            img = detect_and_crop_face(img)      # face crop
            query_images.append(img.copy())

print("Loading gallery images...")
for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        gallery_filenames.append(filename)
        with Image.open(os.path.join(gallery_folder, filename)) as img:
            img = img.convert("RGB")
            img = ImageOps.exif_transpose(img)  # rotation fix
            img = detect_and_crop_face(img)      # face crop
            gallery_images.append(img.copy())

print(f"Query images: {len(query_images)}")
print(f"Gallery images: {len(gallery_images)}")

# ── Extract features ──────────────────────────────────────────────────────────
print("Processing query images...")
query_features = batching(query_images)
torch.cuda.empty_cache()
print("Processing gallery images...")
gallery_features = batching(gallery_images)

# ── Normalize ─────────────────────────────────────────────────────────────────
print("Normalizing features...")
query_features   = torch.nn.functional.normalize(query_features,   p=2, dim=1)
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
result_filename = f"results/results_{MODEL_NAME}_{RESOLUTION}_{timestamp}.txt"
os.makedirs("results", exist_ok=True)
with open(result_filename, "w") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Resolution: {RESOLUTION}x{RESOLUTION}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Face detection: MTCNN\n")
    f.write(f"Rotation fix: EXIF transpose\n")
    f.write(f"Dataset: {DATA_FOLDER}\n")
    f.write(f"Top-1:  {top1/total:.2%}\n")
    f.write(f"Top-5:  {top5/total:.2%}\n")
    f.write(f"Top-10: {top10/total:.2%}\n")
print(f"Results saved to {result_filename}")

# ── Submit (uncomment on competition day) ─────────────────────────────────────
# submit(
#     results=results,
#     groupname=GROUP_NAME,
#     url="http://competition-server-url/retrieval/"
# )