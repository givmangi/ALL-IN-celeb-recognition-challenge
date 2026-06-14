"""
DINOv2 Experiment Script
=========================
Author: Kristine Paegle
Branch: kristine-dinov2

Configurable script for testing different DINOv2 configurations.
Images are loaded on-the-fly during batching to avoid memory issues.

All experiments include:
- EXIF rotation fix
- L2 normalized cosine similarity retrieval
- Local evaluation (no server needed)
- Face detection available but disabled for large datasets

Used to compare configurations (model size, resolution, face cropping)
on LFW and VGGFace2-HQ before committing to a setup for fine-tuning and
the competition pipeline. For results and conclusions from these runs,
see DINOv2_experiment_log.md.
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
MODEL_NAME   = "dinov2_vitb14"
RESOLUTION   = 224
BATCH_SIZE   = 16
DATA_FOLDER  = "/home/disi/data/vggface2_train_split/test"
GROUP_NAME   = "ALL-IN-dinov2"
FACE_CROP    = False    # set True to enable face detection (slow on large datasets)
# ─────────────────────────────────────────────────────────────────────────────

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
print(f"Model: {MODEL_NAME} | Resolution: {RESOLUTION}x{RESOLUTION} | Batch: {BATCH_SIZE} | Face crop: {FACE_CROP}")

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

# ── Load MTCNN face detector (only if needed) ─────────────────────────────────
if FACE_CROP:
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

# ── Face detection (optional) ─────────────────────────────────────────────────
def detect_and_crop_face(img):
    face_tensor = mtcnn(img)
    if face_tensor is not None:
        return T.ToPILImage()(face_tensor.cpu().clamp(0, 1))
    return img

# ── Load single image from disk ───────────────────────────────────────────────
def load_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)       # rotation fix always on
        if FACE_CROP:
            img = detect_and_crop_face(img)      # face crop optional
        return img.copy()

# ── Extract features from file paths (on the fly, no RAM overload) ────────────
def extract_features(file_paths, batch_size=BATCH_SIZE):
    features = []
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i+batch_size]
        imgs = [load_image(p) for p in batch_paths]
        inputs = torch.stack([preprocess(img) for img in imgs]).to(device)
        with torch.no_grad():
            feats = model(inputs)
            features.append(feats.cpu())  # move to CPU to free GPU memory
        if (i // batch_size) % 100 == 0:
            print(f"  {i}/{len(file_paths)} images processed...")
    return torch.cat(features, dim=0)

# ── Collect file paths ────────────────────────────────────────────────────────
query_folder   = os.path.join(DATA_FOLDER, "query")
gallery_folder = os.path.join(DATA_FOLDER, "gallery")

query_paths, query_filenames     = [], []
gallery_paths, gallery_filenames = [], []

for filename in os.listdir(query_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        query_paths.append(os.path.join(query_folder, filename))
        query_filenames.append(filename)

for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        gallery_paths.append(os.path.join(gallery_folder, filename))
        gallery_filenames.append(filename)

print(f"Query images: {len(query_paths)}")
print(f"Gallery images: {len(gallery_paths)}")

# ── Extract or load features ──────────────────────────────────────────────────
embedding_file = f"results/embeddings_{MODEL_NAME}_{RESOLUTION}_{os.path.basename(DATA_FOLDER)}.pt"
os.makedirs("results", exist_ok=True)

if os.path.exists(embedding_file):
    print(f"Loading saved embeddings from {embedding_file}...")
    saved = torch.load(embedding_file)
    query_features    = saved['query_features']
    gallery_features  = saved['gallery_features']
    query_filenames   = saved['query_filenames']
    gallery_filenames = saved['gallery_filenames']
    print("Embeddings loaded!")
else:
    print("Processing query images...")
    query_features = extract_features(query_paths)
    torch.cuda.empty_cache()
    print("Processing gallery images...")
    gallery_features = extract_features(gallery_paths)
    torch.cuda.empty_cache()

    torch.save({
        'query_features':    query_features,
        'gallery_features':  gallery_features,
        'query_filenames':   query_filenames,
        'gallery_filenames': gallery_filenames
    }, embedding_file)
    print(f"Embeddings saved to {embedding_file}")

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
    f.write(f"Face detection: {FACE_CROP}\n")
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
