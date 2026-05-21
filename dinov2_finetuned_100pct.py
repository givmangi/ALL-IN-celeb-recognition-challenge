"""
DINOv2 ViT-B/14 - Final Competition Script (336x336 resolution)
================================================================
Author: Kristine
Branch: kristine-dinov2

Model: DINOv2 ViT-B/14 — self-supervised vision transformer by Meta AI.
Used as a pretrained feature extractor, optionally fine-tuned with triplet loss.

Key features:
- On-the-fly image loading (no RAM overload)
- EXIF rotation fix
- Optional MTCNN face detection cropping (recommended for small datasets)
- Embedding saving/loading to avoid recomputation
- Local evaluation + competition server submission

Results (no fine-tuning, no face crop):
- LFW:         Top-1: 18.63%, Top-5: 27.62%, Top-10: 31.55% (336)
- VGGFace2-HQ: Top-1: 32.44%, Top-5: 46.58%, Top-10: 53.59% (336)
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
BATCH_SIZE   = 8
FACE_CROP    = False      # True for competition (~3000 images), False for large datasets
SAVE_EMBEDDINGS = True   # save embeddings to avoid recomputing
LOAD_EMBEDDINGS = False  # set True to load saved embeddings instead of recomputing
GROUP_NAME   = "ALL-IN"

# Update this path on competition day:
DATA_FOLDER  = "/home/disi/data/vggface2_train_split/test"
# ─────────────────────────────────────────────────────────────────────────────

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
print(f"Model: {MODEL_NAME} | Resolution: {RESOLUTION} | Face crop: {FACE_CROP}")

def submit_and_log(res_dict, model_name, group_name="ALL-IN", url="", log_file="/home/disi/logs/submission_log.txt"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Submitting '{model_name}' as '{group_name}'...")
    
    payload = json.dumps({
        "groupname": group_name,
        "images": res_dict      # ← keeping 'images' as per professor's baseline
    })
    
    server_response_text = ""
    try:
        response = requests.post(url, payload, timeout=30)
        response.raise_for_status()
        server_response_text = response.text.strip()
        print(f"\nSUCCESS! Server responded: {server_response_text}\n")
    except requests.exceptions.HTTPError as http_err:
        server_response_text = f"HTTP Error: {http_err} - {response.text}"
        print(f"\nERROR: {server_response_text}\n")
    except Exception as err:
        server_response_text = f"Connection FAILED: {err}"
        print(f"\nERROR: {server_response_text}\n")

    log_entry = (
        f"Time:\t {timestamp}\n"
        f"Group:\t {group_name}\n"
        f"Model:\t {model_name}\n"
        f"Result:\t {server_response_text}\n"
        f"{'-'*60}\n"
    )
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"Log saved to: {log_file}")
    except Exception as e:
        print(f"Warning: Could not write log. Error: {e}")

# ── Load DINOv2 ───────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)

# Load fine-tuned weights if available
checkpoint_path = "checkpoints/best_model_100pct.pth"
if os.path.exists(checkpoint_path):
    print(f"Loading fine-tuned weights from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Fine-tuned weights loaded!")
else:
    print("No fine-tuned weights found, using pretrained model.")

model = model.to(device)
model.eval()

# ── Load MTCNN face detector ──────────────────────────────────────────────────
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

# ── Face detection ────────────────────────────────────────────────────────────
def detect_and_crop_face(img):
    face_tensor = mtcnn(img)
    if face_tensor is not None:
        return T.ToPILImage()(face_tensor.cpu().clamp(0, 1))
    return img

# ── Load single image ─────────────────────────────────────────────────────────
def load_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)
        if FACE_CROP:
            img = detect_and_crop_face(img)
        return img.copy()

# ── Extract features on the fly ───────────────────────────────────────────────
def extract_features(file_paths, batch_size=BATCH_SIZE):
    features = []
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i+batch_size]
        imgs = [load_image(p) for p in batch_paths]
        inputs = torch.stack([preprocess(img) for img in imgs]).to(device)
        with torch.no_grad():
            feats = model(inputs)
            features.append(feats.cpu())
        if (i // batch_size) % 50 == 0:
            print(f"  {min(i+batch_size, len(file_paths))}/{len(file_paths)} images processed...")
    return torch.cat(features, dim=0)

# ── Collect file paths ────────────────────────────────────────────────────────
query_folder   = os.path.join(DATA_FOLDER, "query")
gallery_folder = os.path.join(DATA_FOLDER, "gallery")

query_paths, query_filenames     = [], []
gallery_paths, gallery_filenames = [], []

for filename in sorted(os.listdir(query_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        query_paths.append(os.path.join(query_folder, filename))
        query_filenames.append(filename)

for filename in sorted(os.listdir(gallery_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        gallery_paths.append(os.path.join(gallery_folder, filename))
        gallery_filenames.append(filename)

print(f"Query images: {len(query_paths)}")
print(f"Gallery images: {len(gallery_paths)}")

# ── Extract or load features ──────────────────────────────────────────────────
if LOAD_EMBEDDINGS and os.path.exists("query_embeddings.pt"):
    print("Loading saved embeddings...")
    query_features   = torch.load("query_embeddings.pt")
    gallery_features = torch.load("gallery_embeddings.pt")
    print("Embeddings loaded!")
else:
    print("Processing query images...")
    query_features = extract_features(query_paths)
    torch.cuda.empty_cache()
    print("Processing gallery images...")
    gallery_features = extract_features(gallery_paths)
    torch.cuda.empty_cache()

    if SAVE_EMBEDDINGS:
        torch.save(query_features,   "query_embeddings.pt")
        torch.save(gallery_features, "gallery_embeddings.pt")
        print("Embeddings saved!")

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
os.makedirs("results", exist_ok=True)
result_filename = f"results/results_final_{MODEL_NAME}_{RESOLUTION}_{timestamp}.txt"
with open(result_filename, "w") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Resolution: {RESOLUTION}x{RESOLUTION}\n")
    f.write(f"Face detection: {FACE_CROP}\n")
    f.write(f"Fine-tuned: {os.path.exists(checkpoint_path)}\n")
    f.write(f"Dataset: {DATA_FOLDER}\n")
    f.write(f"--- Local Evaluation (VGGFace2 format) ---\n")
    f.write(f"Top-1:  {top1/total:.2%}\n")
    f.write(f"Top-5:  {top5/total:.2%}\n")
    f.write(f"Top-10: {top10/total:.2%}\n")
    f.write(f"--- Server Score (competition) ---\n")
print(f"Results saved to {result_filename}")

# ── Submit (uncomment on competition day) ─────────────────────────────────────
submit_and_log(
    res_dict=results,
    model_name="DINOv2_finetuned_100pct",  # change per script
    group_name=GROUP_NAME,
    url="http://competition-server-url/retrieval/",
    log_file="/home/disi/logs/submission_log.txt"
)