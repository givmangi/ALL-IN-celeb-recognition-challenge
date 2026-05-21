import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import clip
import json
import requests
import torchvision.transforms as T
import datetime

# ══════════════════════════════════════════════════════════════════════════════
# COMPETITION DAY SETTINGS - only touch these
# ══════════════════════════════════════════════════════════════════════════════

MODE = "competition"  # "local" for testing, "competition" for submission

# competition mode settings
QUERY_FOLDER = "/home/disi/test/query"                      # un-augmented queries
GALLERY_FOLDER = "/home/disi/augmented_gallery"             # GFPGAN-augmented gallery
SUBMIT_URL = "xxx"                                          # change on competition day
GROUP_NAME = "ALL-IN"

# local evaluation mode settings
LOCAL_TEST_FOLDER = "vggface2_train_split/val"

# which weights to use
WEIGHTS_PATH = "clip_competition_epoch5.pt"

# tag for the submission log so we can tell variants apart
MODEL_TAG = "CLIP_ViT-L-14_clip_competition_epoch5_AUGgallery"

# ══════════════════════════════════════════════════════════════════════════════

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading fine-tuned CLIP model...")
model, _ = clip.load("ViT-L/14", device=device)
model = model.float()

if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"Loaded weights from {WEIGHTS_PATH}")
else:
    print(f"WARNING: {WEIGHTS_PATH} not found, running zero-shot CLIP (no fine-tuning)")

model.eval()

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# ── Transform ─────────────────────────────────────────────────────────────────
n_px = 224
clean_transform = T.Compose([
    T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(n_px),
    T.ToTensor(),
    T.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])


# ── Embed a flat folder ───────────────────────────────────────────────────────
def extract_embeddings_flat(folder, batch_size=64):
    filenames = [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS)]
    all_features = []

    for i in range(0, len(filenames), batch_size):
        batch = filenames[i:i + batch_size]
        imgs = []
        for f in batch:
            try:
                img = Image.open(os.path.join(folder, f)).convert("RGB")
                imgs.append(clean_transform(img))
            except Exception as e:
                print(f"Skipping {f}: {e}")
        if not imgs:
            continue
        inputs = torch.stack(imgs).to(device)
        with torch.no_grad():
            features = model.encode_image(inputs)
        features = F.normalize(features, p=2, dim=1)
        all_features.append(features.cpu().numpy())
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(filenames)}...")

    return np.vstack(all_features), filenames


# ── Embed a folder organized as identity/image.jpg subfolders ─────────────────
def extract_embeddings_subfolders(folder, batch_size=64):
    filepaths = []
    identities = []
    for identity in os.listdir(folder):
        identity_path = os.path.join(folder, identity)
        if not os.path.isdir(identity_path):
            continue
        for f in os.listdir(identity_path):
            if f.lower().endswith(VALID_EXTENSIONS):
                filepaths.append(os.path.join(identity_path, f))
                identities.append(identity)

    all_features = []
    for i in range(0, len(filepaths), batch_size):
        batch = filepaths[i:i + batch_size]
        imgs = []
        for path in batch:
            try:
                img = Image.open(path).convert("RGB")
                imgs.append(clean_transform(img))
            except Exception as e:
                print(f"Skipping {path}: {e}")
        if not imgs:
            continue
        inputs = torch.stack(imgs).to(device)
        with torch.no_grad():
            features = model.encode_image(inputs)
        features = F.normalize(features, p=2, dim=1)
        all_features.append(features.cpu().numpy())
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(filepaths)}...")

    return np.vstack(all_features), filepaths, identities


# ── Submission + Log ──────────────────────────────────────────────────────────
def submit_and_log(res_dict, model_name, group_name="ALL-IN", url="", log_file="submission_log.txt"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Connecting to server to submit '{group_name}'...")

    payload = {
        "groupname": group_name,
        "images": res_dict
    }
    server_response_text = ""
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        server_response_text = response.text.strip()
        print(f"\nSUCCESS, Server responded: {server_response_text}\n")
    except requests.exceptions.HTTPError as http_err:
        server_response_text = f"HTTP Error: {http_err} - Server said: {response.text}"
        print(f"\nERROR: {server_response_text}\n")
    except Exception as err:
        server_response_text = f"Connection FAILED: {err}"
        print(f"\nERROR: {server_response_text}\n")

    log_entry = (
        f"Time:\t {timestamp}\n"
        f"Group:\t {group_name}\n"
        f"Model:\t {model_name}\n"
        f"Result:\t {server_response_text}\n"
        f"{'-' * 60}\n"
    )
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"Log appended to: {log_file}")
    except Exception as e:
        print(f"Warning: Could not write to log file. Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
if MODE == "competition":
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n--- COMPETITION MODE (augmented gallery) ---")

    # Sanity check that augmented gallery has same images as original
    print(f"Query folder:   {QUERY_FOLDER}")
    print(f"Gallery folder: {GALLERY_FOLDER}")
    n_query   = len([f for f in os.listdir(QUERY_FOLDER)   if f.lower().endswith(VALID_EXTENSIONS)])
    n_gallery = len([f for f in os.listdir(GALLERY_FOLDER) if f.lower().endswith(VALID_EXTENSIONS)])
    print(f"  → {n_query} queries, {n_gallery} gallery images")

    print("\nExtracting query embeddings...")
    query_features, query_filenames = extract_embeddings_flat(QUERY_FOLDER)
    print(f"Queries: {len(query_filenames)}")

    print("Extracting gallery embeddings...")
    gallery_features, gallery_filenames = extract_embeddings_flat(GALLERY_FOLDER)
    print(f"Gallery: {len(gallery_filenames)}")

    print("Building KNN index...")
    knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    knn.fit(gallery_features)

    print("Retrieving top 10 matches...")
    distances, indices = knn.kneighbors(query_features)

    results = {}
    for i, query_filename in enumerate(query_filenames):
        results[query_filename] = [gallery_filenames[j] for j in indices[i]]

    print(f"\nSample result:")
    first_query = query_filenames[0]
    print(f"  Query: {first_query}")
    for rank, match in enumerate(results[first_query], 1):
        print(f"    {rank}: {match}")

    print("\nSubmitting to server...")
    submit_and_log(
        res_dict=results,
        model_name=MODEL_TAG,
        group_name=GROUP_NAME,
        url=SUBMIT_URL
    )

# ══════════════════════════════════════════════════════════════════════════════
else:  # MODE == "local"
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n--- LOCAL EVALUATION MODE ---")
    print("Loading local test data...")
    _, filepaths, identities = extract_embeddings_subfolders(LOCAL_TEST_FOLDER)

    identity_dict = {}
    for path, identity in zip(filepaths, identities):
        identity_dict.setdefault(identity, []).append(path)

    query_paths, query_ids = [], []
    gallery_paths, gallery_ids = [], []
    for identity, paths in identity_dict.items():
        if len(paths) >= 2:
            query_paths.append(paths[0])
            query_ids.append(identity)
            gallery_paths.extend(paths[1:])
            gallery_ids.extend([identity] * (len(paths) - 1))

    print(f"Queries: {len(query_paths)} | Gallery: {len(gallery_paths)}")

    def embed_paths(paths):
        all_features = []
        for i in range(0, len(paths), 64):
            batch = paths[i:i + 64]
            imgs = []
            for path in batch:
                try:
                    imgs.append(clean_transform(Image.open(path).convert("RGB")))
                except:
                    pass
            if not imgs:
                continue
            inputs = torch.stack(imgs).to(device)
            with torch.no_grad():
                features = model.encode_image(inputs)
            all_features.append(F.normalize(features, p=2, dim=1).cpu().numpy())
        return np.vstack(all_features)

    query_features = embed_paths(query_paths)
    gallery_features = embed_paths(gallery_paths)

    knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    knn.fit(gallery_features)
    distances, indices = knn.kneighbors(query_features)

    top1, top5, top10, total = 0, 0, 0, 0
    for i in range(len(query_paths)):
        q_id = query_ids[i]
        retrieved_ids = [gallery_ids[j] for j in indices[i]]
        total += 1
        if q_id == retrieved_ids[0]:   top1  += 1
        if q_id in retrieved_ids[:5]:  top5  += 1
        if q_id in retrieved_ids[:10]: top10 += 1

    print("\n==================================================")
    print(f"Top-1  accuracy: {top1 / total:.2%}")
    print(f"Top-5  accuracy: {top5 / total:.2%}")
    print(f"Top-10 accuracy: {top10 / total:.2%}")
    print("==================================================")
