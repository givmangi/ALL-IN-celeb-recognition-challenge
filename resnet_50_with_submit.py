import os
import json
import requests
import datetime
import numpy as np
from PIL import Image
from collections import defaultdict
import kagglehub

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

MODE = "competition"  # "local" for LFW testing, "competition" for actual submission

# competition mode settings
QUERY_FOLDER   = "/home/disi/test/query"
GALLERY_FOLDER = "/home/disi/test/gallery"
SUBMIT_URL     = "http://videosim.disi.unitn.it:3001/retrieval/"        # paste the real submission URL here
GROUP_NAME     = "ALL-IN"
MODEL_TAG      = "ResNet50_ImageNet_baseline"   # shown in submission log

# ══════════════════════════════════════════════════════════════════════════════

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# ── Submission + Log ──────────────────────────────────────────────────────────
def submit_and_log(res_dict, model_name, group_name="ALL-IN", url="", log_file="submission_log.txt"):
    """
    Submits results to the server and appends an entry to a local log file.
    """
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


# ── ResNet-50 as feature extractor ───────────────────────────────────────────
def build_model():
    """
    Load pretrained ResNet-50 and remove the final classification head.
    The output is a 2048-d embedding vector per image.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


# ── Image preprocessing ───────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ── Dataset class ─────────────────────────────────────────────────────────────
class ImageFolderFlat(Dataset):
    """Loads all images from a flat folder (no subfolders)."""
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.filenames = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.folder, fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, fname


# ── Embedding extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(folder, model, transform, batch_size=64):
    """
    Run all images in a folder through the model and return:
      - embeddings: np.array of shape (N, 2048), L2-normalized
      - filenames:  list of filenames in the same order
    """
    dataset = ImageFolderFlat(folder, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    all_filenames = []

    for imgs, fnames in loader:
        imgs = imgs.to(device)
        embs = model(imgs)
        embs = embs.cpu().numpy()
        all_embeddings.append(embs)
        all_filenames.extend(fnames)

    embeddings = np.concatenate(all_embeddings, axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return embeddings, all_filenames


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query_embeddings, query_filenames,
             gallery_embeddings, gallery_filenames,
             top_k=10):
    """For each query, return top_k most similar gallery filenames."""
    sim = query_embeddings @ gallery_embeddings.T
    results = {}
    for i, q_fname in enumerate(query_filenames):
        ranked_indices = np.argsort(sim[i])[::-1][:top_k]
        results[q_fname] = [gallery_filenames[j] for j in ranked_indices]
    return results


# ── Local evaluation (only used in MODE="local") ──────────────────────────────
def evaluate_local(results, query_filenames):
    """Compute Top-1, Top-5, Top-10 accuracy on LFW-style filenames."""
    def get_identity(fname):
        return fname.split('__')[0]

    top1, top5, top10 = 0, 0, 0
    n = len(query_filenames)
    for q_fname in query_filenames:
        q_id = get_identity(q_fname)
        retrieved_ids = [get_identity(g) for g in results[q_fname]]
        if q_id in retrieved_ids[:1]:  top1  += 1
        if q_id in retrieved_ids[:5]:  top5  += 1
        if q_id in retrieved_ids[:10]: top10 += 1

    print(f"Evaluated on {n} queries")
    print(f"  Top-1  accuracy: {top1/n:.3f} ({top1}/{n})")
    print(f"  Top-5  accuracy: {top5/n:.3f} ({top5}/{n})")
    print(f"  Top-10 accuracy: {top10/n:.3f} ({top10}/{n})")
    return top1/n, top5/n, top10/n


# ── LFW data preparation (only used in MODE="local") ──────────────────────────
def prepare_lfw_data(kaggle_path, query_folder="test_data/query", gallery_folder="test_data/gallery", seed=1):
    """Splits LFW into query/gallery folders for local testing."""
    import shutil
    import random

    lfw_root = None
    for root, dirs, files in os.walk(kaggle_path):
        subdirs = [d for d in dirs if not d.startswith('.')]
        if len(subdirs) > 10:
            lfw_root = root
            break

    if lfw_root is None:
        raise RuntimeError(f"Could not find LFW root under {kaggle_path}")
    print(f"Found LFW root at: {lfw_root}")

    identity_images = defaultdict(list)
    for identity in os.listdir(lfw_root):
        identity_path = os.path.join(lfw_root, identity)
        if not os.path.isdir(identity_path):
            continue
        images = [f for f in os.listdir(identity_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) >= 2:
            identity_images[identity] = images
    print(f"Identities with at least 2 images: {len(identity_images)}")

    os.makedirs(query_folder, exist_ok=True)
    os.makedirs(gallery_folder, exist_ok=True)

    if os.listdir(query_folder) and os.listdir(gallery_folder):
        print("Query and gallery folders already populated, skipping copy.")
        return query_folder, gallery_folder

    random.seed(seed)
    for identity, images in identity_images.items():
        images_shuffled = images.copy()
        random.shuffle(images_shuffled)
        src = os.path.join(lfw_root, identity, images_shuffled[0])
        shutil.copy(src, os.path.join(query_folder, f"{identity}__{images_shuffled[0]}"))
        for img in images_shuffled[1:]:
            src = os.path.join(lfw_root, identity, img)
            shutil.copy(src, os.path.join(gallery_folder, f"{identity}__{img}"))

    print(f"Query images:   {len(os.listdir(query_folder))}")
    print(f"Gallery images: {len(os.listdir(gallery_folder))}")
    return query_folder, gallery_folder


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if MODE == "competition":
        # ── COMPETITION MODE ─────────────────────────────────────────────────
        print("\n--- COMPETITION MODE ---")
        print(f"Query folder:   {QUERY_FOLDER}")
        print(f"Gallery folder: {GALLERY_FOLDER}")

        print("\nBuilding model...")
        model = build_model()

        print("Extracting query embeddings...")
        q_embs, q_fnames = extract_embeddings(QUERY_FOLDER, model, preprocess)
        print(f"  {len(q_fnames)} query images → embeddings shape: {q_embs.shape}")

        print("Extracting gallery embeddings...")
        g_embs, g_fnames = extract_embeddings(GALLERY_FOLDER, model, preprocess)
        print(f"  {len(g_fnames)} gallery images → embeddings shape: {g_embs.shape}")

        print("\nRunning retrieval...")
        results = retrieve(q_embs, q_fnames, g_embs, g_fnames, top_k=10)

        print(f"\nSample result:")
        first_query = q_fnames[0]
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

    else:
        # ── LOCAL TEST MODE (LFW) ─────────────────────────────────────────────
        print("\n--- LOCAL MODE (LFW) ---")
        KAGGLE_PATH = kagglehub.dataset_download("jessicali9530/lfw-dataset")
        print("Path to dataset files:", KAGGLE_PATH)

        LOCAL_QUERY   = "test_data/query"
        LOCAL_GALLERY = "test_data/gallery"

        print("Preparing data...")
        LOCAL_QUERY, LOCAL_GALLERY = prepare_lfw_data(KAGGLE_PATH, LOCAL_QUERY, LOCAL_GALLERY)

        print("Building model...")
        model = build_model()

        print("Extracting query embeddings...")
        q_embs, q_fnames = extract_embeddings(LOCAL_QUERY, model, preprocess)
        print(f"  {len(q_fnames)} query images → embeddings shape: {q_embs.shape}")

        print("Extracting gallery embeddings...")
        g_embs, g_fnames = extract_embeddings(LOCAL_GALLERY, model, preprocess)
        print(f"  {len(g_fnames)} gallery images → embeddings shape: {g_embs.shape}")

        print("Running retrieval...")
        results = retrieve(q_embs, q_fnames, g_embs, g_fnames, top_k=10)

        print("\nLocal evaluation:")
        evaluate_local(results, q_fnames)
