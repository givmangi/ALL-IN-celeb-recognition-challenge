import os
import json
import requests
import numpy as np
from PIL import Image
from collections import defaultdict
import kagglehub

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# ── Submission helper ─────────────────────────────────────────────────────────
def submit(results, groupname, url):
    res = {'groupname': groupname, 'images': results}
    response = requests.post(url, json.dumps(res))
    try:
        result = json.loads(response.text)
        print(f"Accuracy: {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


# ── ResNet-50 as feature extractor ───────────────────────────────────────────
def build_model():
    """
    Load pretrained ResNet-50 and remove the final classification head.
    The output is a 2048-d embedding vector per image.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove the last fully-connected layer (the classifier)
    # nn.Identity() is a no-op: it just passes the input through unchanged
    model.fc = nn.Identity()
    model.eval()
    return model.to(device)


# ── Image preprocessing ───────────────────────────────────────────────────────
# ResNet was trained on ImageNet with these exact mean/std values.
# We must apply the same normalization at inference time.
preprocess = transforms.Compose([
    transforms.Resize(256),          # resize shorter side to 256
    transforms.CenterCrop(224),      # crop to 224×224 (ResNet input size)
    transforms.ToTensor(),           # convert PIL image to tensor [0,1]
    transforms.Normalize(            # normalize to match ImageNet training
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ── Dataset class ─────────────────────────────────────────────────────────────
class ImageFolderFlat(Dataset):
    """
    Loads all images from a flat folder (no subfolders).
    Filenames are expected in the format: identity__imagename.jpg
    (as produced by the data preparation script above).
    """
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
@torch.no_grad()  # disable gradient computation — we are not training
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
        embs = model(imgs)             # shape: (batch, 2048)
        embs = embs.cpu().numpy()
        all_embeddings.append(embs)
        all_filenames.extend(fnames)

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 2048)

    # L2 normalization: divide each vector by its norm
    # This makes cosine similarity equivalent to dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # +1e-8 to avoid division by zero

    return embeddings, all_filenames


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query_embeddings, query_filenames,
             gallery_embeddings, gallery_filenames,
             top_k=10):
    """
    For each query, compute cosine similarity against all gallery images
    and return the top_k most similar gallery filenames.

    Cosine similarity matrix: (n_queries, n_gallery)
    Since embeddings are L2-normalized, this is just a matrix multiplication.
    """
    # sim[i, j] = cosine similarity between query i and gallery image j
    sim = query_embeddings @ gallery_embeddings.T  # (n_queries, n_gallery)

    results = {}
    for i, q_fname in enumerate(query_filenames):
        # argsort gives ascending order; [::-1] reverses to descending
        ranked_indices = np.argsort(sim[i])[::-1][:top_k]
        results[q_fname] = [gallery_filenames[j] for j in ranked_indices]

    return results


# ── Local evaluation ──────────────────────────────────────────────────────────
def evaluate_local(results, query_filenames):
    """
    Compute Top-1, Top-5, Top-10 accuracy locally.

    Since filenames are formatted as 'identity__imagename.jpg',
    we extract the identity from the filename and check if any
    retrieved gallery image shares the same identity.
    """
    def get_identity(fname):
        # 'George_W_Bush__George_W_Bush_0001.jpg' → 'George_W_Bush'
        return fname.split('__')[0]

    top1, top5, top10 = 0, 0, 0
    n = len(query_filenames)

    for q_fname in query_filenames:
        q_identity = get_identity(q_fname)
        retrieved = results[q_fname]
        retrieved_identities = [get_identity(g) for g in retrieved]

        if q_identity in retrieved_identities[:1]:
            top1 += 1
        if q_identity in retrieved_identities[:5]:
            top5 += 1
        if q_identity in retrieved_identities[:10]:
            top10 += 1

    print(f"Evaluated on {n} queries")
    print(f"  Top-1  accuracy: {top1/n:.3f} ({top1}/{n})")
    print(f"  Top-5  accuracy: {top5/n:.3f} ({top5}/{n})")
    print(f"  Top-10 accuracy: {top10/n:.3f} ({top10}/{n})")
    return top1/n, top5/n, top10/n


# ── Data loading ─────────────────────────────────────────────────────────────
def prepare_lfw_data(kaggle_path, query_folder="test_data/query", gallery_folder="test_data/gallery", seed=1):
    """
    Finds the LFW root directory, filters identities with at least 2 images,
    and splits them into query and gallery folders.
    """
    import shutil
    import random

    # Find the actual image folder (nested inside kaggle cache)
    lfw_root = None
    for root, dirs, files in os.walk(kaggle_path):
        subdirs = [d for d in dirs if not d.startswith('.')]
        if len(subdirs) > 10:
            lfw_root = root
            break

    if lfw_root is None:
        raise RuntimeError(f"Could not find LFW root under {kaggle_path}")
    print(f"Found LFW root at: {lfw_root}")

    # Filter identities with at least 2 images
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

    # Create query and gallery folders
    os.makedirs(query_folder, exist_ok=True)
    os.makedirs(gallery_folder, exist_ok=True)

    # Skip if already populated
    if os.listdir(query_folder) and os.listdir(gallery_folder):
        print("Query and gallery folders already populated, skipping copy.")
        return query_folder, gallery_folder

    random.seed(seed)
    for identity, images in identity_images.items():
        images_shuffled = images.copy()
        random.shuffle(images_shuffled)

        # First image → query
        src = os.path.join(lfw_root, identity, images_shuffled[0])
        shutil.copy(src, os.path.join(query_folder, f"{identity}__{images_shuffled[0]}"))

        # Rest → gallery
        for img in images_shuffled[1:]:
            src = os.path.join(lfw_root, identity, img)
            shutil.copy(src, os.path.join(gallery_folder, f"{identity}__{img}"))

    print(f"Query images:   {len(os.listdir(query_folder))}")
    print(f"Gallery images: {len(os.listdir(gallery_folder))}")
    return query_folder, gallery_folder


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Download latest version
    KAGGLE_PATH = kagglehub.dataset_download("jessicali9530/lfw-dataset")

    print("Path to dataset files:", KAGGLE_PATH)
    QUERY_FOLDER   = "test_data/query"
    GALLERY_FOLDER = "test_data/gallery"

    print("Preparing data...")
    QUERY_FOLDER, GALLERY_FOLDER = prepare_lfw_data(KAGGLE_PATH, QUERY_FOLDER, GALLERY_FOLDER)

    print("Building model...")
    model = build_model()

    print("Extracting query embeddings...")
    q_embs, q_fnames = extract_embeddings(QUERY_FOLDER, model, preprocess)
    print(f"  {len(q_fnames)} query images → embeddings shape: {q_embs.shape}")

    print("Extracting gallery embeddings...")
    g_embs, g_fnames = extract_embeddings(GALLERY_FOLDER, model, preprocess)
    print(f"  {len(g_fnames)} gallery images → embeddings shape: {g_embs.shape}")

    print("Running retrieval...")
    results = retrieve(q_embs, q_fnames, g_embs, g_fnames, top_k=10)

    print("\nLocal evaluation:")
    evaluate_local(results, q_fnames)

    # ── Uncomment to submit ────────────────────────────────────────────────────
    # submit(results, groupname="YourGroupName", url="http://your-submission-url")