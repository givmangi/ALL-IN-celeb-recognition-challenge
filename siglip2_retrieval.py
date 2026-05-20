"""
SigLIP2 face retrieval baseline.

Why SigLIP2:
  - Trained on diverse image styles (photos, illustrations, generated images)
  - Strong cross-domain semantic alignment
  - Likely more robust to real vs synthetic gap than face-recognition-specific models

Pipeline:
  1. Face detection & cropping (optional but recommended for face tasks)
  2. SigLIP2 embedding extraction
  3. L2 normalization + cosine similarity
  4. Optional: re-ranking with query expansion

Setup:
    pip install torch torchvision transformers Pillow numpy requests tqdm
    # optional, for face alignment:
    pip install facenet-pytorch
"""

import os
import json
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor

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


# ── Model loading ─────────────────────────────────────────────────────────────
def load_siglip2(model_name="google/siglip2-base-patch16-224"):
    """
    Load SigLIP2. Options (larger = better but slower):
      - google/siglip2-base-patch16-224    (~200M params)
      - google/siglip2-large-patch16-384   (~650M params)
      - google/siglip2-so400m-patch16-384  (~1.1B params, best)

    Start with base for fast iteration, switch to so400m for final submission
    if you have time and the model fits in GPU memory.
    """
    print(f"Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, processor


# ── Optional face alignment ───────────────────────────────────────────────────
def init_face_detector():
    """
    Initialize MTCNN face detector. Returns None if facenet-pytorch isn't installed,
    in which case we skip face cropping and use the full image.
    """
    try:
        from facenet_pytorch import MTCNN
        detector = MTCNN(
            image_size=224,       # output size matches SigLIP2 base input
            margin=20,            # extra padding around detected face
            keep_all=False,       # only the largest face per image
            device=device
        )
        print("Face detector loaded (MTCNN)")
        return detector
    except ImportError:
        print("facenet-pytorch not installed — skipping face alignment.")
        print("  Install with: pip install facenet-pytorch")
        return None


def crop_face(img, detector):
    """
    Crop the largest detected face from a PIL image.
    Falls back to the original image if no face is detected.
    """
    if detector is None:
        return img

    try:
        face = detector(img)  # returns a tensor of the cropped face, or None
        if face is not None:
            # MTCNN returns a tensor in [-1, 1]; convert back to PIL
            face = (face + 1) / 2  # to [0, 1]
            face = (face.clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
            return Image.fromarray(face)
    except Exception:
        pass

    return img  # fallback: use original image


# ── Dataset ───────────────────────────────────────────────────────────────────
class FaceImageDataset(Dataset):
    """
    Loads images from a flat folder.
    If a face detector is provided, crops faces during loading.
    Filenames format: identity__imagename.jpg
    """
    def __init__(self, folder, face_detector=None):
        self.folder = folder
        self.face_detector = face_detector
        self.filenames = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.folder, fname)).convert('RGB')
        # Note: face cropping happens here (in workers if num_workers>0)
        # so it doesn't block the main GPU pipeline
        img = crop_face(img, self.face_detector)
        return img, fname


def collate_pil(batch):
    """Keep PIL images as a list — processors expect this format."""
    imgs, fnames = zip(*batch)
    return list(imgs), list(fnames)


# ── Embedding extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(folder, model, processor, face_detector=None, batch_size=32):
    dataset = FaceImageDataset(folder, face_detector=face_detector)
    # Note: face_detector uses GPU, so num_workers must be 0 to avoid issues
    # If you skip face detection, you can use more workers for faster loading
    num_workers = 0 if face_detector is not None else (0 if os.name == 'nt' else 4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_pil)

    all_embeddings = []
    all_filenames = []

    for imgs, fnames in tqdm(loader, desc="  Extracting"):
        inputs = processor(images=imgs, return_tensors="pt").to(device)
        outputs = model.vision_model(**inputs)
        embs = outputs.pooler_output.cpu().numpy()
        all_embeddings.append(embs)
        all_filenames.extend(fnames)

    embeddings = np.concatenate(all_embeddings, axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return embeddings, all_filenames


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query_embeddings, query_filenames,
             gallery_embeddings, gallery_filenames, top_k=10):
    sim = query_embeddings @ gallery_embeddings.T
    results = {}
    for i, q_fname in enumerate(query_filenames):
        ranked = np.argsort(sim[i])[::-1][:top_k]
        results[q_fname] = [gallery_filenames[j] for j in ranked]
    return results, sim


# ── Query expansion (optional improvement) ────────────────────────────────────
def query_expansion(query_embs, gallery_embs, top_k_expand=3, weight=0.5):
    """
    For each query, find its top-k most similar gallery items, and create a new
    query embedding that's a weighted average of the original query and those
    nearest gallery items.

    Intuition: if the top-k results are correct, this pulls the query closer to
    the cluster of correct matches, improving retrieval on the second pass.
    Risk: if top-k results are wrong, this amplifies the error.

    Use cautiously — test on validation before applying to submission.
    """
    sim = query_embs @ gallery_embs.T
    top_indices = np.argsort(-sim, axis=1)[:, :top_k_expand]  # (n_queries, top_k_expand)

    expanded = []
    for i in range(len(query_embs)):
        neighbor_avg = gallery_embs[top_indices[i]].mean(axis=0)
        new_q = (1 - weight) * query_embs[i] + weight * neighbor_avg
        new_q = new_q / (np.linalg.norm(new_q) + 1e-8)
        expanded.append(new_q)

    return np.stack(expanded)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_local(results, query_filenames, label=""):
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

    print(f"  [{label}]")
    print(f"    Top-1:  {top1/n:.3f} ({top1}/{n})")
    print(f"    Top-5:  {top5/n:.3f} ({top5}/{n})")
    print(f"    Top-10: {top10/n:.3f} ({top10}/{n})")
    return top1/n, top5/n, top10/n


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    QUERY_FOLDER   = "test_data/query"
    GALLERY_FOLDER = "test_data/gallery"

    # Load model + face detector
    model, processor = load_siglip2()
    face_detector = init_face_detector()

    # Extract embeddings
    print("\nExtracting query embeddings...")
    q_embs, q_fnames = extract_embeddings(QUERY_FOLDER, model, processor, face_detector)
    print(f"  {q_embs.shape}")

    print("\nExtracting gallery embeddings...")
    g_embs, g_fnames = extract_embeddings(GALLERY_FOLDER, model, processor, face_detector)
    print(f"  {g_embs.shape}")

    # Free model memory before evaluation
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Evaluate baseline ────────────────────────────────────────────────────
    print("\n=== Results ===")
    results, sim = retrieve(q_embs, q_fnames, g_embs, g_fnames)
    evaluate_local(results, q_fnames, "SigLIP2 baseline")

    # ── Evaluate with query expansion ────────────────────────────────────────
    q_expanded = query_expansion(q_embs, g_embs, top_k_expand=3, weight=0.3)
    results_qe, _ = retrieve(q_expanded, q_fnames, g_embs, g_fnames)
    evaluate_local(results_qe, q_fnames, "SigLIP2 + query expansion")

    # ── For submission ───────────────────────────────────────────────────────
    # submit(results, groupname="YourGroupName", url="http://submission-url")
