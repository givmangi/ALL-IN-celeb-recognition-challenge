# ALL-IN-celeb-recognition-challenge
hi :3


WHAT TO DO AFTER OPENING WINDOWS POWERSHELL TO CONNECT TO THE AZURE LAB AND CREATE ENVIROMENT FOR THE FIRST TIME:
ssh -p 5027 disi@lab-xxxxxx.westeurope.cloudapp.azure.com      ## YOUR AZURE LAB COMMAND FROM THE WEBSITE

conda create -n mlproject python=3.10
conda activate mlproject

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install git+https://github.com/openai/CLIP.git scikit-learn pillow kagglehub

### CHECK IF IT WORKS:
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

### NOW TO CREATE THE SCRIPT TO DOWNLOAD THE DATASET:
nano script1.py

#### PASTE THIS INSIDE
```
# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
#
# print("Path to dataset files:", path)

import os
import shutil
from collections import defaultdict


# ── Point to your already-downloaded dataset ──────────────────────────────────
kaggle_path = r"/home/disi/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4"

# find the actual image folder (it's nested inside)
lfw_root = None
for root, dirs, files in os.walk(kaggle_path):
    if any(os.path.isdir(os.path.join(root, d)) for d in dirs):
        subdirs = [d for d in dirs if not d.startswith('.')]
        if len(subdirs) > 10:  # the identity folders level
            lfw_root = root
            break

print(f"Found LFW root at: {lfw_root}")

# ── Filter identities with at least 2 images ─────────────────────────────────
identity_images = defaultdict(list)
for identity in os.listdir(lfw_root):
    identity_path = os.path.join(lfw_root, identity)
    if not os.path.isdir(identity_path):
        continue
    images = [f for f in os.listdir(identity_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) >= 2:
        identity_images[identity] = images

print(f"Identities with at least 2 images: {len(identity_images)}")

# ── Split into query and gallery ──────────────────────────────────────────────
query_folder = "test_data/query"
gallery_folder = "test_data/gallery"
os.makedirs(query_folder, exist_ok=True)
os.makedirs(gallery_folder, exist_ok=True)
#
# for identity, images in identity_images.items():
#     # first image goes to query, the rest go to gallery
#     src = os.path.join(lfw_root, identity, images[0])
#     shutil.copy(src, os.path.join(query_folder, f"{identity}__{images[0]}"))
#
#     for img in images[1:]:
#         src = os.path.join(lfw_root, identity, img)
#         shutil.copy(src, os.path.join(gallery_folder, f"{identity}__{img}"))

import random
random.seed(1)  # everyone on the team uses the same seed

for identity, images in identity_images.items():
    images_shuffled = images.copy()
    random.shuffle(images_shuffled)

    # first image goes to query, the rest go to gallery
    src = os.path.join(lfw_root, identity, images_shuffled[0])
    shutil.copy(src, os.path.join(query_folder, f"{identity}__{images_shuffled[0]}"))

    for img in images_shuffled[1:]:
        src = os.path.join(lfw_root, identity, img)
        shutil.copy(src, os.path.join(gallery_folder, f"{identity}__{img}"))


print(f"Query images:   {len(os.listdir(query_folder))}")
print(f"Gallery images: {len(os.listdir(gallery_folder))}")
print("Done. Run your pipeline now with data_folder = 'test_data'")
```


#### THEN THE SECOND SCRIPT (MODEL):

nano all-in.py

#### THEN INSIDE:
```
import os
import json
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import clip

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
    res = {
        'groupname': groupname,
        'images': results
    }
    response = requests.post(url, json.dumps(res))
    try:
        result = json.loads(response.text)
        print(f"Accuracy: {result['accuracy']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")


# ── Load CLIP ─────────────────────────────────────────────────────────────────
print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


# ── Image loading ─────────────────────────────────────────────────────────────
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(VALID_EXTENSIONS):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            filenames.append(filename)
    return images, filenames


# ── Embedding extraction ──────────────────────────────────────────────────────
def extract_embeddings(images, batch_size=32):
    all_features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            features = model.encode_image(inputs)
        all_features.append(features)
    return torch.cat(all_features, dim=0)


# ── Data folders ──────────────────────────────────────────────────────────────
# For LFW testing: point both query and gallery at the same LFW root,
# then split manually. For the actual competition, use the real folders.
data_folder = "test_data"       # change this
query_folder = os.path.join(data_folder, "query")
gallery_folder = os.path.join(data_folder, "gallery")

print("Loading images...")
query_images, query_filenames = load_images_from_folder(query_folder)
gallery_images, gallery_filenames = load_images_from_folder(gallery_folder)
print(f"Query images: {len(query_images)}")
print(f"Gallery images: {len(gallery_images)}")


# ── Extract and normalize embeddings ─────────────────────────────────────────
print("Extracting query embeddings...")
query_features = extract_embeddings(query_images)
print("Extracting gallery embeddings...")
gallery_features = extract_embeddings(gallery_images)

query_features = F.normalize(query_features, p=2, dim=1)
gallery_features = F.normalize(gallery_features, p=2, dim=1)


# ── KNN retrieval ─────────────────────────────────────────────────────────────
print("Building KNN index...")
gallery_np = gallery_features.cpu().numpy()
query_np = query_features.cpu().numpy()

knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
knn.fit(gallery_np)

print("Retrieving top 10 matches per query...")
distances, indices = knn.kneighbors(query_np)


# ── Build results ─────────────────────────────────────────────────────────────
results = {}
for i, query_filename in enumerate(query_filenames):
    results[query_filename] = [gallery_filenames[j] for j in indices[i]]


# ── Optional: print top matches ───────────────────────────────────────────────
for query_filename, matches in list(results.items())[:3]:  # print first 3
    print(f"\nQuery: {query_filename}")
    for rank, match in enumerate(matches, 1):
        print(f"  {rank}: {match}")


# ── Submit ────────────────────────────────────────────────────────────────────
# submit(
#     results=results,
#     groupname="all-in",
#     url="http://localhost:3001/retrieval/"   # change to real URL
# )

# ── Local evaluation (no server needed) ──────────────────────────────────────
top1, top5, top10, total = 0, 0, 0, 0

for i, query_filename in enumerate(query_filenames):
    query_identity = query_filename.split("__")[0]
    retrieved = [gallery_filenames[j].split("__")[0] for j in indices[i]]

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
```


#### ALL DONE, NOW RUN:

python script1.py

python all-in.py
