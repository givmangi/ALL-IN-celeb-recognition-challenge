"""
DINOv2 Baseline - First Experiment
===================================
Author: Kristine
Branch: kristine-dinov2

This is my first attempt at using DINOv2 (Vision Transformer, ViT-B/14)
for the celebrity retrieval competition.

The model is used as a pretrained feature extractor without any fine-tuning.
DINOv2 is self-supervised, meaning it was trained without labels, which
hopefully makes it robust to the domain shift between real query images
and synthetic gallery images.

Approach:
- Extract 768-dimensional embeddings from DINOv2 for all query and gallery images
- Normalize embeddings to unit length
- Rank gallery images by cosine similarity to each query
"""

import os
import json
import datetime
from PIL import Image
import requests
import torch
import torchvision.transforms as T

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Submit function
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

# Load DINOv2
print("Loading DINOv2 model...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model = model.to(device)
model.eval()

# Preprocessing
preprocess = T.Compose([
    T.Resize(336), #changed from 256
    T.CenterCrop(336), #changed from 224
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# Batching function
def batching(images, batch_size=8): #changed from 16
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

# Paths
data_folder = "/home/disi/data/lfw_split"
query_folder = os.path.join(data_folder, "query")
gallery_folder = os.path.join(data_folder, "gallery")

# Load images - using 'with' to avoid too many open files
query_images, query_filenames = [], []
gallery_images, gallery_filenames = [], []

for filename in os.listdir(query_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        query_filenames.append(filename)
        with Image.open(os.path.join(query_folder, filename)) as img:
            query_images.append(img.convert("RGB").copy())

for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        gallery_filenames.append(filename)
        with Image.open(os.path.join(gallery_folder, filename)) as img:
            gallery_images.append(img.convert("RGB").copy())

print(f"Query images: {len(query_images)}")
print(f"Gallery images: {len(gallery_images)}")

# Extract features
print("Processing query images...")
query_features = batching(query_images, batch_size=8) #changed from 16
print("Processing gallery images...")
gallery_features = batching(gallery_images, batch_size=8) #changed from 16

# Normalize
print("Normalizing features...")
query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)

# Compute similarity
print("Computing similarity matrix...")
similarity_matrix = torch.matmul(query_features, gallery_features.T)

# Get top 10 matches
top_k = 10
_, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)

# Build results
results = {}
for i, query_filename in enumerate(query_filenames):
    results[query_filename] = [
        gallery_filenames[idx] for idx in top_k_indices[i]
    ]

# Local evaluation
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

# Save results to file
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
with open(f"results_{timestamp}.txt", "w") as f:
    f.write(f"Model: DINOv2 vitb14\n")
    f.write(f"Dataset: LFW split\n")
    f.write(f"Top-1:  {top1/total:.2%}\n")
    f.write(f"Top-5:  {top5/total:.2%}\n")
    f.write(f"Top-10: {top10/total:.2%}\n")
print(f"Results saved to results_{timestamp}.txt")

# Submit (uncomment on competition day)
# submit(
#     results=results,
#     groupname="ALL-IN-dinov2",
#     url="http://competition-server-url/retrieval/"
# )
