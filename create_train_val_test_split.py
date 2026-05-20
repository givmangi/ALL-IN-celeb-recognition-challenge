import os
import shutil
import random
from collections import defaultdict

random.seed(1)

vgg_root = "/home/disi/data/vggface2_raw/1"
output_root = "/home/disi/data/vggface2_train_split"

train_folder = os.path.join(output_root, "train")
val_folder   = os.path.join(output_root, "val")
test_query   = os.path.join(output_root, "test/query")
test_gallery = os.path.join(output_root, "test/gallery")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder,   exist_ok=True)
os.makedirs(test_query,   exist_ok=True)
os.makedirs(test_gallery, exist_ok=True)

# Get all identities with at least 2 images
identities = []
for identity in os.listdir(vgg_root):
    identity_path = os.path.join(vgg_root, identity)
    if not os.path.isdir(identity_path):
        continue
    images = [f for f in os.listdir(identity_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(images) >= 2:
        identities.append(identity)

random.shuffle(identities)
print(f"Total identities: {len(identities)}")

n_train = int(len(identities) * 0.80)
n_val   = int(len(identities) * 0.10)

train_ids = identities[:n_train]
val_ids   = identities[n_train:n_train+n_val]
test_ids  = identities[n_train+n_val:]

print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

# Copy train and val folders
for identity in train_ids:
    shutil.copytree(
        os.path.join(vgg_root, identity),
        os.path.join(train_folder, identity)
    )
print("Train copied!")

for identity in val_ids:
    shutil.copytree(
        os.path.join(vgg_root, identity),
        os.path.join(val_folder, identity)
    )
print("Val copied!")

# Test: first image → query, rest → gallery
for identity in test_ids:
    identity_path = os.path.join(vgg_root, identity)
    images = sorted([f for f in os.listdir(identity_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    random.shuffle(images)
    shutil.copy(
        os.path.join(identity_path, images[0]),
        os.path.join(test_query, f"{identity}__{images[0]}")
    )
    for img in images[1:]:
        shutil.copy(
            os.path.join(identity_path, img),
            os.path.join(test_gallery, f"{identity}__{img}")
        )

print(f"Test query: {len(os.listdir(test_query))}")
print(f"Test gallery: {len(os.listdir(test_gallery))}")
print("Done!")
