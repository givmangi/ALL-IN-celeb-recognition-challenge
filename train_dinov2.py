"""
DINOv2 Fine-tuning with Triplet Loss
======================================
Author: Kristine
Branch: kristine-dinov2

Fine-tunes DINOv2 ViT-B/14 using triplet loss on VGGFace2-HQ.
The model learns to pull same-identity embeddings closer together
and push different-identity embeddings further apart.

Training strategy:
- Anchor: one image of a celebrity
- Positive: different image of the same celebrity  
- Negative: image of a different celebrity
- Loss: triplet margin loss

Best base configuration from experiments:
- Model: DINOv2 ViT-B/14
- Resolution: 336x336
- Preprocessing: MTCNN face crop + EXIF rotation fix
"""

import os
import random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME   = "dinov2_vitb14"
RESOLUTION   = 224
BATCH_SIZE   = 16        # triplets per batch
EPOCHS       = 15
LR           = 1e-5      # small learning rate for fine-tuning
MARGIN       = 0.3       # triplet loss margin
SEED         = 1         # for reproducible train/val split
NUM_TRIPLETS  = 5000     # triplets per epoch
EMBEDDING_DIM = 768      # ViT-B/14 output dimension

DATA_FOLDER  = "/home/disi/data/vggface2_raw/1"  # we'll create this split
SAVE_PATH    = "checkpoints"                       # where to save best model
# ─────────────────────────────────────────────────────────────────────────────

# ── Device setup ──────────────────────────────────────────────────────────────
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

os.makedirs(SAVE_PATH, exist_ok=True)

# ── Face detector ─────────────────────────────────────────────────────────────
mtcnn = MTCNN(
    image_size=RESOLUTION,
    margin=20,
    device=device,
    keep_all=False
)

# ── Augmentation transform (for training) ─────────────────────────────────────
augment_transform = T.Compose([
    T.Resize(356),                          # slightly larger than needed
    T.RandomHorizontalFlip(p=0.5),          # mirroring
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02
    ),                                       # mild color shift
    T.RandomGrayscale(p=0.3),               # handles possible B&W gallery
    T.CenterCrop(RESOLUTION),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Clean transform (for validation, no augmentation) ─────────────────────────
clean_transform = T.Compose([
    T.Resize(RESOLUTION),
    T.CenterCrop(RESOLUTION),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── Face crop helper ──────────────────────────────────────────────────────────
def detect_and_crop_face(img):
    face_tensor = mtcnn(img)
    if face_tensor is not None:
        return T.ToPILImage()(face_tensor.cpu().clamp(0, 1))
    return img

def load_image(path):
    """Load image with rotation fix and face crop."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)  # rotation fix
        # img = detect_and_crop_face(img)      # face crop disabled
        return img.copy()

# ── Triplet Dataset ───────────────────────────────────────────────────────────
class TripletFaceDataset(Dataset):
    """
    Creates triplets (anchor, positive, negative) from a folder of identities.
    Uses num_triplets to control epoch length regardless of dataset size.
    """
    def __init__(self, data_folder, transform, num_triplets=NUM_TRIPLETS):
        self.transform = transform
        self.num_triplets = num_triplets
        self.identity_to_images = {}

        # Build dictionary: identity -> list of image paths
        for identity in os.listdir(data_folder):
            identity_path = os.path.join(data_folder, identity)
            if not os.path.isdir(identity_path):
                continue
            images = [
                os.path.join(identity_path, f)
                for f in os.listdir(identity_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            # Need at least 2 images for anchor + positive
            if len(images) >= 2:
                self.identity_to_images[identity] = images

        self.identities = list(self.identity_to_images.keys())
        total_imgs = sum(len(v) for v in self.identity_to_images.values())
        print(f"Dataset loaded: {len(self.identities)} identities, {total_imgs} images")

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        # ── Anchor and Positive (same identity) ──────────────────────────────
        anchor_identity = random.choice(self.identities)
        anchor_path, positive_path = random.sample(
            self.identity_to_images[anchor_identity], 2
        )

        # ── Negative (different identity) ────────────────────────────────────
        negative_identity = random.choice(self.identities)
        while negative_identity == anchor_identity:
            negative_identity = random.choice(self.identities)
        negative_path = random.choice(
            self.identity_to_images[negative_identity]
        )

        anchor   = self.transform(load_image(anchor_path))
        positive = self.transform(load_image(positive_path))
        negative = self.transform(load_image(negative_path))

        return anchor, positive, negative

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
model = model.to(device)

# ── Freeze most layers, only fine-tune last 2 blocks ─────────────────────────
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 2 transformer blocks
for block in model.blocks[-2:]:
    for param in block.parameters():
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} parameters")


# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# ── Triplet Loss ──────────────────────────────────────────────────────────────
triplet_loss = nn.TripletMarginLoss(margin=MARGIN, p=2)

# ── Dataset and DataLoader ────────────────────────────────────────────────────
print("Loading training dataset...")
train_dataset = TripletFaceDataset(
    data_folder=DATA_FOLDER,
    transform=augment_transform
)
dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print(f"{len(dataloader)} batches per epoch")

# ── Training loop ─────────────────────────────────────────────────────────────
best_loss = float('inf')
best_epoch = 0

print(f"Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        # Forward pass
        anchor_emb   = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        # Normalize
        anchor_emb   = torch.nn.functional.normalize(anchor_emb,   p=2, dim=1)
        positive_emb = torch.nn.functional.normalize(positive_emb, p=2, dim=1)
        negative_emb = torch.nn.functional.normalize(negative_emb, p=2, dim=1)

        # Check for NaN
        if torch.isnan(anchor_emb).any():
            print(f"NaN detected at batch {batch_idx}, skipping...")
            continue

        # Compute loss
        loss = triplet_loss(anchor_emb, positive_emb, negative_emb)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
    print(f"Epoch {epoch+1}/{EPOCHS} complete — Avg Loss: {avg_loss:.4f}")

    # Save checkpoint every epoch
    torch.save(model.state_dict(), f"{SAVE_PATH}/checkpoint_100pct_epoch{epoch+1}.pth")
    print(f"  Checkpoint saved: checkpoint_epoch{epoch+1}.pth")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), f"{SAVE_PATH}/best_model_100pct.pth")
        print(f"  ✓ Best model saved (epoch {best_epoch}, loss {best_loss:.4f})")

print(f"\nTraining complete!")
print(f"Best model at epoch {best_epoch} with loss {best_loss:.4f}")
print(f"Saved to {SAVE_PATH}/best_model.pth")