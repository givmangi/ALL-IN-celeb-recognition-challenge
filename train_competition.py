"""
DINOv2 Competition Day Fine-tuning
====================================
Author: Kristine
Branch: kristine-dinov2

Fine-tunes the pre-trained split model on competition training data.
Designed to run during competition with limited time — only 3 epochs.

Starting point: best_model_split.pth (trained on VGGFace2-HQ)
Training data: competition training data (provided on competition day)
Output: best_model_competition.pth
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
MODEL_NAME    = "dinov2_vitb14"
RESOLUTION    = 224
BATCH_SIZE    = 8
EPOCHS        = 3           # limited time on competition day
LR            = 1e-5
MARGIN        = 0.3
SEED          = 1
NUM_TRIPLETS  = 5000

# !! UPDATE THIS ON COMPETITION DAY !!
DATA_FOLDER   = "/path/to/competition/training/data"
CHECKPOINT_IN  = "checkpoints/best_model_split_v3_epoch3.pth"   # starting point
CHECKPOINT_OUT = "checkpoints/best_model_competition.pth"  # output
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

os.makedirs("checkpoints", exist_ok=True)

# ── Augmentation transform ────────────────────────────────────────────────────
augment_transform = T.Compose([
    T.Resize(256),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.RandomGrayscale(p=0.3),
    T.CenterCrop(RESOLUTION),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Image loading ─────────────────────────────────────────────────────────────
def load_image(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)
        return img.copy()

# ── Triplet Dataset ───────────────────────────────────────────────────────────
class TripletFaceDataset(Dataset):
    def __init__(self, data_folder, transform, num_triplets=NUM_TRIPLETS):
        self.transform = transform
        self.num_triplets = num_triplets
        self.identity_to_images = {}

        for identity in os.listdir(data_folder):
            identity_path = os.path.join(data_folder, identity)
            if not os.path.isdir(identity_path):
                continue
            images = [
                os.path.join(identity_path, f)
                for f in os.listdir(identity_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if len(images) >= 2:
                self.identity_to_images[identity] = images

        self.identities = list(self.identity_to_images.keys())
        total_imgs = sum(len(v) for v in self.identity_to_images.values())
        print(f"Dataset: {len(self.identities)} identities, {total_imgs} images")

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        anchor_identity = random.choice(self.identities)
        anchor_path, positive_path = random.sample(
            self.identity_to_images[anchor_identity], 2
        )
        negative_identity = random.choice(self.identities)
        while negative_identity == anchor_identity:
            negative_identity = random.choice(self.identities)
        negative_path = random.choice(self.identity_to_images[negative_identity])

        anchor   = self.transform(load_image(anchor_path))
        positive = self.transform(load_image(positive_path))
        negative = self.transform(load_image(negative_path))

        return anchor, positive, negative

# ── Load model from checkpoint ────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} from {CHECKPOINT_IN}...")
model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
model.load_state_dict(torch.load(CHECKPOINT_IN, map_location=device))
print("Checkpoint loaded!")
model = model.to(device)

# ── Freeze most layers, only fine-tune last 2 blocks ─────────────────────────
for param in model.parameters():
    param.requires_grad = False
for block in model.blocks[-2:]:
    for param in block.parameters():
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} parameters")

# ── Optimizer and Loss ────────────────────────────────────────────────────────
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)
triplet_loss = nn.TripletMarginLoss(margin=MARGIN, p=2)

# ── Dataset and DataLoader ────────────────────────────────────────────────────
print("Loading competition training data...")
dataset = TripletFaceDataset(data_folder=DATA_FOLDER, transform=augment_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=True)
print(f"{len(dataloader)} batches per epoch")

# ── Training loop ─────────────────────────────────────────────────────────────
best_loss = float('inf')
best_epoch = 0

print(f"Starting fine-tuning for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_emb   = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        anchor_emb   = torch.nn.functional.normalize(anchor_emb,   p=2, dim=1)
        positive_emb = torch.nn.functional.normalize(positive_emb, p=2, dim=1)
        negative_emb = torch.nn.functional.normalize(negative_emb, p=2, dim=1)

        if torch.isnan(anchor_emb).any():
            continue

        loss = triplet_loss(anchor_emb, positive_emb, negative_emb)

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
    torch.save(model.state_dict(), f"checkpoints/competition_epoch{epoch+1}.pth")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), CHECKPOINT_OUT)
        print(f"  ✓ Best model saved (epoch {best_epoch})")

print(f"\nFine-tuning complete!")
print(f"Best model: epoch {best_epoch}, loss {best_loss:.4f}")
print(f"Saved to {CHECKPOINT_OUT}")