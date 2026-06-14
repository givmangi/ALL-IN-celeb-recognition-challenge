import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import torchvision.transforms as T

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS - only touch these
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_DIR       = "train"          # change to competition training folder on the day

# STARTING_WEIGHTS = "clip_finetuned_vgg.pt"  # our best pre-trained model
STARTING_WEIGHTS = ""  # out-of-box model  <--------- THIS ONE WE'VE USED

NUM_TRIPLETS = 10000   # reduce to 5000 if time is short
NUM_EPOCHS   = 5       # reduce to 2 if time is short
BATCH_SIZE   = 32
LR           = 1e-6

SAVE_NAME    = "clip_competition.pt"   # final output weights

# ══════════════════════════════════════════════════════════════════════════════

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Load CLIP and restore pre-trained weights ─────────────────────────────────
print("Loading CLIP model...")
model, _ = clip.load("ViT-L/14", device=device)
model = model.float()

if os.path.exists(STARTING_WEIGHTS):
    model.load_state_dict(torch.load(STARTING_WEIGHTS, map_location=device))
    print(f"Loaded starting weights from: {STARTING_WEIGHTS}")
else:
    print(f"WARNING: {STARTING_WEIGHTS} not found, starting from raw CLIP")

# ── Freeze all, unfreeze last 2 transformer blocks ───────────────────────────
for param in model.parameters():
    param.requires_grad = False
for block in model.visual.transformer.resblocks[-2:]:
    for param in block.parameters():
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,}")

# ── Augmentation ──────────────────────────────────────────────────────────────
augment_transform = T.Compose([
    T.Resize(256),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    T.RandomGrayscale(p=0.3),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std= (0.26862954, 0.26130258, 0.27577711)
    ),
])


# ── Triplet Dataset ───────────────────────────────────────────────────────────
class TripletFaceDataset(Dataset):
    """
    Supports two folder structures:
    1. identity subfolders:  train_dir/Brad_Pitt/001.jpg
    2. flat with prefix:     train_dir/Brad_Pitt__001.jpg
    """
    def __init__(self, train_dir, transform, num_triplets=10000):
        self.transform    = transform
        self.num_triplets = num_triplets
        self.identity_images = {}

        for entry in os.listdir(train_dir):
            entry_path = os.path.join(train_dir, entry)

            # structure 1: identity subfolders
            if os.path.isdir(entry_path):
                images = [
                    os.path.join(entry_path, f)
                    for f in os.listdir(entry_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                if len(images) >= 2:
                    self.identity_images[entry] = images

            # structure 2: flat identity__filename.jpg
            elif "__" in entry and entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                identity = entry.split("__")[0]
                self.identity_images.setdefault(identity, []).append(entry_path)

        # remove identities with only 1 image
        self.identity_images = {k: v for k, v in self.identity_images.items() if len(v) >= 2}
        self.identities = list(self.identity_images.keys())

        total_imgs = sum(len(v) for v in self.identity_images.values())
        print(f"Loaded {len(self.identities)} identities, {total_imgs} images")

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        # anchor + positive: two different images of same person
        anchor_identity          = random.choice(self.identities)
        anchor_path, positive_path = random.sample(self.identity_images[anchor_identity], 2)

        # negative: image of a different person
        negative_identity = random.choice(self.identities)
        while negative_identity == anchor_identity:
            negative_identity = random.choice(self.identities)
        negative_path = random.choice(self.identity_images[negative_identity])

        anchor   = self.transform(Image.open(anchor_path).convert("RGB"))
        positive = self.transform(Image.open(positive_path).convert("RGB"))
        negative = self.transform(Image.open(negative_path).convert("RGB"))

        return anchor, positive, negative


# ── Setup ─────────────────────────────────────────────────────────────────────
dataset    = TripletFaceDataset(TRAIN_DIR, transform=augment_transform, num_triplets=NUM_TRIPLETS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer    = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)


# ── Encode ────────────────────────────────────────────────────────────────────
def encode(imgs):
    features = model.encode_image(imgs.float())
    return F.normalize(features, p=2, dim=1)


# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nStarting training: {NUM_EPOCHS} epochs x {NUM_TRIPLETS} triplets")

# Initialize a tracker for the best loss
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    total_loss  = 0
    num_batches = 0

    for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
        anchors   = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        anchor_emb   = encode(anchors)
        positive_emb = encode(positives)
        negative_emb = encode(negatives)

        if torch.isnan(anchor_emb).any():
            print(f"NaN at batch {batch_idx}, skipping...")
            continue

        loss = triplet_loss(anchor_emb, positive_emb, negative_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

        if batch_idx % 25 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")

    # 1. Save every epoch just as a backup
    ckpt = f"clip_competition_epoch{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt)
    
    # 2. Check if this is the best epoch and overwrite SAVE_NAME if it is
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), SAVE_NAME)
        print(f"⭐ New best model saved to {SAVE_NAME} (Loss improved to {best_loss:.4f})")
    else:
        print(f"Checkpoint saved: {ckpt}. (Did not beat best loss of {best_loss:.4f})")

print(f"\nTraining complete. The best weights are saved in {SAVE_NAME}")

'''
On competition day the only things to change are at the top:
TRAIN_DIR        = "/path/to/competition/train"
STARTING_WEIGHTS = "clip_finetuned_vgg.pt"   # already on the VM
SAVE_NAME        = "clip_competition.pt"
'''