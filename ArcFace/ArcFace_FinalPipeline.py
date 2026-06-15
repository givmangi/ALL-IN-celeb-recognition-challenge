import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

VGGFACE2_DIR   = 'C:\\Users\\mangi\\OneDrive\\Desktop\\sku\\ml_challenge\\ALL-IN-celeb-recognition-challenge\\dataset\\VGGface2_hQ\\full_dataset'
COMP_TRAIN_DIR = 'C:\\Users\\mangi\\OneDrive\\Desktop\\sku\\ml_challenge\\ALL-IN-celeb-recognition-challenge\\comp_data\\train'
COMP_QUERY_DIR = 'C:\\Users\\mangi\\OneDrive\\Desktop\\sku\\ml_challenge\\ALL-IN-celeb-recognition-challenge\\comp_data\\query'
COMP_GALLERY_DIR = 'C:\\Users\\mangi\\OneDrive\\Desktop\\sku\\ml_challenge\\ALL-IN-celeb-recognition-challenge\\comp_data\\gallery'

TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_vggface_dataloaders(data_dir=VGGFACE2_DIR, batch_size=64, split_ratio=(0.8, 0.1, 0.1)):
    """
    Loads VGGFace2-HQ with an 80/10/10 train/val/test split (fixed seed=1 for
    cross-team reproducibility, matching the DINOv2 and CLIP pipelines).
    Returns train, val, test loaders plus num_classes and test_size.
    """
    print(f"[VGGFace2] Loading dataset from {data_dir}...")
    dataset = datasets.ImageFolder(root=data_dir, transform=TRANSFORM)
    num_classes  = len(dataset.classes)
    total        = len(dataset)

    train_size = int(split_ratio[0] * total)
    val_size   = int(split_ratio[1] * total)
    test_size  = total - train_size - val_size

    generator = torch.Generator().manual_seed(1)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[VGGFace2] Split: {train_size} train | {val_size} val | {test_size} test | {num_classes} identities")
    return train_loader, val_loader, test_loader, num_classes, test_size

def get_competition_train_loader(data_dir=COMP_TRAIN_DIR, batch_size=64):
    """
    Loads the full competition training set (250 identities, ~5 000 images).
    All images are used for training; no internal split is applied.
    """
    print(f"[Competition] Loading training set from {data_dir}...")
    dataset     = datasets.ImageFolder(root=data_dir, transform=TRANSFORM)
    num_classes = len(dataset.classes)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"[Competition] {len(dataset)} images | {num_classes} identities")
    return loader, num_classes

class FlatImageDataset(Dataset):
    """
    Loads all images from a flat directory (no class sub-folders) for inference.
    Used for the competition query and gallery sets.
    """
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root, transform=None):
        self.transform = transform
        self.paths = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ])
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.paths[idx])

def get_competition_retrieval_loaders(
    query_dir=COMP_QUERY_DIR,
    gallery_dir=COMP_GALLERY_DIR,
    batch_size=64
):
    """
    Returns loaders for the competition query and gallery sets.
    Used exclusively at evaluation time; no labels are required.
    """
    query_ds   = FlatImageDataset(query_dir,   transform=TRANSFORM)
    gallery_ds = FlatImageDataset(gallery_dir, transform=TRANSFORM)
    query_loader   = DataLoader(query_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    gallery_loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[Retrieval] {len(query_ds)} queries | {len(gallery_ds)} gallery images")
    return query_loader, gallery_loader


class ArcMarginProduct(nn.Module):
    """
    ArcFace classification head.
    The fallback phi = cos(theta) - sin(pi - m)*m is applied when
    cos(theta) <= cos(pi - m) for numerical stability.
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.s  = s
        self.m  = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)          # fallback threshold
        self.mm    = math.sin(math.pi - m) * m       # fallback offset

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
        sine   = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi    = cosine * self.cos_m - sine * self.sin_m          # cos(theta + m)
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)  # fallback

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output  = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class CelebrityModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy         = torch.zeros(1, 3, 112, 112)
            flattened_size = self.backbone_conv(dummy).shape[1]

        self.embedding_layer = nn.Linear(flattened_size, 512)
        self.arc_face        = ArcMarginProduct(512, num_classes)

    def replace_head(self, num_classes, device):
        """
        Replaces the ArcFace head with a fresh one for a new class set.
        The backbone and embedding layer weights are preserved.
        """
        self.arc_face = ArcMarginProduct(512, num_classes).to(device)

    def forward(self, x, labels=None):
        x          = self.backbone_conv(x)
        embeddings = self.embedding_layer(x)
        if labels is not None:
            return self.arc_face(embeddings, labels)
        return F.normalize(embeddings, dim=1)


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images, labels)
        loss   = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"  Epoch {epoch+1}/{total_epochs} | Loss: {avg_loss:.4f}")
    return avg_loss


def extract_embeddings(model, loader, device):
    """Runs inference and returns (N, 512) L2-normalised embedding matrix."""
    model.eval()
    embeddings, filenames = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            embs   = model(images)         
            embeddings.append(embs.cpu())
            filenames.extend(batch[1] if isinstance(batch[1][0], str) else
                             [str(x.item()) for x in batch[1]])
    return torch.cat(embeddings, dim=0), filenames


def compute_retrieval_metrics(query_embeddings, query_labels,
                               gallery_embeddings, gallery_labels, device):
    """
    Computes accuracies and top scores
    """
    q = query_embeddings.to(device)
    g = gallery_embeddings.to(device)

    sim_matrix  = torch.mm(q, g.T)
    top10_idx   = torch.topk(sim_matrix, 10, dim=1).indices.cpu()

    n = len(query_labels)
    top1_correct = top5_correct = top10_correct = 0

    for i in range(n):
        q_label   = query_labels[i]
        retrieved = gallery_labels[top10_idx[i]]
        if q_label == retrieved[0]:
            top1_correct  += 1
        if q_label in retrieved[:5]:
            top5_correct  += 1
        if q_label in retrieved[:10]:
            top10_correct += 1

    top1  = top1_correct  / n
    top5  = top5_correct  / n
    top10 = top10_correct / n
    score = 600 * top1 + 300 * top5 + 100 * top10

    return top1, top5, top10, score


def evaluate_on_vggface_test(model, test_loader, test_size, device):
    """
    Self-retrieval evaluation on the VGGFace2 test split.
    """
    model.eval()
    all_embs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            all_embs.append(model(images.to(device)).cpu())
            all_labels.append(labels)
    embs   = torch.cat(all_embs,   dim=0)
    labels = torch.cat(all_labels, dim=0)

    sim = torch.mm(embs.to(device), embs.to(device).T)
    sim.fill_diagonal_(-1)
    top10_idx = torch.topk(sim, 10, dim=1).indices.cpu()

    top1_c = top5_c = top10_c = 0
    for i in range(test_size):
        q_lbl     = labels[i].item()
        retrieved = labels[top10_idx[i]].numpy()
        if q_lbl == retrieved[0]:            top1_c  += 1
        if q_lbl in retrieved[:5]:           top5_c  += 1
        if q_lbl in retrieved[:10]:          top10_c += 1

    top1  = top1_c  / test_size
    top5  = top5_c  / test_size
    top10 = top10_c / test_size
    score = 600 * top1 + 300 * top5 + 100 * top10
    return top1, top5, top10, score


def log_results(tag, top1, top5, top10, score, epochs, extra=""):
    """Formats and prints a results block. returns the string for file logging."""
    sep  = "-" * 50
    text = (
        f"\n{sep}\n"
        f"Run:            {tag}\n"
        f"Epochs trained: {epochs}\n"
        f"{sep}\n"
        f"Top-1:  {top1 * 100:.2f}%\n"
        f"Top-5:  {top5 * 100:.2f}%\n"
        f"Top-10: {top10 * 100:.2f}%\n"
        f"Score:  {score:.2f}  (600*T1 + 300*T5 + 100*T10)\n"
    )
    if extra:
        text += extra + "\n"
    text += sep + "\n"
    print(text)
    return text


def save_log(text, path):
    with open(path, "a") as f:
        f.write(text)


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def run_vggface_only(device, batch_size=64, epochs=10):
    """
    Trains exclusively on VGGFace2-HQ (80% split) and evaluates on the
    held-out 10% test split via self-retrieval.
    Corresponds to the 'ArcFace split' row in Table 2.
    """
    print("\n" + "="*60)
    print("PIPELINE: VGGFace2-HQ only (80/10/10 split)")
    print("="*60)

    train_loader, _, test_loader, num_classes, test_size = \
        get_vggface_dataloaders(batch_size=batch_size)

    model     = CelebrityModel(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, device, epoch, epochs)

    top1, top5, top10, score = evaluate_on_vggface_test(
        model, test_loader, test_size, device
    )
    text = log_results("VGGFace2 split (80%)", top1, top5, top10, score, epochs)
    save_log(text, "arcface_vggface_only.txt")
    save_model(model, "arcface_vggface_only.pth")
    return model


def run_competition_only(device, batch_size=64, epoch_checkpoints=(10, 20, 30, 40)):
    """
    Trains exclusively on the competition dataset for increasing numbers of epochs, evaluating at each checkpoint.
    """
    print("\n" + "="*60)
    print("PIPELINE: Competition only")
    print("="*60)

    comp_loader, num_classes = get_competition_train_loader(batch_size=batch_size)
    query_loader, gallery_loader = get_competition_retrieval_loaders(batch_size=batch_size)

    model     = CelebrityModel(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    all_logs    = ""
    best_score  = -1
    best_epoch  = -1
    epoch_checkpoints = sorted(epoch_checkpoints)
    max_epochs  = epoch_checkpoints[-1]

    for epoch in range(max_epochs):
        train_one_epoch(model, comp_loader, optimizer, device, epoch, max_epochs)

        if (epoch + 1) in epoch_checkpoints:
            print(f"\n[Checkpoint] Evaluating at epoch {epoch+1}...")

            q_embs, q_files  = extract_embeddings(model, query_loader,   device)
            g_embs, g_files  = extract_embeddings(model, gallery_loader, device)

            def label_from_filename(fname):
                return int(fname.split('_')[0])

            q_labels = torch.tensor([label_from_filename(f) for f in q_files])
            g_labels = torch.tensor([label_from_filename(f) for f in g_files])

            top1, top5, top10, score = compute_retrieval_metrics(
                q_embs, q_labels, g_embs, g_labels, device
            )
            text = log_results(
                f"Competition only — epoch {epoch+1}",
                top1, top5, top10, score, epoch + 1
            )
            all_logs += text
            save_model(model, f"arcface_comp_only_ep{epoch+1}.pth")

            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                save_model(model, "arcface_comp_only_best.pth")

    save_log(all_logs, "arcface_comp_only.txt")
    print(f"\n[Best] Epoch {best_epoch} | Score {best_score:.2f}")
    return model


def run_vggface_then_competition(device, batch_size=64,
                                  vgg_epochs=10, comp_epochs=10):
    """
    Two-stage pipeline: first trains on VGGFace2-HQ (80% split), then fine-tunes on the competition dataset.
    """
    print("\n" + "="*60)
    print("PIPELINE: VGGFace2 (split) + Competition fine-tuning")
    print("="*60)

    train_loader, _, _, vgg_num_classes, _ = \
        get_vggface_dataloaders(batch_size=batch_size)

    model     = CelebrityModel(vgg_num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n[Stage 1] VGGFace2 training for {vgg_epochs} epochs...")
    for epoch in range(vgg_epochs):
        train_one_epoch(model, train_loader, optimizer, device, epoch, vgg_epochs)

    comp_loader, comp_num_classes = get_competition_train_loader(batch_size=batch_size)
    model.replace_head(comp_num_classes, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n[Stage 2] Competition fine-tuning for {comp_epochs} epochs...")
    for epoch in range(comp_epochs):
        train_one_epoch(model, comp_loader, optimizer, device, epoch, comp_epochs)

    query_loader, gallery_loader = get_competition_retrieval_loaders(batch_size=batch_size)
    q_embs, q_files = extract_embeddings(model, query_loader,   device)
    g_embs, g_files = extract_embeddings(model, gallery_loader, device)

    def label_from_filename(fname):
        return int(fname.split('_')[0])

    q_labels = torch.tensor([label_from_filename(f) for f in q_files])
    g_labels = torch.tensor([label_from_filename(f) for f in g_files])

    top1, top5, top10, score = compute_retrieval_metrics(
        q_embs, q_labels, g_embs, g_labels, device
    )
    text = log_results(
        f"VGGFace2 split + Competition ({vgg_epochs}+{comp_epochs} epochs)",
        top1, top5, top10, score, vgg_epochs + comp_epochs
    )
    save_log(text, "arcface_vgg_then_comp.txt")
    save_model(model, "arcface_vgg_then_comp.pth")
    return model



def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # run_vggface_only(device, batch_size=64, epochs=10)

    # run_vggface_then_competition(device, batch_size=64, vgg_epochs=10, comp_epochs=10)

    run_competition_only(device, batch_size=64, epoch_checkpoints=(10, 20, 30, 40))

if __name__ == "__main__":
    run_pipeline()