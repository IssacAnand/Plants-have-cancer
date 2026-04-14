import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
 
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image
import timm
import evaluate

from dataset import PlantWildDataset


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "mobilevitv2_150"  
IMAGES_DIR  = "./data/images/plantwild"
IMG_SIZE    = 320      
BATCH_SIZE  = 16        
EPOCHS      = 50     
BACKBONE_LR = 1e-5      
HEAD_LR     = 1e-3      
SAVE_DIR    = f"./checkpoints"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDED_SUFFIX = "leaf"

torch.cuda.manual_seed(42)
torch.manual_seed(42)

metric_f1  = evaluate.load("f1")
metric_acc = evaluate.load("accuracy")


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────
class MobileViT(nn.Module):
    """
    Full MobileViT fine-tune with all layers trainable.
    """
 
    def __init__(self, num_classes: int,
                 model_name: str, dropout: float = 0.2):
        super().__init__()
 
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.embed_dim = self.backbone.num_features
 
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

        for param in self.backbone.parameters():
            param.requires_grad = True
 
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model           : {model_name}")
        print(f"Total params    : {total / 1e6:.2f}M")
        print(f"Trainable params: {trainable / 1e6:.2f}M  "
              f"({100 * trainable / total:.1f}% unfrozen)")
 
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))
 
    def get_encoder(self):
        return self.backbone
 
    def get_param_groups(self, backbone_lr: float, head_lr: float):
        """
        Separate LRs for backbone and head.
        Backbone gets a lower LR to preserve pretrained weights.
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(),     "lr": head_lr},
        ]


# ──────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────────────────────────────────────
def get_transforms(img_size: int = IMG_SIZE, train: bool = True):
    """
    Data augmentation and normalization similar to pre-trained ImageNet configs.
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_filtered_label_map(images_dir: str):
    """Build a label map that excludes class folders ending with 'leaf'."""
    class_dirs = sorted([d for d in Path(images_dir).iterdir() if d.is_dir()])
    excluded = [d.name for d in class_dirs if d.name.endswith(EXCLUDED_SUFFIX)]
    kept = [d.name for d in class_dirs if not d.name.endswith(EXCLUDED_SUFFIX)]

    if not kept:
        raise ValueError(f"No classes remain after excluding labels ending with '{EXCLUDED_SUFFIX}'.")

    print(f"Excluding {len(excluded)} classes ending with '{EXCLUDED_SUFFIX}':")
    print(", ".join(excluded))
    print(f"Keeping {len(kept)} classes for MobileViT training/testing.")

    return {class_name: idx for idx, class_name in enumerate(kept)}


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────
def train(model, train_loader, test_loader, device, save_dir, class_weights=None):
 
    os.makedirs(save_dir, exist_ok=True)
 
    model = model.to(device)
 
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=0.1,
    )
 
    optimizer = torch.optim.AdamW(
        model.get_param_groups(BACKBONE_LR, HEAD_LR), weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
 
    scaler   = GradScaler("cuda")
    best_acc = 0.0
    best_f1 = 0.0

    # Training loop ---------------------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
 
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True):
            images, labels = images.to(device), labels.to(device)
 
            optimizer.zero_grad()
            with autocast("cuda"):  # auto switch between fp32 and fp16 for sensitive / non-sensitive computations
                loss = criterion(model(images), labels)
 
            scaler.scale(loss).backward()   # scale the loss to prevent underflow in fp16
            scaler.unscale_(optimizer)      # scale back down
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # cap gradients to prevent gradient explosion from bad batch
            scaler.step(optimizer)      # check if any gradients are inf/nan, and skip optimizer step if yes
            scaler.update()             # update the scale factor for next iteration
 
            train_loss += loss.item()
 
        scheduler.step(epoch)  
 
        # Test loop -------------------------------------------------------------------------------
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing", leave=True):
                images, labels = images.to(device), labels.to(device)
                with autocast("cuda"):
                    preds = model(images).argmax(1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
 
        f1  = metric_f1.compute(predictions=all_preds, references=all_labels, average="macro")
        acc = metric_acc.compute(predictions=all_preds, references=all_labels)
        macro_f1 = f1["f1"] * 100
        accuracy = acc["accuracy"] * 100

        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"Loss: {train_loss / len(train_loader):.4f}  "
              f"Test Acc: {accuracy:.2f}%  "
              f"Macro F1: {macro_f1:.2f}%")
 
        if accuracy > best_acc:
            best_acc = accuracy
            best_f1 = macro_f1
            torch.save(
                model.get_encoder().state_dict(),
                os.path.join(save_dir, "best_image_encoder.pt")
            )
            print(f"  ✓ Best encoder saved (Acc: {best_acc:.2f}%  Macro F1: {best_f1:.2f}%)")
 
    print(f"\nFinetuning complete — best Test Acc: {best_acc:.2f}%  Macro F1: {best_f1:.2f}%")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# EMBEDDING EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
def extract_embeddings(encoder, dataloader, device):
    encoder.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting", leave=True):
            images = images.to(device)
            with autocast("cuda"):
                embeddings = encoder(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)
 
 
def save_embeddings(encoder, train_ds, test_loader, save_dir, device):
    # use test transforms — no augmentation for embedding extraction
    train_ds_eval = PlantWildDataset(
        IMAGES_DIR,
        transform=get_transforms(IMG_SIZE, train=False),
        split="train",
        label_map=train_ds.label_map,
    )
    train_loader_eval = DataLoader(
        train_ds_eval, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
 
    print("Extracting train embeddings...")
    train_emb, train_lbl = extract_embeddings(encoder, train_loader_eval, device)
    print("Extracting test embeddings...")
    test_emb, test_lbl   = extract_embeddings(encoder, test_loader, device)
 
    print(f"Train embeddings : {train_emb.shape}")   # (14832, 768)
    print(f"Test embeddings  : {test_emb.shape}")    # (3708, 768)
 
    save_path = os.path.join(save_dir, "image_embeddings.pt")
    torch.save({
        "train_embeddings": train_emb,
        "train_labels":     train_lbl,
        "test_embeddings":  test_emb,
        "test_labels":      test_lbl,
    }, save_path)
    print(f"Embeddings saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_NAME}")
 
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    label_map = build_filtered_label_map(IMAGES_DIR)
 
    # Load datasets
    train_ds = PlantWildDataset(IMAGES_DIR,
                                transform=get_transforms(IMG_SIZE, train=True),
                                split="train",
                                label_map=label_map)
    test_ds  = PlantWildDataset(IMAGES_DIR,
                                transform=get_transforms(IMG_SIZE, train=False),
                                split="test", label_map=label_map)
 
    train_ds.save_label_map("./data/label_map.json")
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True)
 
    # Train model
    model = MobileViT(
        num_classes=len(train_ds.classes),
        model_name=MODEL_NAME,
        dropout=0.2,
    )

    class_counts  = train_ds.get_class_counts()
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    print(f"Class weights computed for {len(class_weights)} classes")

    model = train(model, train_loader, test_loader, DEVICE, SAVE_DIR, class_weights=class_weights)
 
    # Extract embeddings 
    encoder = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
    encoder.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_image_encoder.pt"),
                   map_location=DEVICE)
    )
    encoder = encoder.to(DEVICE)
    save_embeddings(encoder, train_ds, test_loader, SAVE_DIR, DEVICE)
