import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS      = 50
LR          = 1e-3
BATCH_SIZE  = 32
DROPOUT     = 0.3
 
IMAGE_EMB_PATH = "./checkpoints/image_embeddings.pt"
TEXT_EMB_PATH  = "./checkpoints/text_embeddings.pt"
MLP_SAVE       = "./checkpoints/best_multimodal_mlp.pt"
LABEL_MAP_PATH = Path("./data/label_map.json")
EXCLUDED_SUFFIX = "leaf"

torch.cuda.manual_seed(42)
torch.manual_seed(42)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalMLP(nn.Module):
    """
    Fuses image and text embeddings and classifies plant disease.
    """
 
    def __init__(self, image_dim=768, text_dim=768,
                 num_classes=None, dropout=DROPOUT):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes must be provided.")
 
        fused_dim = image_dim + text_dim  # 1536
 
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
 
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
 
            nn.Linear(256, num_classes),
        )
 
    def forward(self, image_emb, text_emb):
        fused = torch.cat([image_emb, text_emb], dim=1)  # (B, 1536)
        return self.mlp(fused)
    

# ──────────────────────────────────────────────────────────────────────────────
# DATA 
# ──────────────────────────────────────────────────────────────────────────────

def load_embeddings():
    """Load image and text embeddings."""
    image_data = torch.load(IMAGE_EMB_PATH)
    img_train_embeddings = image_data["train_embeddings"]
    img_train_labels     = image_data["train_labels"]
    img_test_embeddings  = image_data["test_embeddings"]
    img_test_labels      = image_data["test_labels"]
 
    text_data = torch.load(TEXT_EMB_PATH)
    txt_train_embeddings = text_data["train_features"]
    txt_train_labels     = text_data["train_labels"]
    txt_test_embeddings  = text_data["test_features"]
    txt_test_labels      = text_data["test_labels"]
 
    print(f"Image train embeddings shape:  {img_train_embeddings.shape}")
    print(f"Text  train embeddings shape:  {txt_train_embeddings.shape}")
    print(f"Image test  embeddings shape:  {img_test_embeddings.shape}")
    print(f"Text  test  embeddings shape:  {txt_test_embeddings.shape}")
    print(f"Unique image classes: {img_train_labels.unique().shape[0]}")
    print(f"Unique text  classes: {txt_train_labels.unique().shape[0]}")
 
    return (img_train_embeddings, img_train_labels,
            img_test_embeddings,  img_test_labels,
            txt_train_embeddings, txt_train_labels,
            txt_test_embeddings,  txt_test_labels)


def load_filtered_label_info():
    """Return the class IDs to keep after excluding labels that end with 'leaf'."""
    with LABEL_MAP_PATH.open(encoding="utf-8") as f:
        label_map = json.load(f)

    ordered_labels = sorted(label_map.items(), key=lambda item: item[1])
    excluded = [(name, idx) for name, idx in ordered_labels if name.endswith(EXCLUDED_SUFFIX)]
    kept = [(name, idx) for name, idx in ordered_labels if not name.endswith(EXCLUDED_SUFFIX)]

    if not kept:
        raise ValueError(f"No classes remain after excluding labels ending with '{EXCLUDED_SUFFIX}'.")

    old_to_new = {old_idx: new_idx for new_idx, (_, old_idx) in enumerate(kept)}

    print(f"Excluding {len(excluded)} classes ending with '{EXCLUDED_SUFFIX}':")
    print(", ".join(name for name, _ in excluded))
    print(f"Keeping {len(kept)} classes for training/testing.")

    return {
        "excluded": excluded,
        "kept": kept,
        "old_to_new": old_to_new,
        "num_classes": len(kept),
    }


def filter_and_remap_embeddings(embeddings, labels, old_to_new):
    """Drop excluded labels and remap the remaining labels to 0..N-1."""
    remap = torch.full((max(old_to_new) + 1,), -1, dtype=torch.long)
    for old_idx, new_idx in old_to_new.items():
        remap[old_idx] = new_idx

    mapped_labels = remap[labels.long()]
    keep_mask = mapped_labels >= 0

    return embeddings[keep_mask], mapped_labels[keep_mask]
 
 
def align_embeddings(img_emb, img_lbl, txt_emb, txt_lbl, num_classes):
    """
    Align image and text embeddings by class label.
    Keeps all image samples; samples text embeddings with replacement
    to match the image count per class.
    """
    aligned_img, aligned_txt, aligned_lbl = [], [], []
 
    for class_idx in range(num_classes):
        img_mask  = (img_lbl == class_idx)
        txt_mask  = (txt_lbl == class_idx)
        img_class = img_emb[img_mask]
        txt_class = txt_emb[txt_mask]
 
        if len(img_class) == 0 or len(txt_class) == 0:
            continue
 
        n = len(img_class)
        repeat_idx = torch.randint(0, len(txt_class), (n,))
        aligned_img.append(img_class)
        aligned_txt.append(txt_class[repeat_idx])
        aligned_lbl.append(torch.full((n,), class_idx, dtype=torch.long))

    if not aligned_img:
        raise ValueError("No aligned classes found after filtering/remapping embeddings.")
 
    return (torch.cat(aligned_img),
            torch.cat(aligned_txt),
            torch.cat(aligned_lbl))
 
 
def build_loaders(train_img, train_txt, train_lbl,
                  test_img,  test_txt,  test_lbl):
    train_dataset = TensorDataset(train_img, train_txt, train_lbl)
    test_dataset  = TensorDataset(test_img,  test_txt,  test_lbl)
 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
 
    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train(model, train_loader, test_loader):
 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
 
    best_acc = 0.0
 
    for epoch in range(1, EPOCHS + 1):
 
        # Train loop ------------------------------------------------------------------------------
        model.train()
        train_loss    = 0.0
        train_correct = train_total = 0
 
        for img_emb, txt_emb, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True):
            img_emb = img_emb.to(DEVICE)
            txt_emb = txt_emb.to(DEVICE)
            labels  = labels.to(DEVICE)
 
            optimizer.zero_grad()
            logits = model(img_emb, txt_emb)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
 
            train_loss    += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)
 
        scheduler.step(epoch)
 
        # Test loop -------------------------------------------------------------------------------
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for img_emb, txt_emb, labels in tqdm(test_loader, desc="Testing", leave=True):
                img_emb = img_emb.to(DEVICE)
                txt_emb = txt_emb.to(DEVICE)
                labels  = labels.to(DEVICE)
                preds   = model(img_emb, txt_emb).argmax(1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
 
        train_acc = 100 * train_correct / train_total
        test_acc  = 100 * correct / total
 
        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"Loss: {train_loss / len(train_loader):.4f}  "
              f"Train Acc: {train_acc:.2f}%  "
              f"Test Acc: {test_acc:.2f}%")
 
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MLP_SAVE)
            print(f"  ✓ Best MLP saved ({best_acc:.2f}%)")
 
    print(f"\nDone — best Test Acc: {best_acc:.2f}%")
    return train_loss


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_full(model, loader):
    """Compute accuracy, macro precision, recall, and F1."""
    model.eval()
    all_preds, all_labels = [], []
 
    with torch.no_grad():
        for img_emb, txt_emb, labels in tqdm(loader, desc="Evaluating", leave=True):
            img_emb = img_emb.to(DEVICE)
            txt_emb = txt_emb.to(DEVICE)
            labels  = labels.to(DEVICE)
            preds   = model(img_emb, txt_emb).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
    acc       = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall    = recall_score(all_labels,    all_preds, average="macro", zero_division=0) * 100
    f1        = f1_score(all_labels,        all_preds, average="macro", zero_division=0) * 100
 
    return acc, precision, recall, f1


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"PyTorch {torch.__version__}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
 
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    label_info = load_filtered_label_info()
 
    # Data
    (img_train_emb, img_train_lbl,
     img_test_emb,  img_test_lbl,
     txt_train_emb, txt_train_lbl,
     txt_test_emb,  txt_test_lbl) = load_embeddings()

    img_train_emb, img_train_lbl = filter_and_remap_embeddings(
        img_train_emb, img_train_lbl, label_info["old_to_new"])
    img_test_emb, img_test_lbl = filter_and_remap_embeddings(
        img_test_emb, img_test_lbl, label_info["old_to_new"])
    txt_train_emb, txt_train_lbl = filter_and_remap_embeddings(
        txt_train_emb, txt_train_lbl, label_info["old_to_new"])
    txt_test_emb, txt_test_lbl = filter_and_remap_embeddings(
        txt_test_emb, txt_test_lbl, label_info["old_to_new"])

    print("\nFiltered embeddings:")
    print(f"Image train embeddings shape:  {img_train_emb.shape}")
    print(f"Text  train embeddings shape:  {txt_train_emb.shape}")
    print(f"Image test  embeddings shape:  {img_test_emb.shape}")
    print(f"Text  test  embeddings shape:  {txt_test_emb.shape}")
    print(f"Unique filtered image classes: {img_train_lbl.unique().shape[0]}")
    print(f"Unique filtered text  classes: {txt_train_lbl.unique().shape[0]}")
 
    print("\nAligning embeddings...")
    train_img, train_txt, train_lbl = align_embeddings(
        img_train_emb, img_train_lbl, txt_train_emb, txt_train_lbl, label_info["num_classes"])
    test_img, test_txt, test_lbl = align_embeddings(
        img_test_emb, img_test_lbl, txt_test_emb, txt_test_lbl, label_info["num_classes"])
 
    print(f"Aligned train: img={train_img.shape}, txt={train_txt.shape}, lbl={train_lbl.shape}")
    print(f"Aligned test:  img={test_img.shape},  txt={test_txt.shape},  lbl={test_lbl.shape}")
 
    train_loader, test_loader = build_loaders(
        train_img, train_txt, train_lbl,
        test_img,  test_txt,  test_lbl)
 
    # Model
    model     = MultimodalMLP(num_classes=label_info["num_classes"]).to(DEVICE)
 
    # Train
    print(f"\nTraining for {EPOCHS} epochs on {DEVICE}...\n")
    train(model, train_loader, test_loader)
 
    # Final metrics
    model.load_state_dict(torch.load(MLP_SAVE))
    print("\nFinal evaluation on test set:")
    acc, precision, recall, f1 = evaluate_full(model, test_loader)
    print(f"  Acc: {acc:.2f}%  Precision: {precision:.2f}%  "
          f"Recall: {recall:.2f}%  F1: {f1:.2f}%")
 
 
if __name__ == "__main__":
    main()
