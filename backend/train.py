"""
train.py
--------
  build_optimiser()  — two-LR AdamW (ViT fine-tune vs MLP from scratch)
  build_scheduler()  — linear warmup → cosine decay
  train_one_epoch()  — single forward/backward pass over training set
  save_checkpoint()  — persist model + optimiser state to disk
  run_training()     — full loop with early stopping, called by main.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from bert_encoder import CachedTextEmbeddings, CACHE_DIR
from vit import ViTEncoder
from multimodal import MultimodalDiseaseClassifier
from dataset import build_dataloaders
from evaluate import evaluate, load_checkpoint

# ── Config ────────────────────────────────────────────────────────────────────

VIT_LR        = 2e-5   # fine-tuning pretrained ViT layers
MLP_LR        = 1e-4   # training MLP head from scratch
WEIGHT_DECAY  = 0.01
WARMUP_EPOCHS = 2
FROZEN_LAYERS = 10     # freeze bottom 10 of 12 ViT blocks
HIDDEN_DIM    = 512
DROPOUT       = 0.3
PATIENCE      = 5      # early stopping patience (epochs without improvement)

CHECKPOINT_DIR = Path("checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")


# ── Optimiser ─────────────────────────────────────────────────────────────────

def build_optimiser(model: MultimodalDiseaseClassifier) -> torch.optim.Optimizer:
    """
    Two param groups with different learning rates:
      ViT unfrozen params → VIT_LR  (low  — fine-tuning pretrained weights)
      MLP params          → MLP_LR  (high — training from scratch)
    """
    vit_params = [p for p in model.vit.parameters() if p.requires_grad]
    mlp_params = list(model.mlp.parameters())

    return torch.optim.AdamW(
        [
            {"params": vit_params, "lr": VIT_LR},
            {"params": mlp_params, "lr": MLP_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ── Scheduler ─────────────────────────────────────────────────────────────────

def build_scheduler(optimiser: torch.optim.Optimizer, total_steps: int, steps_per_epoch: int):
    """
    Linear warmup for the first WARMUP_EPOCHS × steps_per_epoch steps,
    then cosine annealing to near-zero for the remainder.
    """
    warmup_steps = max(1, steps_per_epoch * WARMUP_EPOCHS)

    warmup = LinearLR(optimiser, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimiser,
                               T_max=max(1, total_steps - warmup_steps),
                               eta_min=1e-6)
    return SequentialLR(optimiser, schedulers=[warmup, cosine],
                        milestones=[warmup_steps])


# ── Single epoch ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model:        MultimodalDiseaseClassifier,
    loader:       DataLoader,
    criterion:    nn.Module,
    optimiser:    torch.optim.Optimizer,
    scheduler,
    augment_text: bool = True,
) -> dict:
    """
    One full pass over the training set.
    Returns dict with keys: loss, accuracy, time.
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0
    t0         = time.time()

    for images, label_ids in loader:
        images    = images.to(DEVICE)
        label_ids = label_ids.to(DEVICE)

        optimiser.zero_grad()
        logits = model(images, label_ids, augment_text=augment_text)
        loss   = criterion(logits, label_ids)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        scheduler.step()

        total_loss += loss.item() * len(label_ids)
        correct    += (logits.argmax(-1) == label_ids).sum().item()
        n          += len(label_ids)

    return {
        "loss":     total_loss / n,
        "accuracy": correct / n,
        "time":     time.time() - t0,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model:     MultimodalDiseaseClassifier,
    optimiser: torch.optim.Optimizer,
    epoch:     int,
    metrics:   dict,
    path:      Path,
):
    torch.save(
        {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimiser.state_dict(),
            "metrics":          metrics,
        },
        path,
    )


# ── Full training run ─────────────────────────────────────────────────────────

def run_training(args):
    """
    Full training pipeline. Called by main.py when neither --eval_only
    nor --predict is set.

    Steps:
      1. Load BERT embedding cache
      2. Build train / val DataLoaders
      3. Build MultimodalDiseaseClassifier
      4. Train with early stopping, saving best and latest checkpoints
      5. Write training history to JSON
    """
    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {args.epochs}")
    print(f"{'='*60}\n")

    # ── Text embeddings ────────────────────────────────────────────────────
    cache = CachedTextEmbeddings(args.dataset, Path(args.cache_dir))

    # ── Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, num_classes = build_dataloaders(args.dataset, cache)

    # ── Model ──────────────────────────────────────────────────────────────
    vit = ViTEncoder(
        frozen_layers=args.frozen_layers,
        checkpoint=args.vit_checkpoint or None,
    )
    model = MultimodalDiseaseClassifier(
        vit_encoder=vit,
        text_cache=cache,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    params = model.trainable_params()
    print(f"ViT trainable : {params['vit_trainable']:,}")
    print(f"MLP trainable : {params['mlp_trainable']:,}")
    print(f"Total         : {params['total']:,}")
    print(f"BERT          : frozen + pre-computed (not in graph)\n")

    # ── Optimiser / scheduler / loss ───────────────────────────────────────
    optimiser   = build_optimiser(model)
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    scheduler       = build_scheduler(optimiser, total_steps, steps_per_epoch)
    criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 1
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    if args.resume:
        latest = CHECKPOINT_DIR / f"{args.dataset}_latest.pt"
        if latest.exists():
            start_epoch, _ = load_checkpoint(model, optimiser, latest)
            start_epoch += 1

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc     = 0.0
    epochs_no_improv = 0
    history          = []

    for epoch in range(start_epoch, args.epochs + 1):

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimiser, scheduler,
            augment_text=True,
        )
        val_metrics = evaluate(model, val_loader, criterion, cache)

        print(
            f"Epoch {epoch:>3}/{args.epochs}  "
            f"loss={train_metrics['loss']:.4f}  "
            f"train_acc={train_metrics['accuracy']:.3f}  "
            f"val_top1={val_metrics['top1_accuracy']:.3f}  "
            f"val_top5={val_metrics['top5_accuracy']:.3f}  "
            f"val_macro={val_metrics['macro_accuracy']:.3f}  "
            f"({train_metrics['time']:.1f}s)"
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        # Save latest checkpoint (for --resume)
        save_checkpoint(
            model, optimiser, epoch, val_metrics,
            CHECKPOINT_DIR / f"{args.dataset}_latest.pt",
        )

        # Save best checkpoint
        if val_metrics["top1_accuracy"] > best_val_acc:
            best_val_acc     = val_metrics["top1_accuracy"]
            epochs_no_improv = 0
            save_checkpoint(
                model, optimiser, epoch, val_metrics,
                CHECKPOINT_DIR / f"{args.dataset}_best.pt",
            )
            print(f"  ✓ Best model saved  (val_top1={best_val_acc:.3f})")
        else:
            epochs_no_improv += 1
            if epochs_no_improv >= PATIENCE:
                print(f"\nEarly stopping — no improvement for {PATIENCE} epochs.")
                break

    # Save training history
    history_path = CHECKPOINT_DIR / f"{args.dataset}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nHistory  → {history_path}")
    print(f"Best val top-1 : {best_val_acc:.4f}")