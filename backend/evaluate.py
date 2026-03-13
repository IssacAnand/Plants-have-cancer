"""
evaluate.py
-----------
  evaluate()       — runs model over a DataLoader, returns metrics dict
  run_evaluation() — loads best checkpoint, prints full metrics report
  run_predict()    — classifies a single image file
"""

from pathlib import Path

import torch
import torch.nn as nn

from bert_encoder import CachedTextEmbeddings, CACHE_DIR
from vit import ViTEncoder, EVAL_TRANSFORMS
from multimodal import MultimodalDiseaseClassifier
from dataset import build_dataloaders

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path("checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")


# ── Core eval loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:     MultimodalDiseaseClassifier,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    cache:     CachedTextEmbeddings,
) -> dict:
    """
    Runs the model over every batch in loader (no gradients).

    Returns
    -------
    dict with keys:
        loss            — cross-entropy averaged over all samples
        top1_accuracy   — fraction of samples where argmax == true label
        top5_accuracy   — fraction of samples where true label is in top-5
        macro_accuracy  — mean per-class accuracy (equal weight per class)
    """
    model.eval()

    total_loss      = 0.0
    correct_top1    = 0
    correct_top5    = 0
    n               = 0
    num_classes     = cache.num_classes()
    per_class_hits  = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)

    for images, label_ids in loader:
        images    = images.to(DEVICE)
        label_ids = label_ids.to(DEVICE)

        logits = model(images, label_ids, augment_text=False)
        loss   = criterion(logits, label_ids)

        total_loss   += loss.item() * len(label_ids)
        correct_top1 += (logits.argmax(-1) == label_ids).sum().item()

        k    = min(5, num_classes)
        top5 = logits.topk(k, dim=-1).indices
        correct_top5 += sum(
            true.item() in row.tolist() for true, row in zip(label_ids, top5)
        )

        for pred, true in zip(logits.argmax(-1).cpu(), label_ids.cpu()):
            per_class_total[true] += 1
            if pred == true:
                per_class_hits[true] += 1

        n += len(label_ids)

    macro_acc = (per_class_hits / per_class_total.clamp(min=1)).mean().item()

    return {
        "loss":           total_loss / n,
        "top1_accuracy":  correct_top1 / n,
        "top5_accuracy":  correct_top5 / n,
        "macro_accuracy": macro_acc,
    }


# ── Load checkpoint helper (used by both train.py and evaluate.py) ────────────

def load_checkpoint(
    model:     MultimodalDiseaseClassifier,
    optimiser: torch.optim.Optimizer | None,
    path:      Path,
) -> tuple[int, dict]:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimiser and "optim_state_dict" in ckpt:
        optimiser.load_state_dict(ckpt["optim_state_dict"])
    print(f"Loaded checkpoint: {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt.get("metrics", {})


# ── Full evaluation run ───────────────────────────────────────────────────────

def run_evaluation(args):
    """
    Loads the best saved checkpoint for a dataset and prints a metrics report.
    Called by main.py when --eval_only is set.
    """
    cache = CachedTextEmbeddings(args.dataset, Path(args.cache_dir))
    _, val_loader, num_classes = build_dataloaders(args.dataset, cache)

    vit   = ViTEncoder(frozen_layers=args.frozen_layers)
    model = MultimodalDiseaseClassifier(vit, cache, num_classes).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / f"{args.dataset}_best.pt"
    load_checkpoint(model, None, ckpt_path)

    criterion = nn.CrossEntropyLoss()
    metrics   = evaluate(model, val_loader, criterion, cache)

    print(f"\nEvaluation — {args.dataset}")
    print(f"  Top-1 accuracy : {metrics['top1_accuracy']:.4f}")
    print(f"  Top-5 accuracy : {metrics['top5_accuracy']:.4f}")
    print(f"  Macro accuracy : {metrics['macro_accuracy']:.4f}")
    print(f"  Loss           : {metrics['loss']:.4f}")


# ── Single-image prediction ───────────────────────────────────────────────────

def run_predict(args):
    """
    Classifies a single image and prints the top-5 predicted disease classes.
    Called by main.py when --predict is set.

    At inference there is no ground-truth label, so we run the model once for
    every possible text prototype and take the class with the highest logit.
    """
    from PIL import Image

    cache       = CachedTextEmbeddings(args.dataset, Path(args.cache_dir))
    num_classes = cache.num_classes()

    vit   = ViTEncoder(frozen_layers=args.frozen_layers)
    model = MultimodalDiseaseClassifier(vit, cache, num_classes).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / f"{args.dataset}_best.pt"
    load_checkpoint(model, None, ckpt_path)
    model.eval()

    image = EVAL_TRANSFORMS(Image.open(args.predict).convert("RGB"))
    image = image.unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

    # Score each class by passing its text prototype through the model
    scores = []
    with torch.no_grad():
        for lid in range(num_classes):
            label_id = torch.tensor([lid], device=DEVICE)
            logits   = model(image, label_id)          # (1, num_classes)
            scores.append(logits[0, lid].item())

    scores = torch.tensor(scores)
    top5   = scores.topk(min(5, num_classes))

    print(f"\nImage  : {args.predict}")
    print(f"Dataset: {args.dataset}\n")
    print("Top-5 predictions:")
    for rank, (score, idx) in enumerate(
        zip(top5.values.tolist(), top5.indices.tolist()), start=1
    ):
        print(f"  {rank}. {cache.label_names[idx]:<40}  ({score:+.3f})")