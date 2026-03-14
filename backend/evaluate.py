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
    Runs the model over every batch in loader without using ground-truth labels
    to look up text embeddings.

    For each image the model is scored against EVERY class prototype and the
    argmax is taken — identical to how run_predict() works on a single image.

    Previously, true label_ids were passed directly into model(), which caused
    the text branch to always receive the correct class embedding, massively
    inflating all metrics.

    Returns
    -------
    dict with keys: loss, top1_accuracy, top5_accuracy, macro_accuracy
    """
    model.eval()

    num_classes     = cache.num_classes()
    total_loss      = 0.0
    correct_top1    = 0
    correct_top5    = 0
    n               = 0
    per_class_hits  = torch.zeros(num_classes)
    per_class_total = torch.zeros(num_classes)

    # Pre-stack all text prototypes once — (num_classes, 768)
    all_text_protos = cache.proto_matrix.to(DEVICE)

    for images, true_label_ids in loader:
        images         = images.to(DEVICE)
        true_label_ids = true_label_ids.to(DEVICE)
        B              = images.shape[0]

        # ── Image features (computed once per batch) ───────────────────────
        vit_feats = model.vit(images)   # (B, 768)

        # ── Score image against every class prototype ──────────────────────
        # For class c: feed (vit_feats, text_proto_c) → MLP → take logit[c]
        # This mirrors inference: no prior knowledge of the true class.
        class_scores = torch.zeros(B, num_classes, device=DEVICE)

        for c in range(num_classes):
            text_feats    = all_text_protos[c].unsqueeze(0).expand(B, -1)  # (B, 768)
            logits_c      = model.mlp(vit_feats, text_feats)               # (B, num_classes)
            class_scores[:, c] = logits_c[:, c]

        # ── Loss using true labels ─────────────────────────────────────────
        loss = criterion(class_scores, true_label_ids)
        total_loss += loss.item() * B

        # ── Metrics ───────────────────────────────────────────────────────
        preds = class_scores.argmax(-1)
        correct_top1 += (preds == true_label_ids).sum().item()

        k    = min(5, num_classes)
        top5 = class_scores.topk(k, dim=-1).indices
        correct_top5 += sum(
            true.item() in row.tolist()
            for true, row in zip(true_label_ids, top5)
        )

        for pred, true in zip(preds.cpu(), true_label_ids.cpu()):
            per_class_total[true] += 1
            if pred == true:
                per_class_hits[true] += 1

        n += B

    macro_acc = (per_class_hits / per_class_total.clamp(min=1)).mean().item()

    return {
        "loss":           total_loss / n,
        "top1_accuracy":  correct_top1 / n,
        "top5_accuracy":  correct_top5 / n,
        "macro_accuracy": macro_acc,
    }


# ── Load checkpoint helper ────────────────────────────────────────────────────

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
    Classifies a single image + user symptom text.

    Both modalities are required:
      --predict  path/to/image.jpg
      --text     "Leaves have brown spots and yellowing edges"

    The user's text is encoded live by BERT (not a prototype lookup),
    then fused with ViT image features by the MLP.
    """
    from PIL import Image
    from bert_encoder import LiveTextEncoder

    if not args.text:
        raise ValueError(
            "Symptom description required.\n"
            "Use: python main.py --predict image.jpg --text \"describe symptoms\""
        )

    cache       = CachedTextEmbeddings(args.dataset, Path(args.cache_dir))
    num_classes = cache.num_classes()

    vit   = ViTEncoder(frozen_layers=args.frozen_layers)
    model = MultimodalDiseaseClassifier(vit, cache, num_classes).to(DEVICE)

    ckpt_path = CHECKPOINT_DIR / f"{args.dataset}_best.pt"
    load_checkpoint(model, None, ckpt_path)
    model.eval()

    # ── Encode image ──────────────────────────────────────────────────────
    image = EVAL_TRANSFORMS(Image.open(args.predict).convert("RGB"))
    image = image.unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

    # ── Encode user text live with BERT ───────────────────────────────────
    print(f"Encoding text: \"{args.text}\"")
    text_encoder   = LiveTextEncoder()
    text_embedding = text_encoder.encode(args.text)   # (768,) on CPU

    # ── Single forward pass ───────────────────────────────────────────────
    with torch.no_grad():
        logits = model.forward_with_text(image, text_embedding)  # (1, num_classes)

    top5 = logits[0].topk(min(5, num_classes))

    print(f"\nImage  : {args.predict}")
    print(f"Text   : {args.text}")
    print(f"Dataset: {args.dataset}\n")
    print("Top-5 predictions:")
    for rank, (score, idx) in enumerate(
        zip(top5.values.tolist(), top5.indices.tolist()), start=1
    ):
        print(f"  {rank}. {cache.label_names[idx]:<40}  ({score:+.3f})")