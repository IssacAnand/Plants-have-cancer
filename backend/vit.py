"""
vit.py
------
  ViTEncoder   — wraps a HuggingFace ViT-Base pretrained on ImageNet,
                 fine-tuned on PlantVillage (or loaded from a checkpoint).
                 Always outputs a single (B, 768) L2-normalised vector.

Imported by multimodal.py. Not used standalone.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from torchvision import datasets, transforms

# ── Constants ─────────────────────────────────────────────────────────────────

VIT_MODEL   = "google/vit-base-patch16-224"
VIT_DIM     = 768
IMAGES_ROOT = Path("data/images")  # data/images/plantvillage/, data/images/plantdoc/, ...

def get_image_dir(dataset_name: str) -> Path:
    """
    Returns the image folder for a given dataset name.
      plantvillage → data/images/plantvillage/
      plantdoc     → data/images/plantdoc/
      plantwild    → data/images/plantwild/
    """
    p = IMAGES_ROOT / dataset_name
    if not p.exists():
        raise FileNotFoundError(
            f"Image directory not found: {p}\n"
            f"Expected: data/images/{dataset_name}/<class_name>/<images>"
        )
    return p

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")

# ── Image transforms ──────────────────────────────────────────────────────────

# ViT-Base was pretrained with ImageNet normalisation at 224×224.
# Keep these values fixed — do not change mean/std.
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

def get_dataset(
    dataset_name: str,
    train: bool      = True,
    val_split: float = 0.15,
    seed: int        = 42,
):
    """
    Load images from data/images/<dataset_name>/ using torchvision ImageFolder.
    Each subfolder name is treated as a class label.

    data/images/
      plantvillage/
        Apple___Apple_scab/
          img001.jpg ...
      plantdoc/
        Apple Scab Leaf/
          img001.jpg ...
      plantwild/
        apple black rot/
          img001.jpg ...

    Returns (train_subset, val_subset, class_names) when train=True,
            (full_dataset, class_names)              when train=False.
    """
    from torch.utils.data import Subset
    import random
    from collections import defaultdict

    image_dir = get_image_dir(dataset_name)

    # Two separate ImageFolder objects — same files, different transforms.
    # Ensures val images are never augmented even when called with train=True.
    train_full = datasets.ImageFolder(str(image_dir), transform=TRAIN_TRANSFORMS)
    eval_full  = datasets.ImageFolder(str(image_dir), transform=EVAL_TRANSFORMS)

    if not train:
        return eval_full, eval_full.classes

    # Stratified split — indices computed once, applied to both folders
    random.seed(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(train_full.samples):
        by_class[label].append(idx)

    train_idx, val_idx = [], []
    for indices in by_class.values():
        random.shuffle(indices)
        split = max(1, int(len(indices) * (1 - val_split)))
        train_idx.extend(indices[:split])
        val_idx.extend(indices[split:])

    # Train subset → augmented   |   Val subset → clean eval transforms
    return (
        Subset(train_full, train_idx),
        Subset(eval_full,  val_idx),
        train_full.classes,
    )


# ── ViTEncoder ────────────────────────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """
    ViT-Base image encoder.

    Input  : (B, 3, 224, 224)  — standard ImageNet normalisation expected
    Output : (B, 768)          — L2-normalised CLS token embedding

    Freezing strategy
    -----------------
    frozen_layers=0   → all ViT layers train  (default for fine-tuning)
    frozen_layers=N   → bottom N blocks frozen, top (12-N) blocks train
    frozen_layers=12  → entire ViT frozen (feature extractor only)
    """

    def __init__(
        self,
        model_name: str    = VIT_MODEL,
        frozen_layers: int = 0,
        checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self._apply_freeze(frozen_layers)

        if checkpoint:
            self.load_checkpoint(checkpoint)

    def _apply_freeze(self, frozen_layers: int):
        if frozen_layers == 0:
            return  # everything trains

        # Always freeze patch embeddings
        for param in self.vit.embeddings.parameters():
            param.requires_grad = False

        # Freeze the bottom N transformer blocks
        for i, block in enumerate(self.vit.encoder.layer):
            if i < frozen_layers:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values : (B, 3, 224, 224)
        returns      : (B, 768)  L2-normalised CLS token
        """
        out       = self.vit(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0, :]       # (B, 768)
        return F.normalize(cls_token, p=2, dim=-1)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Accept either a raw state_dict or a training checkpoint dict
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        if "vit_state_dict" in state:
            state = state["vit_state_dict"]
        self.vit.load_state_dict(state, strict=False)
        print(f"ViT checkpoint loaded from {path}")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)