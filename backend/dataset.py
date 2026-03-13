"""
dataset.py
----------
  build_label_map()   — maps folder integer ids → cache integer ids
  MappedSubset        — applies the label map inside __getitem__
  build_dataloaders() — returns (train_loader, val_loader, num_classes)
"""

import re

import torch
from torch.utils.data import DataLoader

from bert_encoder import CachedTextEmbeddings
from vit import get_dataset

# ── Config ────────────────────────────────────────────────────────────────────

BATCH_SIZE  = 32
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")


# ── Label mapping ─────────────────────────────────────────────────────────────

def build_label_map(
    folder_classes: list[str],
    cache: CachedTextEmbeddings,
) -> torch.Tensor:
    """
    ImageFolder sorts folder names alphabetically and assigns integer ids:
        0 → "Apple___Apple_scab"
        1 → "Apple___Black_rot"  ...

    CachedTextEmbeddings sorts JSON keys alphabetically and assigns integer ids:
        0 → "Apple Apple scab"
        1 → "Apple Black rot"  ...

    Normalisation applied to folder names before matching:
        "Apple___Apple_scab"          →  "Apple Apple scab"
        "Corn_(maize)___Common_rust_" →  "Corn (maize) Common rust"

    Returns a (num_folder_classes,) long tensor where tensor[folder_id] = cache_id.
    """
    def normalise(name: str) -> str:
        return re.sub(r"_+", " ", name).strip()

    mapping = torch.zeros(len(folder_classes), dtype=torch.long)
    missing = []

    for folder_id, folder_name in enumerate(folder_classes):
        norm = normalise(folder_name)
        if norm in cache.label2id:
            mapping[folder_id] = cache.label2id[norm]
        else:
            # Case-insensitive fallback
            match = next(
                (l for l in cache.label_names if l.lower() == norm.lower()),
                None,
            )
            if match:
                mapping[folder_id] = cache.label2id[match]
            else:
                missing.append(f"  '{folder_name}'  →  '{norm}'  (not in cache)")

    if missing:
        raise ValueError(
            "Label mismatch — these image folders have no matching JSON class:\n"
            + "\n".join(missing)
            + "\n\nCache labels:\n"
            + "\n".join(f"  {l}" for l in cache.label_names)
        )

    return mapping


# ── MappedSubset ──────────────────────────────────────────────────────────────

class MappedSubset(torch.utils.data.Dataset):
    """
    Wraps a torchvision Subset and remaps folder label ids → cache label ids
    on the fly so the DataLoader yields (image, cache_label_id) pairs.
    """

    def __init__(self, subset, label_map: torch.Tensor):
        self.subset    = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, folder_label = self.subset[idx]
        return image, self.label_map[folder_label].item()


# ── build_dataloaders ─────────────────────────────────────────────────────────

def build_dataloaders(
    dataset_name: str,
    cache: CachedTextEmbeddings,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Builds train and val DataLoaders for a given dataset.
    Returns (train_loader, val_loader, num_classes).
    """
    train_subset, val_subset, folder_classes = get_dataset(
        dataset_name=dataset_name, train=True
    )

    label_map   = build_label_map(folder_classes, cache)
    num_classes = cache.num_classes()

    train_ds = MappedSubset(train_subset, label_map)
    val_ds   = MappedSubset(val_subset,   label_map)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")
    print(f"Classes       : {num_classes}")

    return train_loader, val_loader, num_classes