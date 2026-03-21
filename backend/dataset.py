from pathlib import Path
import json
from PIL import Image

import torch
from torch.utils.data import Dataset

class PlantWildDataset(Dataset):
    """
    Args:
        images_dir (str) : Path to plantwild/ folder.
        transform        : torchvision transform pipeline.
        label_map (dict) : {class_name: int}. Auto-built from folders if None.
        split     (str)  : "train" | "test" | "all". Default "all".
        test_size (float): Fraction reserved for test. Default 0.2.
        seed      (int)  : Random seed for reproducible splits. Default 42.
    """
 
    def __init__(self, images_dir: str, transform=None, label_map: dict = None,
                 split: str = "all", test_size: float = 0.2, seed: int = 42):
        self.images_dir = Path(images_dir)
        self.transform  = transform
 
        class_dirs     = sorted([d for d in self.images_dir.iterdir() if d.is_dir()])
        self.label_map = label_map or {d.name: i for i, d in enumerate(class_dirs)}
        self.classes   = list(self.label_map.keys())
 
        all_samples = [
            (img_path, self.label_map[cls_dir.name])
            for cls_dir in class_dirs
            if cls_dir.name in self.label_map
            for img_path in sorted(cls_dir.glob("*.jpg"),
                                   key=lambda p: int(p.stem) if p.stem.isdigit() else float("inf"))
        ]
 
        if split == "all":
            self.samples = all_samples
        else:
            generator = torch.Generator().manual_seed(seed)
            indices   = torch.randperm(len(all_samples), generator=generator).tolist()
            n_test    = int(len(all_samples) * test_size)
            n_train   = len(all_samples) - n_test
            train_idx = indices[:n_train]
            test_idx  = indices[n_train:]
            self.samples = [all_samples[i] for i in (train_idx if split == "train" else test_idx)]
 
        print(f"PlantWildDataset | split={split} | "
              f"{len(self.classes)} classes | {len(self.samples)} images")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
 
    def save_label_map(self, path: str):
        with open(path, "w") as f:
            json.dump(self.label_map, f, indent=2)
        print(f"Label map saved → {path}")

    def get_class_counts(self) -> torch.Tensor:
        """Returns a tensor of per-class sample counts for weighted loss."""
        counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        return counts