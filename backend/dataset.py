from pathlib import Path
import json
from PIL import Image

from torch.utils.data import Dataset

class PlantWildDataset(Dataset):
    """
    Loads images from backend/data/images/plantwild/.
    Returns (label, index) pairs where label is an integer index of the class.
    """
 
    def __init__(self, images_dir: str, transform=None, label_map: dict = None):
        self.images_dir = Path(images_dir)
        self.transform  = transform
 
        class_dirs     = sorted([d for d in self.images_dir.iterdir() if d.is_dir()])
        self.label_map = label_map or {d.name: i for i, d in enumerate(class_dirs)}
        self.classes   = list(self.label_map.keys())
 
        self.samples = [
            (img_path, self.label_map[cls_dir.name])
            for cls_dir in class_dirs
            if cls_dir.name in self.label_map
            for img_path in sorted(cls_dir.glob("*.jpg"),
                                   key=lambda p: int(p.stem) if p.stem.isdigit() else float("inf"))
        ]
 
        print(f"PlantWildDataset | {len(self.classes)} classes | "
              f"{len(self.samples)} images")
 
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

dataset = PlantWildDataset("./data/images/plantwild")
dataset.save_label_map("./data/label_map.json")