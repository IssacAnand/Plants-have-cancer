"""
Step 1: Generate training data for the heatmap generator model.

For each training image, this script:
  1. Runs the MobileViT backbone → spatial_feat (C, H, W)
  2. Runs full HiResCAM (gradient-based) → ground-truth heatmap (1, H_out, W_out)
  3. Saves (spatial_feat, heatmap) pairs to disk

Usage:
  cd backend
  python generate_heatmap_data.py

Output:
  ./checkpoints/heatmap_training_data/
    spatial_feats.pt   — tensor of shape (N, C, H_s, W_s)
    heatmaps.pt        — tensor of shape (N, 1, 320, 320)
"""

import os
import json
import glob
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from mlp import MultimodalMLP
from bert import FeatureExtractorModel

# ── Configuration ─────────────────────────────────────────────────────────────

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE        = 320
NUM_CLASSES     = 89

IMAGES_DIR      = "./data/images/plantwild"
TEXT_DATA_DIR   = "./data/text"
LABEL_MAP_PATH  = "./data/label_map.json"

IMAGE_MODEL_PATH = "./checkpoints/best_image_encoder.pt"
TEXT_MODEL_PATH  = "./checkpoints/best_text_encoder.pt"
MLP_MODEL_PATH   = "./checkpoints/best_multimodal_mlp.pt"

OUTPUT_DIR       = "./checkpoints/heatmap_training_data"
MAX_SAMPLES      = 2000  # adjust based on your dataset size and time

# ── Model setup ───────────────────────────────────────────────────────────────

class HeatmapDataGenerator:
    def __init__(self):
        print(f"Device: {DEVICE}")
        print("Loading models...")

        with open(LABEL_MAP_PATH, 'r') as f:
            self.label_map = json.load(f)

        # Image backbone (MobileViTv2)
        self.vit = timm.create_model("mobilevitv2_150", pretrained=False,
                                     num_classes=0, global_pool='')
        self.vit.load_state_dict(
            torch.load(IMAGE_MODEL_PATH, map_location=DEVICE), strict=False
        )
        self.vit.to(DEVICE).eval()

        # Hook the last three stages for multi-layer HiResCAM
        self.activations = {}
        self.gradients = {}

        def get_forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook

        def get_backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook

        for i, layer_idx in enumerate([-3, -2, -1]):
            layer = self.vit.stages[layer_idx]
            layer.register_forward_hook(get_forward_hook(f"stage_{i}"))
            layer.register_full_backward_hook(get_backward_hook(f"stage_{i}"))

        # Text encoder (agriculture-BERT)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
        bert_base = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
        self.bert = FeatureExtractorModel(bert_base).to(DEVICE).eval()

        # Fusion MLP
        self.mlp = MultimodalMLP(num_classes=NUM_CLASSES).to(DEVICE).eval()
        self.mlp.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=DEVICE))

        self.transform = transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 1.15)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        print("Models loaded.\n")

    def generate_one(self, image_path, text_description, disease_name):
        """
        Returns:
            spatial_feat: (C, H, W) tensor from backbone forward pass
            heatmap:      (1, IMG_SIZE, IMG_SIZE) normalised gradient-based heatmap
        """
        raw_image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(raw_image).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad = True

        encoded_text = self.tokenizer(
            text_description, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        ).to(DEVICE)

        self.vit.zero_grad()
        self.mlp.zero_grad()

        # Forward pass
        spatial_features = self.vit(img_tensor)
        pooled_img_emb = F.adaptive_avg_pool2d(spatial_features, (1, 1)).flatten(1)

        text_emb = self.bert(
            encoded_text["input_ids"], encoded_text["attention_mask"]
        ).detach()
        logits = self.mlp(pooled_img_emb, text_emb)

        # Get target class
        target_class = self.label_map.get(disease_name)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass for gradients
        score = logits[0, target_class]
        score.backward()

        # Multi-layer HiResCAM fusion
        fused_cam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        for name in self.activations.keys():
            act = self.activations[name]
            grad = self.gradients[name]
            cam = torch.sum(grad * act, dim=1).squeeze()
            cam = F.relu(cam).cpu().detach().numpy()
            cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            fused_cam += cam

        # Normalise to [0, 1]
        if fused_cam.max() > 0:
            fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)
        else:
            fused_cam = np.zeros_like(fused_cam)

        # Get spatial_feat from forward pass (detach, move to CPU)
        spatial_feat = spatial_features.squeeze(0).detach().cpu()  # (C, H, W)

        # Heatmap as tensor (1, IMG_SIZE, IMG_SIZE)
        heatmap = torch.from_numpy(fused_cam).unsqueeze(0)  # (1, 320, 320)

        return spatial_feat, heatmap


def collect_samples(images_dir, text_data_dir):
    """Collect (image_path, text, disease_name) tuples from the dataset."""
    samples = []

    # Load all text descriptions
    all_descriptions = {}
    json_files = glob.glob(os.path.join(text_data_dir, '*.json'))
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for disease, descs in data.items():
                    if disease not in all_descriptions:
                        all_descriptions[disease] = []
                    all_descriptions[disease].extend(descs)
        except Exception:
            pass

    # Iterate class directories
    class_dirs = [d for d in glob.glob(os.path.join(images_dir, '*')) if os.path.isdir(d)]

    for class_dir in class_dirs:
        disease_name = os.path.basename(class_dir)
        if disease_name not in all_descriptions:
            continue

        valid_extensions = {".jpg", ".jpeg", ".png"}
        image_files = [f for f in glob.glob(os.path.join(class_dir, '*.*'))
                       if os.path.splitext(f)[1].lower() in valid_extensions]

        descs = all_descriptions[disease_name]
        for img_path in image_files:
            text = random.choice(descs)
            samples.append((img_path, text, disease_name))

    return samples


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generator = HeatmapDataGenerator()

    print("Collecting samples...")
    samples = collect_samples(IMAGES_DIR, TEXT_DATA_DIR)
    random.shuffle(samples)

    if MAX_SAMPLES and len(samples) > MAX_SAMPLES:
        samples = samples[:MAX_SAMPLES]

    print(f"Generating heatmap data for {len(samples)} images...\n")

    all_spatial_feats = []
    all_heatmaps = []
    skipped = 0

    for img_path, text, disease in tqdm(samples, desc="Generating"):
        try:
            spatial_feat, heatmap = generator.generate_one(img_path, text, disease)
            all_spatial_feats.append(spatial_feat)
            all_heatmaps.append(heatmap)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Skipped {os.path.basename(img_path)}: {e}")

    print(f"\nDone. {len(all_spatial_feats)} pairs generated, {skipped} skipped.")

    # Stack and save
    spatial_feats_tensor = torch.stack(all_spatial_feats)  # (N, C, H_s, W_s)
    heatmaps_tensor = torch.stack(all_heatmaps)            # (N, 1, 320, 320)

    print(f"  spatial_feats shape: {spatial_feats_tensor.shape}")
    print(f"  heatmaps shape:      {heatmaps_tensor.shape}")

    torch.save(spatial_feats_tensor, os.path.join(OUTPUT_DIR, "spatial_feats.pt"))
    torch.save(heatmaps_tensor, os.path.join(OUTPUT_DIR, "heatmaps.pt"))

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
