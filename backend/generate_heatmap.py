import os
import cv2
import torch
import random
import json
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 1. TEAM IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
# Importing the exact architectures your team built from their scripts
from mlp import MultimodalMLP
from bert import FeatureExtractorModel

# ──────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE        = 320
NUM_CLASSES     = 89

IMAGES_DIR      = "./data/images/plantwild"
TEXT_DATA_DIR   = "./data/text"
LABEL_MAP_PATH  = "./data/label_map.json"

IMAGE_MODEL_PATH = "./checkpoints/best_image_encoder.pt"
TEXT_MODEL_PATH  = "./checkpoints/best_text_encoder.pt"
MLP_MODEL_PATH   = "./checkpoints/best_multimodal_mlp.pt"

# ──────────────────────────────────────────────────────────────────────────────
# 3. ADVANCED XAI PIPELINE (HiResCAM + Multi-Layer + Contour Mapping)
# ──────────────────────────────────────────────────────────────────────────────
class MultimodalGradCAM:
    def __init__(self):
        print(f"Device : {DEVICE}")
        print("Loading models...")
        
        with open(LABEL_MAP_PATH, 'r') as f:
            self.label_map = json.load(f)

        self.vit = timm.create_model("mobilevitv2_150", pretrained=False, num_classes=0, global_pool='')
        self.vit.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE), strict=False) 
        self.vit.to(DEVICE).eval()

        # Hook the last three stages to capture both high-res edges and deep semantics
        self.activations = {}
        self.gradients = {}
        
        def get_forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
            
        def get_backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        for i, layer_idx in enumerate([-3, -2, -1]):
            layer = self.vit.stages[layer_idx]
            layer.register_forward_hook(get_forward_hook(f"stage_{i}"))
            layer.register_full_backward_hook(get_backward_hook(f"stage_{i}"))

        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATH)
        bert_base = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
        
        # Instantiate your team's text model
        self.bert = FeatureExtractorModel(bert_base).to(DEVICE).eval()

        # Instantiate your team's fusion MLP
        self.mlp = MultimodalMLP(num_classes=NUM_CLASSES).to(DEVICE).eval()
        self.mlp.load_state_dict(torch.load(MLP_MODEL_PATH, map_location=DEVICE))

        self.transform = transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 1.15)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def generate_heatmap(self, image_path, text_description, true_disease_name):
        raw_image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(raw_image).unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad = True

        visual_transform = transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 1.15)),
            transforms.CenterCrop(IMG_SIZE)
        ])
        cropped_raw_image = visual_transform(raw_image)
        raw_image_np = np.array(cropped_raw_image) / 255.0

        encoded_text = self.tokenizer(text_description, padding=True, truncation=True, 
                                      max_length=128, return_tensors="pt").to(DEVICE)

        self.vit.zero_grad()
        self.mlp.zero_grad()
        
        spatial_features = self.vit(img_tensor) 
        pooled_img_emb = F.adaptive_avg_pool2d(spatial_features, (1, 1)).flatten(1)
        
        text_emb = self.bert(encoded_text["input_ids"], encoded_text["attention_mask"]).detach()
        logits = self.mlp(pooled_img_emb, text_emb)
        
        target_class = self.label_map.get(true_disease_name)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        print(f"Explaining True Class Index: {target_class}")

        score = logits[0, target_class]
        score.backward()

        # Multi-Spot HiResCAM Fusion
        fused_cam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        for name in self.activations.keys():
            act = self.activations[name]
            grad = self.gradients[name]
            
            cam = torch.sum(grad * act, dim=1).squeeze()
            cam = F.relu(cam).cpu().numpy()
            
            cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            fused_cam += cam 

        if fused_cam.max() > 0:
            fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)
        else:
            fused_cam = np.zeros_like(fused_cam)

        # --- CONTOUR MAPPING (The "AI Scanner" Look) ---
        base_leaf = raw_image_np * 0.8  

        hot_mask = np.uint8(fused_cam > 0.4) * 255
        
        contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * fused_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0 
        
        alpha_mask = (fused_cam > 0.2).astype(float)[..., None] 
        overlay = (alpha_mask * heatmap) + ((1.0 - alpha_mask) * base_leaf)
        
        overlay_uint8 = np.uint8(overlay * 255)
        cv2.drawContours(overlay_uint8, contours, -1, (0, 255, 0), 2)
        overlay = overlay_uint8 / 255.0

        return cropped_raw_image, heatmap, overlay

# ──────────────────────────────────────────────────────────────────────────────
# 4. RANDOM DATA SELECTOR
# ──────────────────────────────────────────────────────────────────────────────
def get_random_sample(images_base_dir, text_data_dir):
    class_dirs = [d for d in glob.glob(os.path.join(images_base_dir, '*')) if os.path.isdir(d)]
    random_class_dir = random.choice(class_dirs)
    disease_name = os.path.basename(random_class_dir)
    print(f"Selected random disease: {disease_name}")

    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in glob.glob(os.path.join(random_class_dir, '*.*')) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    random_image_path = random.choice(image_files)
    print(f"Selected random image  : {os.path.basename(random_image_path)}")

    all_descriptions = []
    json_files = glob.glob(os.path.join(text_data_dir, '*.json'))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if disease_name in data:
                    all_descriptions.extend(data[disease_name])
        except Exception:
            pass

    random_description = random.choice(all_descriptions)
    print(f"Selected random text   : {random_description[:75]}...") 

    return random_image_path, random_description, disease_name

# ──────────────────────────────────────────────────────────────────────────────
# 5. EXECUTION
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    explainer = MultimodalGradCAM()
    
    try:
        test_img_path, test_text, disease = get_random_sample(IMAGES_DIR, TEXT_DATA_DIR)
    except Exception as e:
        print(f"\nError loading random data: {e}")
        exit()
    
    print("\nGenerating Explanations...")
    original, heatmap, overlay = explainer.generate_heatmap(test_img_path, test_text, disease)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    axes[0].imshow(original)
    axes[0].set_title(f"Original Leaf\n({disease})")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title("xAI Heatmap\n(Multi-Layer HiResCAM)")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay\n(AI Scanner Contours)")
    axes[2].axis('off')
    
    max_len = 120
    text_snippet = (test_text[:max_len] + '..') if len(test_text) > max_len else test_text
    plt.suptitle(f"Explanation-Driven Multimodal Fusion for: {disease}\nInput Text: \"{text_snippet}\"", 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    safe_filename = disease.replace(' ', '_').replace('/', '_')
    save_path = os.path.join("heatmap_outputs", f"xai_result_{safe_filename}.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Saved final explainer visualization to {save_path}!")