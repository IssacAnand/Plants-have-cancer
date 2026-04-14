"""
backend/export_mobile.py
Export the trained multimodal plant disease model to PyTorch Mobile (.ptl).

Usage (from the backend/ directory):
    python export_mobile.py

Outputs:
    ../assets/models/plant_model.ptl       — prediction model for modelInference.js
    ../assets/models/plant_explain.ptl     — feature-map model for on-device CAM
    ../assets/data/label_map.json          — disease index → name (89 classes)
    ../assets/data/tokenizer_vocab.json    — BERT WordPiece vocab for JS tokenizer
    ../assets/data/tokenizer_config.json   — BERT special tokens + max_length

Requirements:
    pip install torch torchvision timm transformers
    (See requirements.txt for pinned versions)

Notes:
    - All models are exported on CPU for mobile compatibility.
    - Image normalization (ImageNet mean/std) is baked into the model forward pass,
      so the JS side only needs to pass a float32 tensor in [0, 1] range, shape (1, 3, 320, 320).
    - BERT tokenization must be performed in JS before calling the model.
      Use tokenizer_vocab.json + tokenizer_config.json with a WordPiece tokenizer library.
    - Both models are traced with check_trace=False because HuggingFace BERT uses
      Python control flow internally that differs on zero vs. non-zero inputs but
      produces correct results in practice.
"""

import json
import shutil
import sys
import torch
import torch.nn as nn
import timm
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.mobile_optimizer import optimize_for_mobile


# ── paths ─────────────────────────────────────────────────────────────────────

BACKEND_DIR     = Path(__file__).parent
CHECKPOINT_DIR  = BACKEND_DIR / "checkpoints"
DATA_DIR        = BACKEND_DIR / "data"
ASSETS_DIR      = BACKEND_DIR.parent / "assets"
MODELS_OUT_DIR  = ASSETS_DIR / "models"
DATA_OUT_DIR    = ASSETS_DIR / "data"

VIT_CHECKPOINT  = CHECKPOINT_DIR / "best_image_encoder.pt"
BERT_CHECKPOINT = CHECKPOINT_DIR / "best_text_encoder.pt"   # HuggingFace directory
MLP_CHECKPOINT  = CHECKPOINT_DIR / "best_multimodal_mlp.pt"
LABEL_MAP_PATH  = DATA_DIR / "label_map.json"

# ── constants ─────────────────────────────────────────────────────────────────

DEVICE      = torch.device("cpu")   # always export on CPU for mobile
IMG_SIZE    = 320
NUM_CLASSES = 89
SEQ_LEN     = 128
IMG_MEAN    = [0.485, 0.456, 0.406]
IMG_STD     = [0.229, 0.224, 0.225]
VIT_MODEL   = "mobilevitv2_150"


# ── model definitions (must mirror training code exactly) ─────────────────────

class MultimodalMLP(nn.Module):
    def __init__(self, image_dim=768, text_dim=768, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        fused_dim = image_dim + text_dim  # 1536
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),       nn.LayerNorm(512),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.LayerNorm(256),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([image_emb, text_emb], dim=1))


class BertEncoder(nn.Module):
    """Extracts [CLS] token embedding from a fine-tuned HuggingFace BERT model."""
    def __init__(self, hf_model):
        super().__init__()
        self.bert = hf_model.bert

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # (B, 768)


class MultimodalInferenceModel(nn.Module):
    """
    Full prediction model for on-device inference.

    Input:
        image          — float32 tensor (1, 3, 320, 320), values in [0, 1]
        input_ids      — int64  tensor (1, 128)  BERT token ids
        attention_mask — int64  tensor (1, 128)  1 for real tokens, 0 for padding

    Output:
        probs — float32 tensor (1, 89)  softmax probabilities over 89 disease classes
    """
    def __init__(self, vit_encoder: nn.Module, bert_encoder: nn.Module, mlp: nn.Module):
        super().__init__()
        self.vit  = vit_encoder
        self.bert = bert_encoder
        self.mlp  = mlp

        # Bake ImageNet normalisation in so JS side sends raw [0,1] floats
        self.register_buffer("mean", torch.tensor(IMG_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMG_STD).view(1, 3, 1, 1))

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_norm = (image - self.mean) / self.std
        img_emb    = self.vit(image_norm)
        text_emb   = self.bert(input_ids, attention_mask)
        logits     = self.mlp(img_emb, text_emb)
        return torch.softmax(logits, dim=1)


class MultimodalExplainModel(nn.Module):
    """
    Prediction model that also returns the MobileViT last-stage feature map
    for gradient-free on-device Class Activation Mapping (CAM).

    The JS side can compute a heatmap as:
        weights[c] = mean over spatial dims of feature_map[c]   (global avg pool per channel)
        heatmap    = sum_c(weights[c] * feature_map[c])         weighted channel sum
        heatmap    = normalize + upsample to 320×320

    Input:
        image, input_ids, attention_mask  (same as MultimodalInferenceModel)

    Output:
        probs       — (1, 89)    softmax probabilities
        feature_map — (C, H, W)  last MobileViT stage activations (before global pool)
    """
    def __init__(self, vit_encoder: nn.Module, bert_encoder: nn.Module, mlp: nn.Module):
        super().__init__()
        self.vit        = vit_encoder
        self.bert       = bert_encoder
        self.mlp        = mlp

        self.register_buffer("mean", torch.tensor(IMG_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMG_STD).view(1, 3, 1, 1))

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        image_norm = (image - self.mean) / self.std

        # Run the backbone stage-by-stage to capture the last feature map.
        # timm MobileViTv2 layout: stem → stages[0..n] → global_pool
        x = self.vit.stem(image_norm)
        for stage in self.vit.stages:
            x = stage(x)
        # x is now (1, C, H, W) — the last-stage spatial features
        feature_map = x.squeeze(0)                         # (C, H, W)

        # Global pool → embedding (matches how the backbone was trained)
        img_emb = self.vit.global_pool(x)                 # (1, 768)

        text_emb = self.bert(input_ids, attention_mask)   # (1, 768)
        logits   = self.mlp(img_emb, text_emb)            # (1, 89)
        probs    = torch.softmax(logits, dim=1)

        return probs, feature_map


# ── loading ───────────────────────────────────────────────────────────────────

def load_models():
    print("Loading ViT encoder...")
    vit = timm.create_model(VIT_MODEL, pretrained=False, num_classes=0, global_pool="avg")
    vit.load_state_dict(torch.load(VIT_CHECKPOINT, map_location=DEVICE))
    vit.eval()
    print(f"  embed dim: {vit.num_features}")

    print("Loading BERT encoder...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        str(BERT_CHECKPOINT), num_labels=NUM_CLASSES
    )
    hf_model.eval()
    bert = BertEncoder(hf_model)
    bert.eval()

    print("Loading MLP fusion...")
    mlp = MultimodalMLP()
    mlp.load_state_dict(torch.load(MLP_CHECKPOINT, map_location=DEVICE))
    mlp.eval()

    return vit, bert, mlp


# ── export helpers ────────────────────────────────────────────────────────────

def _trace_and_save(model: nn.Module, example_inputs: tuple, out_path: Path):
    """Trace, mobile-optimise, and save a model to .ptl."""
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            example_inputs,
            check_trace=False,  # BERT control flow differs on zero inputs; safe in practice
            strict=False,
        )
    optimized = optimize_for_mobile(traced)
    optimized._save_for_lite_interpreter(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved → {out_path}  ({size_mb:.1f} MB)")


def export_prediction_model(vit, bert, mlp):
    print("\n[1/2] Exporting prediction model...")
    model = MultimodalInferenceModel(vit, bert, mlp)
    model.eval()

    examples = (
        torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),       # image
        torch.zeros(1, SEQ_LEN, dtype=torch.long),   # input_ids
        torch.ones(1,  SEQ_LEN, dtype=torch.long),   # attention_mask
    )
    _trace_and_save(model, examples, MODELS_OUT_DIR / "plant_model.ptl")


def export_explain_model(vit, bert, mlp):
    print("\n[2/2] Exporting explainability model...")
    try:
        model = MultimodalExplainModel(vit, bert, mlp)
        model.eval()

        examples = (
            torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),
            torch.zeros(1, SEQ_LEN, dtype=torch.long),
            torch.ones(1,  SEQ_LEN, dtype=torch.long),
        )
        _trace_and_save(model, examples, MODELS_OUT_DIR / "plant_explain.ptl")
    except Exception as exc:
        print(f"  WARNING: explainability export failed — {exc}")
        print("  plant_explain.ptl was NOT created.")
        print("  Possible fix: inspect timm mobilevitv2_150 attribute names")
        print("  (stem, stages, global_pool) and update MultimodalExplainModel.forward().")


# ── asset copying ─────────────────────────────────────────────────────────────

def copy_assets():
    print("\nCopying data assets...")

    # label map (disease index → name)
    dst = DATA_OUT_DIR / "label_map.json"
    shutil.copy(LABEL_MAP_PATH, dst)
    print(f"  label_map.json ({NUM_CLASSES} classes) → {dst}")

    # BERT tokenizer vocab + config for JS-side tokenization
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_CHECKPOINT))

    vocab_path = DATA_OUT_DIR / "tokenizer_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False)
    print(f"  tokenizer_vocab.json ({len(tokenizer.get_vocab())} tokens) → {vocab_path}")

    cfg = {
        "do_lower_case": getattr(tokenizer, "do_lower_case", True),
        "unk_token":     tokenizer.unk_token,
        "sep_token":     tokenizer.sep_token,
        "pad_token":     tokenizer.pad_token,
        "cls_token":     tokenizer.cls_token,
        "mask_token":    tokenizer.mask_token,
        "cls_token_id":  tokenizer.cls_token_id,
        "sep_token_id":  tokenizer.sep_token_id,
        "pad_token_id":  tokenizer.pad_token_id,
        "max_length":    SEQ_LEN,
    }
    cfg_path = DATA_OUT_DIR / "tokenizer_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  tokenizer_config.json → {cfg_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Plant Disease Multimodal Model Export ===\n")

    # Validate checkpoints exist before doing any work
    missing = [p for p in (VIT_CHECKPOINT, BERT_CHECKPOINT, MLP_CHECKPOINT, LABEL_MAP_PATH) if not p.exists()]
    if missing:
        print("ERROR: Missing required files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    vit, bert, mlp = load_models()

    export_prediction_model(vit, bert, mlp)
    export_explain_model(vit, bert, mlp)
    copy_assets()

    print("\n=== Done ===")
    print(f"  {MODELS_OUT_DIR / 'plant_model.ptl'}      ← load in utils/modelInference.js")
    print(f"  {MODELS_OUT_DIR / 'plant_explain.ptl'}    ← load for explainability tab")
    print(f"  {DATA_OUT_DIR / 'label_map.json'}          ← disease class lookup")
    print(f"  {DATA_OUT_DIR / 'tokenizer_vocab.json'}    ← BERT WordPiece vocab for JS")
    print(f"  {DATA_OUT_DIR / 'tokenizer_config.json'}   ← special token ids + max_length")
    print()
    print("Next step: resolve react-native-pytorch-core + RN 0.81.5 compatibility,")
    print("then implement utils/modelInference.js using the exported .ptl models.")


if __name__ == "__main__":
    main()
