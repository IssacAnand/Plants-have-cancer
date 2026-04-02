"""
Export the trained multimodal plant classifier to three ONNX models
for fully offline mobile inference.

Models exported
───────────────
  image_backbone.onnx  — MobileViTv2_150; pooled embedding + spatial feature map
  text_encoder.onnx    — fine-tuned agriculture-BERT; returns [CLS] embedding
                         (INT8-quantised to ~105 MB if onnxruntime is available)
  mlp.onnx             — MultimodalMLP fusion head; returns 89-class logits

Asset files
───────────
  label_map.json           — {class_name: int} index
  treatments.json          — {class_name: description_string}
  tokenizer/vocab.json     — {token: id} WordPiece vocabulary for the JS tokenizer

On-device inference flow (modelInference.js)
────────────────────────────────────────────
  user image  → image_backbone.onnx → img_emb (1,768) + spatial_feat (1,C,H,W)
  user text   → JS tokenizer        → input_ids / attention_mask
               → text_encoder.onnx  → text_emb (1,768)
  both        → mlp.onnx            → logits (1,89) → argmax → class label
  spatial_feat                      → mean-CAM → jet colormap → heatmap JPEG
"""

import json
import shutil
import sys
from pathlib import Path

# Windows consoles default to cp1252; PyTorch 2.10 dynamo exporter prints emoji
# that crash on non-UTF-8 terminals.  Force UTF-8 before any torch import.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModelForSequenceClassification

# ── Paths ─────────────────────────────────────────────────────────────────────

CHECKPOINTS = Path("./checkpoints")
DATA_DIR    = Path("./data")
OUT_DIR     = Path("../assets/models")

IMG_SIZE    = 320
NUM_CLASSES = 89
TEXT_MODEL  = CHECKPOINTS / "best_text_encoder.pt"   # HuggingFace directory


# ── Model definitions (must mirror the originals exactly) ─────────────────────

class MultimodalMLP(nn.Module):
    def __init__(self, image_dim=768, text_dim=768,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        fused_dim = image_dim + text_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),       nn.LayerNorm(512),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.LayerNorm(256),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image_emb, text_emb):
        return self.mlp(torch.cat([image_emb, text_emb], dim=1))


class FeatureExtractorModel(nn.Module):
    """
    Extracts the [CLS] token embedding from the fine-tuned BERT model.
    Must mirror bert.py exactly.  return_dict=False avoids tracing issues.
    """
    def __init__(self, original_model):
        super().__init__()
        self.bert = original_model.bert

    def forward(self, input_ids, attention_mask):
        # return_dict=False → tuple output; outputs[0] is last_hidden_state
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return outputs[0][:, 0, :]   # [CLS] token, shape (batch, 768)


class ImageBackboneWrapper(nn.Module):
    """
    Wraps MobileViTv2 (global_pool='') to return BOTH:
      img_emb     — (1, 768) pooled embedding for the MLP
      spatial_feat — (1, C, H, W) spatial map for the gradient-free CAM heatmap
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone   # global_pool='' → forward() returns (B, C, H, W)

    def forward(self, image: torch.Tensor):
        spatial = self.backbone(image)                               # (B, C, H, W)
        pooled  = F.adaptive_avg_pool2d(spatial, (1, 1)).flatten(1) # (B, C)
        return pooled, spatial


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_treatments(plantwild_json: Path, label_map: dict) -> dict:
    with open(plantwild_json, encoding="utf-8") as f:
        raw = json.load(f)
    treatments = {}
    for cls_name in label_map:
        descriptions = raw.get(cls_name, [])
        treatments[cls_name] = (
            str(descriptions[0]) if descriptions
            else "Consult a local agricultural expert for diagnosis and treatment advice."
        )
    return treatments


def quantize_onnx(fp32_path: Path, int8_path: Path) -> bool:
    """Quantise an ONNX model to INT8 weights (~4× smaller). Returns True on success."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        return True
    except ImportError:
        print("  onnxruntime.quantization not available — skipping quantization")
        return False
    except Exception as e:
        print(f"  Quantization failed: {e} — using FP32")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tokenizer").mkdir(exist_ok=True)

    # ── Clean up any leftover dynamo external-data sidecar files ─────────────
    for stale in OUT_DIR.glob("*.onnx.data"):
        stale.unlink()
        print(f"  Removed stale sidecar: {stale.name}")

    # ── Validate checkpoints ──────────────────────────────────────────────────
    for p in [CHECKPOINTS / "best_image_encoder.pt",
              CHECKPOINTS / "best_multimodal_mlp.pt",
              TEXT_MODEL]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\n  Run mobilevit.py, bert.py, and mlp.py first."
            )
    label_map_path = DATA_DIR / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"Missing: {label_map_path}\n  Run mobilevit.py first."
        )

    # ── 1. Image backbone ─────────────────────────────────────────────────────
    print("\n[1/5] Exporting image backbone …")
    backbone = timm.create_model(
        "mobilevitv2_150", pretrained=False, num_classes=0, global_pool=""
    )
    # Weights were trained with global_pool='avg'; strict=False drops the pool layer
    missing, _ = backbone.load_state_dict(
        torch.load(CHECKPOINTS / "best_image_encoder.pt", map_location="cpu"),
        strict=False,
    )
    if missing:
        print(f"  Missing keys (expected for pool change): {len(missing)} keys")
    backbone.eval()

    img_wrapper = ImageBackboneWrapper(backbone)
    img_wrapper.eval()

    dummy_image = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        test_pooled, test_spatial = img_wrapper(dummy_image)
    print(f"  img_emb      : {tuple(test_pooled.shape)}")
    print(f"  spatial_feat : {tuple(test_spatial.shape)}")

    img_onnx = OUT_DIR / "image_backbone.onnx"
    torch.onnx.export(
        img_wrapper, dummy_image, str(img_onnx),
        input_names=["image"],
        output_names=["img_emb", "spatial_feat"],
        dynamic_axes={"image":        {0: "batch"},
                      "img_emb":      {0: "batch"},
                      "spatial_feat": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  saved ({img_onnx.stat().st_size / 1e6:.1f} MB)  ✓")

    # ── 2. MultimodalMLP ──────────────────────────────────────────────────────
    print("\n[2/5] Exporting MultimodalMLP …")
    mlp = MultimodalMLP()
    mlp.load_state_dict(
        torch.load(CHECKPOINTS / "best_multimodal_mlp.pt", map_location="cpu")
    )
    mlp.eval()

    dummy_img_emb  = torch.randn(1, 768)
    dummy_text_emb = torch.randn(1, 768)
    mlp_onnx       = OUT_DIR / "mlp.onnx"
    torch.onnx.export(
        mlp, (dummy_img_emb, dummy_text_emb), str(mlp_onnx),
        input_names=["img_emb", "text_emb"],
        output_names=["logits"],
        dynamic_axes={"img_emb":  {0: "batch"},
                      "text_emb": {0: "batch"},
                      "logits":   {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  saved ({mlp_onnx.stat().st_size / 1e6:.1f} MB)  ✓")

    # ── 3. Text encoder (agriculture-BERT) ───────────────────────────────────
    print("\n[3/5] Exporting text encoder (420 MB → ~105 MB after quantisation) …")
    bert_base = AutoModelForSequenceClassification.from_pretrained(str(TEXT_MODEL))
    bert_feat = FeatureExtractorModel(bert_base)
    bert_feat.eval()

    SEQ_LEN    = 128
    dummy_ids  = torch.ones(1, SEQ_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

    text_onnx_fp32 = OUT_DIR / "text_encoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            bert_feat, (dummy_ids, dummy_mask), str(text_onnx_fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["text_emb"],
            dynamic_axes={"input_ids":      {0: "batch"},
                          "attention_mask": {0: "batch"},
                          "text_emb":       {0: "batch"}},
            opset_version=14,   # BERT requires >= 11; 14 is stable
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  FP32 saved ({text_onnx_fp32.stat().st_size / 1e6:.1f} MB)")

    # INT8 quantisation — ~4× size reduction, ~2× inference speedup on CPU
    text_onnx_int8 = OUT_DIR / "text_encoder_int8.onnx"
    if quantize_onnx(text_onnx_fp32, text_onnx_int8):
        print(f"  INT8 saved ({text_onnx_int8.stat().st_size / 1e6:.1f} MB)  ✓")
        text_onnx_fp32.unlink()
        text_onnx_int8.rename(text_onnx_fp32)
        print("  Replaced FP32 with INT8  ✓")
    else:
        print("  Keeping FP32 model (install onnxruntime to enable quantization)")

    # ── 4. Tokenizer vocabulary ───────────────────────────────────────────────
    print("\n[4/5] Extracting WordPiece vocabulary …")
    tokenizer_json_path = TEXT_MODEL / "tokenizer.json"
    with open(tokenizer_json_path, encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # The vocabulary is nested at model.vocab as {token: id}
    vocab = tokenizer_data["model"]["vocab"]
    vocab_dst = OUT_DIR / "tokenizer" / "vocab.json"
    with open(vocab_dst, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  vocab.json ({len(vocab)} tokens, "
          f"{vocab_dst.stat().st_size / 1e3:.0f} KB)  ✓")
    print(f"  Special tokens: [PAD]={vocab.get('[PAD]')}, "
          f"[CLS]={vocab.get('[CLS]')}, [SEP]={vocab.get('[SEP]')}, "
          f"[UNK]={vocab.get('[UNK]')}")

    # ── 5. Heatmap generator CNN ─────────────────────────────────────────────
    heatmap_ckpt = CHECKPOINTS / "best_heatmap_generator.pt"
    if heatmap_ckpt.exists():
        print("\n[5/7] Exporting heatmap generator …")

        # Import the model class
        sys.path.insert(0, str(Path(__file__).parent))
        from train_heatmap_model import HeatmapGenerator

        ckpt = torch.load(heatmap_ckpt, map_location="cpu")
        in_channels = ckpt["in_channels"]

        heatmap_model = HeatmapGenerator(in_channels=in_channels)
        heatmap_model.load_state_dict(ckpt["model_state_dict"])
        heatmap_model.eval()

        # Use same spatial dimensions as the backbone output
        with torch.no_grad():
            dummy_spatial = torch.randn(1, in_channels, 10, 10)
            test_out = heatmap_model(dummy_spatial)
        print(f"  input:  (1, {in_channels}, H, W)")
        print(f"  output: {tuple(test_out.shape)}")

        heatmap_onnx = OUT_DIR / "heatmap_generator.onnx"
        torch.onnx.export(
            heatmap_model, dummy_spatial, str(heatmap_onnx),
            input_names=["spatial_feat"],
            output_names=["heatmap"],
            dynamic_axes={"spatial_feat": {0: "batch", 2: "height", 3: "width"},
                          "heatmap":      {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        print(f"  saved ({heatmap_onnx.stat().st_size / 1e6:.1f} MB)  ✓")
    else:
        print("\n[5/7] Skipping heatmap generator (no checkpoint found)")
        print(f"  To generate: python generate_heatmap_data.py && python train_heatmap_model.py")

    # ── 6. Label map + treatments ─────────────────────────────────────────────
    print("\n[6/7] Generating label_map.json and treatments.json …")
    shutil.copy(label_map_path, OUT_DIR / "label_map.json")
    print("  label_map.json  ✓")

    with open(label_map_path, encoding="utf-8") as f:
        label_map = json.load(f)
    treatments = build_treatments(DATA_DIR / "text" / "plantwild.json", label_map)
    with open(OUT_DIR / "treatments.json", "w", encoding="utf-8") as f:
        json.dump(treatments, f, indent=2)
    print("  treatments.json ✓")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    print("\n✅  Export complete.  Output files:")
    total = 0
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            total += sz
            print(f"     {str(p.relative_to(OUT_DIR)):<40}  {sz / 1e6:>6.1f} MB")
    print(f"     {'TOTAL':<40}  {total / 1e6:>6.1f} MB")

    print("\nNext steps:")
    print("  1. cd ..  (back to project root)")
    print("  2. npm install")
    print("  3. npx expo run:ios   OR   npx expo run:android")


if __name__ == "__main__":
    main()
