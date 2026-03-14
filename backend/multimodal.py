"""
multimodal.py
-------------
Responsibility: fuse ViT image features with BERT text features
                and classify.

  FusionMLP                   — the only trainable module
  MultimodalDiseaseClassifier — wires ViTEncoder + CachedTextEmbeddings + FusionMLP

Import graph (no cycles):
    multimodal.py
        ├── bert_encoder.py  (CachedTextEmbeddings)
        └── vit.py           (ViTEncoder)

Usage
-----
    from bert_encoder import CachedTextEmbeddings
    from vit import ViTEncoder
    from multimodal import MultimodalDiseaseClassifier

    cache  = CachedTextEmbeddings("plantdoc")
    vit    = ViTEncoder(frozen_layers=10)
    model  = MultimodalDiseaseClassifier(vit, cache, num_classes=27)

    logits = model(images, label_ids)               # training
    logits = model(images, label_ids, augment_text=True)  # with text augment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import CachedTextEmbeddings, BERT_DIM, CACHE_DIR
from vit import ViTEncoder, VIT_DIM

# ── Constants ─────────────────────────────────────────────────────────────────

FUSED_DIM = VIT_DIM + BERT_DIM   # 768 + 768 = 1536

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")


# ── FusionMLP ─────────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    """
    The only module that trains during multimodal training.

    Input  : cat([vit_feats, text_feats])  →  (B, 1536)
    Output : class logits                  →  (B, num_classes)

    Both input vectors are L2-normalised (unit length) before concat,
    so neither modality numerically dominates the other.
    LayerNorm is applied first to re-centre the concatenated vector
    before the first linear layer.
    """

    def __init__(
        self,
        num_classes: int,
        fused_dim:  int   = FUSED_DIM,
        hidden_dim: int   = 512,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, vit_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        vit_feats  : (B, 768) — L2-normalised, from ViTEncoder
        text_feats : (B, 768) — L2-normalised, from CachedTextEmbeddings
        returns    : (B, num_classes)
        """
        fused = torch.cat([vit_feats, text_feats], dim=-1)  # (B, 1536)
        return self.net(fused)


# ── MultimodalDiseaseClassifier ───────────────────────────────────────────────

class MultimodalDiseaseClassifier(nn.Module):
    """
    Full multimodal model.

    ViT branch  : images   → ViTEncoder      → (B, 768)  ← may fine-tune
    Text branch : label_id → cache lookup    → (B, 768)  ← no compute, no grad
    Fusion      : concat   → FusionMLP       → (B, num_classes)

    Parameters
    ----------
    vit_encoder  : ViTEncoder     — your ViT-Base, possibly pre-loaded from checkpoint
    text_cache   : CachedTextEmbeddings — pre-computed BERT embeddings on CPU
    num_classes  : int
    hidden_dim   : int            — FusionMLP hidden layer width
    dropout      : float
    """

    def __init__(
        self,
        vit_encoder:  ViTEncoder,
        text_cache:   CachedTextEmbeddings,
        num_classes:  int,
        hidden_dim:   int   = 512,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.vit        = vit_encoder
        self.text_cache = text_cache   # not an nn.Module — won't appear in state_dict
        self.mlp        = FusionMLP(num_classes, FUSED_DIM, hidden_dim, dropout)

    def forward(
        self,
        images:       torch.Tensor,        # (B, 3, 224, 224)
        label_ids:    torch.Tensor,        # (B,) int — class index per sample
        augment_text: bool = False,        # True during training for text diversity
    ) -> torch.Tensor:
        """
        Returns logits (B, num_classes).

        label_ids is available during training because each training image
        has a known ground-truth label. At inference, pass the predicted
        label_id from a first-pass ViT classification, or iterate over all
        classes and take the argmax.
        """
        # ── Image branch ──────────────────────────────────────────────────
        vit_feats = self.vit(images)                                # (B, 768), L2-normed

        # ── Text branch (no gradient, no BERT, just a tensor index) ───────
        if augment_text:
            # Random caption per sample — adds diversity, slows training slightly
            text_feats = torch.stack([
                self.text_cache.get_random(self.text_cache.label_names[i.item()])
                for i in label_ids
            ]).to(images.device)                                    # (B, 768)
        else:
            # Mean prototype — deterministic, faster
            text_feats = self.text_cache.get_proto_batch(label_ids).to(images.device)  # (B, 768)

        # ── Fusion ─────────────────────────────────────────────────────────
        return self.mlp(vit_feats, text_feats)                      # (B, num_classes)

    def forward_with_text(
        self,
        images:         torch.Tensor,   # (B, 3, 224, 224)
        text_embedding: torch.Tensor,   # (768,) or (B, 768) — live user text
    ) -> torch.Tensor:
        """
        Inference path for user-supplied text.
        text_embedding comes from LiveTextEncoder.encode(), not the cache.

        returns : (B, num_classes) logits
        """
        vit_feats  = self.vit(images)                              # (B, 768)

        # Expand scalar embedding to match batch size if needed
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0).expand(vit_feats.shape[0], -1)

        text_feats = F.normalize(text_embedding.to(images.device), p=2, dim=-1)  # (B, 768)
        return self.mlp(vit_feats, text_feats)                     # (B, num_classes)

    def trainable_params(self) -> dict:
        vit_p = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        mlp_p = sum(p.numel() for p in self.mlp.parameters())
        return {
            "vit_trainable": vit_p,
            "mlp_trainable": mlp_p,
            "total":         vit_p + mlp_p,
        }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal model self-test")
    parser.add_argument("--dataset",       default="plantdoc")
    parser.add_argument("--cache_dir",     default=str(CACHE_DIR))
    parser.add_argument("--frozen_layers", type=int, default=10,
                        help="Number of ViT blocks to freeze (0–12)")
    args = parser.parse_args()

    from pathlib import Path

    print(f"Device : {DEVICE}\n")

    # Load cache
    cache = CachedTextEmbeddings(args.dataset, Path(args.cache_dir))
    num_classes = cache.num_classes()

    # Build model
    vit   = ViTEncoder(frozen_layers=args.frozen_layers)
    model = MultimodalDiseaseClassifier(vit, cache, num_classes).to(DEVICE)

    params = model.trainable_params()
    print(f"ViT trainable params : {params['vit_trainable']:,}")
    print(f"MLP trainable params : {params['mlp_trainable']:,}")
    print(f"Total trainable      : {params['total']:,}")
    print(f"BERT params          : ~110M  (pre-computed, not in graph)\n")

    # Forward pass with mock images
    B          = 4
    images     = torch.randn(B, 3, 224, 224).to(DEVICE)
    label_ids  = torch.randint(0, num_classes, (B,))

    model.eval()
    with torch.no_grad():
        logits = model(images, label_ids)

    print(f"images    : {images.shape}")
    print(f"label_ids : {label_ids.tolist()}")
    print(f"logits    : {logits.shape}")
    print("\nSelf-test passed.")