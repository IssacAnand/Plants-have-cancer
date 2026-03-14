"""
bert_encoder.py
---------------

  FrozenBERTEncoder       — runs BERT once during pre-computation
  precompute_embeddings() — encodes all JSON captions, saves .pt cache files
  CachedTextEmbeddings    — loads cache, serves embeddings during training
  
Usage
-----
    # Build cache (run once)
    python bert_encoder.py --dataset plantdoc --data_dir /path/to/jsons
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

BERT_MODEL = "bert-base-uncased"
BERT_DIM   = 768
MAX_LEN    = 128
BATCH_SIZE = 64

TEXT_DIR  = Path("data/text")           # plantdoc_diverse.json, etc.
CACHE_DIR = Path("data/embeddings_cache")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")


# ── FrozenBERTEncoder ─────────────────────────────────────────────────────────

class FrozenBERTEncoder(nn.Module):
    """
    BERT with all weights permanently frozen.
    Used only during the one-time pre-computation step.
    Never instantiated during training.
    """

    def __init__(self, model_name: str = BERT_MODEL):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool over non-padding tokens → L2-normalise.
        input_ids      : (B, seq_len)
        attention_mask : (B, seq_len)
        returns        : (B, 768)
        """
        out      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask_exp = attention_mask.unsqueeze(-1).float()              # (B, seq, 1)
        pooled   = (out.last_hidden_state * mask_exp).sum(1) \
                   / mask_exp.sum(1).clamp(min=1e-9)                 # (B, 768)
        return F.normalize(pooled, p=2, dim=-1)


# ── precompute_embeddings ─────────────────────────────────────────────────────

def precompute_embeddings(
    dataset_name: str,
    data_dir: str   = str(TEXT_DIR),
    cache_dir: Path = CACHE_DIR,
) -> tuple[dict, dict]:
    """
    Encode every caption in the JSON dataset with frozen BERT.
    Saves two files to cache_dir:

        <dataset>_embeddings.pt   { label: tensor(n_captions, 768) }
        <dataset>_prototypes.pt   { label: tensor(768,) }

    Prefers _diverse.json → _clip.json → raw .json (first found wins).
    """
    data_path = None
    for suffix in ["_diverse", "_clip", ""]:
        p = Path(data_dir) / f"{dataset_name}{suffix}.json"
        if p.exists():
            data_path = p
            break
    if data_path is None:
        raise FileNotFoundError(f"No JSON found for '{dataset_name}' in {data_dir}")

    with open(data_path) as f:
        data: dict[str, list[str]] = json.load(f)

    print(f"Source   : {data_path.name}")
    print(f"Classes  : {len(data)}  |  Captions: {sum(len(v) for v in data.values())}")

    print(f"\nLoading {BERT_MODEL} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    encoder   = FrozenBERTEncoder(BERT_MODEL).to(DEVICE)

    embeddings: dict[str, torch.Tensor] = {}
    prototypes: dict[str, torch.Tensor] = {}

    print("Encoding captions ...")
    for label, captions in data.items():
        batches = []
        for i in range(0, len(captions), BATCH_SIZE):
            enc = tokenizer(
                captions[i : i + BATCH_SIZE],
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(DEVICE)
            batches.append(encoder.encode(enc["input_ids"], enc["attention_mask"]).cpu())

        stacked           = torch.cat(batches, dim=0)            # (n_captions, 768)
        embeddings[label] = stacked
        prototypes[label] = F.normalize(stacked.mean(0), dim=-1) # (768,)

    cache_dir.mkdir(exist_ok=True)
    torch.save(embeddings, cache_dir / f"{dataset_name}_embeddings.pt")
    torch.save(prototypes, cache_dir / f"{dataset_name}_prototypes.pt")
    print(f"Saved → {cache_dir / f'{dataset_name}_embeddings.pt'}")
    print(f"Saved → {cache_dir / f'{dataset_name}_prototypes.pt'}")

    return embeddings, prototypes


# ── CachedTextEmbeddings ──────────────────────────────────────────────────────

class CachedTextEmbeddings:
    """
    Loads pre-computed embeddings from disk.
    Instantiate once; keep on CPU; query per batch.

    get_proto_batch(label_ids)  →  (B, 768)   standard training lookup
    get_random(label)           →  (768,)     text augmentation variant
    proto_matrix                →  (C, 768)   all prototypes stacked
    """

    def __init__(self, dataset_name: str, cache_dir: Path = CACHE_DIR):
        proto_path = cache_dir / f"{dataset_name}_prototypes.pt"
        emb_path   = cache_dir / f"{dataset_name}_embeddings.pt"

        if not proto_path.exists():
            raise FileNotFoundError(
                f"Cache not found: {proto_path}\n"
                f"Run:  python bert_encoder.py --dataset {dataset_name}"
            )

        self.prototypes: dict[str, torch.Tensor] = torch.load(proto_path, weights_only=True)
        self.embeddings: dict[str, torch.Tensor] = torch.load(emb_path,   weights_only=True)

        self.label_names: list[str]   = sorted(self.prototypes.keys())
        self.label2id: dict[str, int] = {l: i for i, l in enumerate(self.label_names)}

        # Pre-stack for O(1) batch lookup
        self._proto_matrix = torch.stack(
            [self.prototypes[l] for l in self.label_names], dim=0
        )  # (num_classes, 768)

    def get_proto_batch(self, label_ids: torch.Tensor) -> torch.Tensor:
        """
        label_ids : (B,) int tensor of class indices
        returns   : (B, 768)
        """
        return self._proto_matrix[label_ids.cpu()]

    def get_random(self, label: str) -> torch.Tensor:
        """
        Randomly sample one caption embedding for a class.
        Use for text-side augmentation during training.
        returns : (768,)
        """
        pool = self.embeddings[label]
        return pool[torch.randint(len(pool), (1,)).item()]

    def num_classes(self) -> int:
        return len(self.label_names)

    @property
    def proto_matrix(self) -> torch.Tensor:
        """(num_classes, 768) — all prototypes stacked."""
        return self._proto_matrix


# ── Live text encoder for inference ──────────────────────────────────────────

class LiveTextEncoder:
    """
    Encodes arbitrary user-supplied text at inference time.
    Keeps the tokenizer and frozen BERT in memory.

    Usage
    -----
        encoder = LiveTextEncoder()
        embedding = encoder.encode("Leaves have brown spots and yellowing edges")
        # embedding : (768,) tensor, L2-normalised, ready to pass to the MLP
    """

    def __init__(self, model_name: str = BERT_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = FrozenBERTEncoder(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a single string → (768,) L2-normalised tensor on CPU.
        """
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        embedding = self.model.encode(enc["input_ids"], enc["attention_mask"])
        return embedding.squeeze(0).cpu()   # (768,)


# ── CLI (pre-computation entry point) ────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute BERT caption embeddings")
    parser.add_argument("--dataset",   required=True, help="e.g. plantdoc")
    parser.add_argument("--data_dir",  default=str(TEXT_DIR),   help="Folder containing JSON files")
    parser.add_argument("--cache_dir", default=str(CACHE_DIR))
    args = parser.parse_args()

    precompute_embeddings(args.dataset, args.data_dir, Path(args.cache_dir))