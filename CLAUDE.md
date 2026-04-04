# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Plant Health Detector — a React Native (Expo) mobile app that lets users photograph plants and diagnose diseases. The ML backend is a multimodal pipeline (vision + text) trained in Python and exported to ONNX; the app runs fully offline inference via `onnxruntime-react-native`.

<<<<<<< HEAD
## Development Commands

```bash
# Install JS dependencies
npm install

# Start Expo dev server (scan QR with Expo Go app on same WiFi)
=======
## Project Structure

```
Plants-have-cancer/
├── frontend/           # React Native (Expo) mobile app
│   ├── app/            # Screens (Expo Router file-based routing)
│   ├── assets/         # Images, ONNX models, label/treatment data
│   ├── components/     # Reusable UI components
│   ├── store/          # Zustand state management
│   ├── utils/          # Inference, tokeniser, storage helpers
│   ├── app.json        # Expo config
│   ├── package.json    # JS dependencies
│   └── ...
├── backend/            # Python ML training & export
│   ├── checkpoints/    # Trained model weights
│   ├── data/           # Training images & text
│   ├── notebooks/      # Jupyter notebooks
│   └── *.py            # Training, export & heatmap scripts
├── .gitignore
└── README.md
```

## Development Commands

```bash
# ── Frontend ──────────────────────────────────────────────
cd frontend

# Install JS dependencies
npm install

# Start Expo dev server
>>>>>>> q0q
npm start

# Platform-specific launchers
npm run android
npm run ios
npm run web
```

<<<<<<< HEAD
> **Note:** Real ML inference requires a native build (`npx expo run:android` / `npx expo run:ios`) because `onnxruntime-react-native` does not work with Expo Go.
=======
> **Note:** Real ML inference requires a native build (`npx expo run:android` / `npx expo run:ios`) because `onnxruntime-react-native` does not work with Expo Go. For cloud builds, use `eas build --profile development --platform ios` (requires Expo account + Apple Developer account for iOS).
>>>>>>> q0q

### Python Backend (ML training)

```bash
cd backend

# Train image encoder (MobileViTv2)
python mobilevit.py

# Train text encoder (agriculture-BERT)
python bert.py

# Train fusion MLP (requires embeddings saved by the encoders above)
python mlp.py

<<<<<<< HEAD
# Export all three trained models to ONNX + generate asset files
python export_for_mobile.py

# Generate GradCAM heatmaps
python generate_heatmap.py
```

The backend requires CUDA and the packages in `backend/requirements.txt`. Use the project `venv/` or a fresh virtualenv. Training notebooks are in `backend/notebooks/`.
=======
# Generate training data for heatmap model (runs HiResCAM on training images)
python generate_heatmap_data.py

# Train heatmap generator CNN (learns to predict gradient-based heatmaps)
python train_heatmap_model.py

# Export all four models to ONNX + generate asset files
python export_for_mobile.py

# Generate standalone GradCAM visualisations (for research, not used by app)
python generate_heatmap.py
```

The backend requires CUDA and the packages in `backend/requirements.txt`. Use the project `.venv-1/` or a fresh virtualenv. Training notebooks are in `backend/notebooks/`.
>>>>>>> q0q

## Architecture

### Mobile App (React Native / Expo Router)

**Navigation flow** (file-based routing via Expo Router):

```
splash → (tabs)/index (Home)
             ↓
         (tabs)/scan → camera → preview → processing → results
             ↓
         (tabs)/history
         (tabs)/profile
```

<<<<<<< HEAD
Key files:
=======
Key files (all under `frontend/`):
>>>>>>> q0q
- `app/_layout.jsx` — root Stack navigator; bootstraps model load + scan history on mount
- `app/(tabs)/_layout.jsx` — bottom tab bar
- `store/usePlantStore.js` — Zustand store; single source of truth for `capturedImageUri`, `capturedText`, `analysisResult`, `recentScans`, `isModelLoaded`, and the `plants` list
- `utils/modelInference.js` — ONNX inference pipeline; exports `loadModel()` and `analyzeLeaf(imageUri, symptomText)`
- `utils/bertTokenizer.js` — custom WordPiece tokeniser for agriculture-BERT; reads `assets/models/tokenizer/vocab.json`
- `utils/storage.js` — AsyncStorage helpers for persisting scan history

**State flow for a scan session:**
1. Camera screen calls `setCapturedImage(uri)`
2. Preview screen collects symptom text; calls `setCapturedText(text)` then navigates to `/processing`
3. Processing screen calls `analyzeLeaf(imageUri, symptomText)` then `setAnalysisResult(result)`
4. Results screen reads `analysisResult`; on "New Scan" calls `resetSession()`
5. `addScan()` persists to AsyncStorage and refreshes `recentScans`

**Styling:** NativeWind (Tailwind CSS for React Native). Classes are transformed at build time via the Babel plugin in `babel.config.js`. Config is in `tailwind.config.js`.

### ML Backend (Python / PyTorch)

Three-stage multimodal pipeline, all targeting **89 plant disease classes** from the PlantWild dataset:

1. **Image encoder** (`backend/mobilevit.py`) — fine-tunes `mobilevitv2_150` (via `timm`) on `./data/images/plantwild/`. Saves backbone weights to `./checkpoints/best_image_encoder.pt` and pre-computed embeddings to `./checkpoints/image_embeddings.pt`.

2. **Text encoder** (`backend/bert.py`) — fine-tunes `recobo/agriculture-bert-uncased` on `./data/text/plantwild.json`. Saves to `./checkpoints/best_text_encoder.pt` (HuggingFace directory) and `./checkpoints/text_embeddings.pt`.

3. **Fusion MLP** (`backend/mlp.py`) — loads frozen image + text embeddings, concatenates them (768 + 768 = 1536-dim), and trains a 4-layer MLP classifier (`1536 → 1024 → 512 → 256 → 89`). Saved to `./checkpoints/best_multimodal_mlp.pt`.

`backend/dataset.py` defines `PlantWildDataset` (image) and `PlantWildTextDataset` (text), both with reproducible 80/20 train/test splits.

Data JSON files (`backend/data/text/plantwild.json`) contain per-class disease descriptions used for text embedding.

### ONNX Export (`backend/export_for_mobile.py`)

<<<<<<< HEAD
Converts the three trained PyTorch models to ONNX and generates all asset files consumed by the app:

| Output file | Description |
|---|---|
| `assets/models/image_backbone.onnx` | MobileViTv2_150 wrapped to return `img_emb (1,768)` + `spatial_feat (1,C,H,W)` |
| `assets/models/text_encoder.onnx` | agriculture-BERT [CLS] extractor, INT8-quantised |
| `assets/models/mlp.onnx` | Fusion MLP; inputs `img_emb`, `text_emb`; output `logits (1,89)` |
| `assets/models/label_map.json` | `{ "class name": int }` — 89 classes |
| `assets/models/treatments.json` | `{ "class name": description }` — one treatment string per class |
| `assets/models/tokenizer/vocab.json` | WordPiece vocabulary extracted from the fine-tuned BERT tokenizer |

### On-Device Inference (`utils/modelInference.js`)
=======
Converts the trained PyTorch models to ONNX and generates all asset files consumed by the app. All output goes to `frontend/assets/models/`:

| Output file | Description |
|---|---|
| `image_backbone.onnx` | MobileViTv2_150 wrapped to return `img_emb (1,768)` + `spatial_feat (1,C,H,W)` |
| `text_encoder.onnx` | agriculture-BERT [CLS] extractor, INT8-quantised |
| `mlp.onnx` | Fusion MLP; inputs `img_emb`, `text_emb`; output `logits (1,89)` |
| `heatmap_generator.onnx` | Small CNN; inputs `spatial_feat`; output `heatmap (1,1,320,320)` — trained to mimic HiResCAM |
| `label_map.json` | `{ "class name": int }` — 89 classes |
| `treatments.json` | `{ "class name": description }` — one treatment string per class |
| `tokenizer/vocab.json` | WordPiece vocabulary extracted from the fine-tuned BERT tokenizer |

### Heatmap Pipeline

Two approaches exist:

1. **Backend (research only):** `generate_heatmap.py` uses multi-layer HiResCAM with PyTorch gradients + contour mapping. Produces high-quality visualisations but requires PyTorch backward pass — cannot run on device.

2. **On-device (production):** A small CNN (`heatmap_generator.onnx`) trained to mimic the gradient-based heatmaps using only forward-pass spatial features. Training pipeline:
   - `generate_heatmap_data.py` — runs HiResCAM on training images to create (spatial_feat, heatmap) pairs
   - `train_heatmap_model.py` — trains a 4-layer CNN on those pairs (MSE loss)
   - `export_for_mobile.py` — exports the trained CNN to ONNX
   
   Falls back to gradient-free mean-CAM if the heatmap generator model fails.

### On-Device Inference (`frontend/utils/modelInference.js`)
>>>>>>> q0q

Full offline pipeline — no network calls:

1. **Image preprocessing** — resize shorter side to 368 (aspect ratio preserved) → centre-crop 320×320 → float32 NCHW tensor with ImageNet normalisation (`MEAN=[0.485,0.456,0.406]`, `STD=[0.229,0.224,0.225]`)
2. **Text preprocessing** — WordPiece tokenise via `bertTokenizer.js` → `input_ids` / `attention_mask` as `BigInt64Array[128]`
3. **Image session** → `img_emb (1,768)` + `spatial_feat`
4. **Text session** → `text_emb (1,768)`
5. **MLP session** → `logits (1,89)` → softmax → argmax → class label + treatment lookup
<<<<<<< HEAD
6. **Heatmap** — gradient-free mean-CAM from `spatial_feat`, bilinear upsample, jet colourmap, JPEG-encoded data URI
=======
6. **Heatmap session** → `spatial_feat` → `heatmap_generator.onnx` → jet-coloured JPEG data URI (fallback: mean-CAM)
>>>>>>> q0q

## Critical Invariants

### Image preprocessing must match training exactly

The image backbone was trained and embeddings were extracted with:
```
Resize(shorter side → 368, aspect ratio preserved) → CenterCrop(320×320)
```
**Do not change** `preprocessImage()` in `utils/modelInference.js` to a direct `Resize(320×320)` — that squishes non-square images and shifts the embedding distribution away from training, causing severe accuracy degradation.

### Text encoder input

The text encoder expects standard BERT uncased tokenisation (WordPiece, lowercase, max length 128). When the user provides no symptom text, `tokenize("")` produces `[CLS][SEP][PAD…]` — a valid null-text BERT input. Do not substitute a generic fallback string (e.g. `"plant disease symptoms"`), as it produces a noisy, class-unrelated embedding that misleads the MLP.

### Tensor dtypes

- Image tensor: `float32`, shape `(1, 3, 320, 320)`
- `input_ids` / `attention_mask`: `int64` via `BigInt64Array`, shape `(1, 128)`
- `img_emb` / `text_emb`: `float32`, shape `(1, 768)`
- `logits`: `float32`, shape `(1, 89)`
