# Plant Health Detector

A React Native (Expo) mobile app that lets users photograph plants and diagnose diseases using a fully offline multimodal ML pipeline (vision + text). The ML backend is trained in Python and exported to ONNX; the app runs inference on-device via `onnxruntime-react-native`.

---

## Getting Started (Build the mobile application)

```bash
git clone https://github.com/IssacAnand/Plants-have-cancer.git
cd Plants-have-cancer/frontend
npm install
npx expo run:ios
```

## Running application

1. Copy this url: https://github.com/IssacAnand/Plants-have-cancer.git
2. Create your project directory in Vscode
3. Within VsCode, go settings -> command palette
4. Enter git clone
5. Paste https://github.com/IssacAnand/Plants-have-cancer.git
6. After downloading, open terminal and run "npm install"
7. Run IOS Build:

> **Note:** Real ML inference requires a native build (`npx expo run:android` / `npx expo run:ios`) because `onnxruntime-react-native` does not work with Expo Go.

---

## Project Structure

```
Plants-have-cancer/
├── frontend/       # React Native (Expo) mobile app
└── backend/        # Python ML training & ONNX export
```

---

## Frontend

**Path:** `frontend/`

### Screens (`app/`)

| File             | Description                                                           |
| ---------------- | --------------------------------------------------------------------- |
| `_layout.jsx`    | Root Stack navigator; bootstraps model load and scan history on mount |
| `camera.jsx`     | Camera capture screen                                                 |
| `preview.jsx`    | Image preview after capture; collects symptom text                    |
| `processing.jsx` | Runs ML inference; shows loading state                                |
| `results.jsx`    | Displays disease label, confidence, treatment, and heatmap            |
| `splash.jsx`     | Splash/loading screen shown on first launch                           |

### Tabs (`app/(tabs)/`)

| File          | Description                         |
| ------------- | ----------------------------------- |
| `_layout.jsx` | Bottom tab bar configuration        |
| `index.jsx`   | Home screen                         |
| `scan.jsx`    | Entry point for starting a new scan |
| `history.jsx` | Recent scan history                 |
| `profile.jsx` | Profile screen                      |

### Components (`components/`)

| File                | Description                           |
| ------------------- | ------------------------------------- |
| `PrimaryButton.jsx` | Reusable styled button                |
| `ScanCard.jsx`      | Card component for history list items |
| `SplashScreen.jsx`  | Animated splash screen component      |

### State (`store/`)

| File               | Description                                                                                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `usePlantStore.js` | Zustand store; single source of truth for `capturedImageUri`, `capturedText`, `analysisResult`, `recentScans`, `isModelLoaded`, and the `plants` list |

### Utilities (`utils/`)

| File                | Description                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------- |
| `modelInference.js` | Full ONNX inference pipeline; exports `loadModel()` and `analyzeLeaf(imageUri, symptomText)` |
| `bertTokenizer.js`  | Custom WordPiece tokeniser for agriculture-BERT; reads `assets/models/tokenizer/vocab.json`  |
| `storage.js`        | AsyncStorage helpers for persisting scan history                                             |

### Model Assets (`assets/models/`)

| File                    | Description                                                       |
| ----------------------- | ----------------------------------------------------------------- |
| `image_backbone.onnx`   | MobileViTv2_150; outputs `img_emb (1,768)` + `spatial_feat`       |
| `text_encoder.onnx`     | agriculture-BERT [CLS] extractor, INT8-quantised                  |
| `mlp.onnx`              | Fusion MLP; inputs `img_emb` + `text_emb`; output `logits (1,89)` |
| `label_map.json`        | `{ "class name": int }` — 89 disease classes                      |
| `treatments.json`       | `{ "class name": description }` — treatment text per class        |
| `disease_analysis.json` | Additional per-class disease metadata                             |
| `tokenizer/vocab.json`  | WordPiece vocabulary for the BERT tokeniser                       |

### Config Files

| File                                     | Description                                      |
| ---------------------------------------- | ------------------------------------------------ |
| `app.json`                               | Expo app configuration                           |
| `babel.config.js`                        | Babel config with NativeWind plugin              |
| `metro.config.js`                        | Metro bundler config                             |
| `tailwind.config.js`                     | Tailwind/NativeWind CSS config                   |
| `eas.json`                               | EAS Build profiles                               |
| `global.css`                             | Global NativeWind CSS entry                      |
| `jsconfig.json`                          | JS path aliases                                  |
| `react-native.config.js`                 | React Native CLI config                          |
| `patches/expo-modules-core+3.0.29.patch` | Patch for expo-modules-core compatibility        |
| `scripts/fix-onnxruntime-cmake.js`       | Post-install script for onnxruntime native build |

---

## Backend

**Path:** `backend/`

### Training Scripts

| File                         | Description                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `mobilevit.py`               | Fine-tunes MobileViTv2_150 image encoder on PlantWild dataset                |
| `bert.py`                    | Fine-tunes agriculture-BERT text encoder on disease descriptions             |
| `mlp.py`                     | Trains the fusion MLP on concatenated image + text embeddings                |
| `dataset.py`                 | `PlantWildDataset` and `PlantWildTextDataset` with 80/20 train/test splits   |
| `generate_heatmap_data.py`   | Runs HiResCAM on training images to produce `(spatial_feat, heatmap)` pairs  |
| `train_heatmap_model.py`     | Trains a small CNN to mimic gradient-based heatmaps (MSE loss)               |
| `mlp_ablation_modalities.py` | Ablation study script for image-only vs. text-only vs. multimodal            |
| `export_for_mobile.py`       | Exports all trained models to ONNX and generates all asset files for the app |

### Notebooks (`notebooks/`)

| File                          | Description                            |
| ----------------------------- | -------------------------------------- |
| `BERT.ipynb`                  | BERT text encoder experiments          |
| `mlp.ipynb`                   | MLP fusion model experiments           |
| `mlp_hp_tuning.ipynb`         | MLP hyperparameter tuning              |
| `ablation_modalities-2.ipynb` | Ablation study across input modalities |
| `plantwild_pipeline.ipynb`    | End-to-end PlantWild pipeline          |
| `vit_test.ipynb`              | ViT model experiments                  |
| `vit_test_colab.ipynb`        | ViT experiments (Colab version)        |

### Training Data (`data/`)

| Path                          | Description                                     |
| ----------------------------- | ----------------------------------------------- |
| `data/images/plantwild/`      | PlantWild training images (primary dataset)     |
| `data/images/plantvillage/`   | PlantVillage training images                    |
| `data/images/plantdoc/`       | PlantDoc training images                        |
| `data/text/plantwild.json`    | Per-class disease descriptions for PlantWild    |
| `data/text/plantvillage.json` | Per-class disease descriptions for PlantVillage |
| `data/text/plantdoc.json`     | Per-class disease descriptions for PlantDoc     |
| `data/label_map.json`         | Class name to index mapping                     |

### Saved Checkpoints (`checkpoints/`)

| Path                     | Description                                           |
| ------------------------ | ----------------------------------------------------- |
| `best_image_encoder.pt`  | Best MobileViTv2 weights                              |
| `best_text_encoder.pt/`  | Best agriculture-BERT weights (HuggingFace directory) |
| `best_multimodal_mlp.pt` | Best fusion MLP weights                               |
| `image_embeddings.pt`    | Pre-computed image embeddings (used to train MLP)     |
| `text_embeddings.pt`     | Pre-computed text embeddings (used to train MLP)      |

### Running the Backend

```bash
cd backend

# 0. Initialize environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Train image encoder
python mobilevit.py

# 2. Train text encoder
python bert.py

# 3. Train fusion MLP (requires embeddings from steps 1 & 2)
python mlp.py

# 4. Generate heatmap training data
python generate_heatmap_data.py

# 5. Train heatmap generator CNN
python train_heatmap_model.py

# 6. Export all models to ONNX + generate app asset files
python export_for_mobile.py
```

The backend requires CUDA and the packages listed in `requirements.txt`.

---

## ML Architecture

Three-stage multimodal pipeline targeting **89 plant disease classes**:

1. **Image Encoder** — MobileViTv2_150 fine-tuned on PlantWild; outputs a 768-dim embedding + spatial feature map
2. **Text Encoder** — agriculture-BERT fine-tuned on disease descriptions; outputs a 768-dim [CLS] embedding
3. **Fusion MLP** — concatenates both embeddings (1536-dim) and classifies into 89 disease classes (`1536 → 1024 → 512 → 256 → 89`)
4. **Heatmap Generator** — small CNN trained to mimic HiResCAM gradient heatmaps using only forward-pass spatial features; enables on-device saliency maps with no backward pass
