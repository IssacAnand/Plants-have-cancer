## Running application
1. Copy this url: https://github.com/IssacAnand/Plants-have-cancer.git
2. Create your project directory in Vscode
3. Within VsCode, go settings -> command palette
4. Enter git clone
5. Paste https://github.com/IssacAnand/Plants-have-cancer.git
6. After downloading, open terminal and run `cd frontend`
7. Run `npm install`
8. Run `npm start`
9. Install Expo Go on from App Store
10. Make sure both laptop and Phone on same wifi before proceeding
11. Scan QR code to launch development on Iphone

## Project Structure
```text
Plants-have-cancer/
├── frontend/
│   ├── app/
│   │   ├── (tabs)/
│   │   │   ├── _layout.jsx          # Bottom tab bar configuration
│   │   │   ├── index.jsx            # Home screen
│   │   │   ├── scan.jsx             # Camera screen
│   │   │   ├── history.jsx          # Recent scans
│   │   │   └── profile.jsx          # Profile screen
│   │   ├── _layout.jsx              # Root layout & model loader
│   │   ├── camera.jsx               # Camera capture screen
│   │   ├── index.jsx                # Entry screen
│   │   ├── preview.jsx              # Image preview after capture
│   │   ├── processing.jsx           # Loading/analyzing screen
│   │   ├── results.jsx              # Disease result screen
│   │   └── splash.jsx               # Splash screen
│   ├── components/
│   │   ├── PrimaryButton.jsx        # Reusable green button
│   │   ├── ScanCard.jsx             # Card for history items
│   │   └── SplashScreen.jsx         # Splash screen component
│   ├── store/
│   │   └── usePlantStore.js         # Zustand global state
│   ├── utils/
│   │   ├── bertTokenizer.js         # BERT tokenizer for text input
│   │   ├── storage.js               # AsyncStorage helpers
│   │   └── modelInference.js        # ONNX inference logic
│   ├── assets/
│   │   └── models/
│   │       ├── disease_analysis.json    # Disease metadata
│   │       ├── heatmap_generator.onnx   # Heatmap generator model
│   │       ├── image_backbone.onnx      # Image encoder model
│   │       ├── label_map.json           # Class label mapping
│   │       ├── mlp.onnx                 # Multimodal MLP model
│   │       ├── tokenizer/               # BERT tokenizer files
│   │       └── treatments.json          # Treatment information
│   ├── babel.config.js              # NativeWind & Expo config
│   ├── global.css                   # Global CSS (NativeWind)
│   └── tailwind.config.js           # Tailwind CSS configuration
├── backend/
│   ├── bert.py                      # BERT text encoder
│   ├── dataset.py                   # Dataset loading utilities
│   ├── mlp.py                       # Multimodal MLP model
│   ├── mobilevit.py                 # MobileViT image encoder
│   ├── generate_heatmap.py          # Heatmap generation
│   ├── train_heatmap_model.py       # Heatmap model training
│   ├── export_for_mobile.py         # Export models to ONNX
│   ├── requirements.txt             # Python dependencies
│   ├── data/
│   │   ├── label_map.json           # Class label mapping
│   │   ├── images/                  # Training images
│   │   └── text/                    # Training text data
│   ├── checkpoints/                 # Saved model weights
│   └── notebooks/                   # Jupyter notebooks
├── android/                         # Android native build files
└── scripts/                         # Build utility scripts
```