## Running application
1. Copy this url: https://github.com/IssacAnand/Plants-have-cancer.git
2. Create your project directory in Vscode
3. Within VsCode, go settings -> command palette
4. Enter git clone
5. Paste https://github.com/IssacAnand/Plants-have-cancer.git
6. After downloading, open terminal and run "npm install"
7. Run "npm start"
8. Install Expo Go on from App Store
9. Make sure both laptop and Phone on same wifi before proceeding
10. Scan QR code to launch development on Iphone

## Project Structure
```text
PlantHealthDetector/
├── app/
│   ├── (tabs)/
│   │   ├── _layout.jsx          # Bottom tab bar configuration
│   │   ├── index.jsx            # Home screen
│   │   ├── scan.jsx             # Camera screen
│   │   ├── history.jsx          # Recent scans
│   │   └── profile.jsx          # Profile screen
│   ├── _layout.jsx              # Root layout & model loader
│   ├── preview.jsx              # Image preview after capture
│   ├── processing.jsx           # Loading/analyzing screen
│   └── results.jsx              # Disease result screen
├── components/
│   ├── PrimaryButton.jsx        # Reusable green button
│   └── ScanCard.jsx             # Card for history items
├── store/
│   └── usePlantStore.js         # Zustand global state
├── utils/
│   ├── storage.js               # AsyncStorage helpers
│   └── modelInference.js        # PyTorch inference logic
├── assets/
│   └── models/
│       └── plant_model.ptl      # Machine Learning Model
├── babel.config.js              # NativeWind & Expo config
└── tailwind.config.js           # Tailwind CSS configuration
```