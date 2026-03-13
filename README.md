## Running application


## Project Structure
PlantHealthDetector/
├── app/
│   ├── _layout.jsx          ← Root layout & model loader
│   ├── preview.jsx          ← Image preview after capture
│   ├── processing.jsx       ← Loading/analyzing screen
│   ├── results.jsx          ← Disease result screen
│   └── (tabs)/
│       ├── _layout.jsx      ← Bottom tab bar
│       ├── index.jsx        ← Home screen
│       ├── scan.jsx         ← Camera screen
│       ├── history.jsx      ← Recent scans
│       └── profile.jsx      ← Profile screen
├── components/
│   ├── PrimaryButton.jsx    ← Reusable green button
│   └── ScanCard.jsx         ← Card for history items
├── store/
│   └── usePlantStore.js     ← Zustand global state
├── utils/
│   ├── storage.js           ← AsyncStorage helpers
│   └── modelInference.js    ← PyTorch inference logic
├── assets/
│   └── models/
│       └── plant_model.ptl  ← YOUR MODEL GOES HERE
├── babel.config.js
└── tailwind.config.js