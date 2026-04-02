# Repository Guidelines

## Project Structure & Module Organization
`app/` contains Expo Router screens and route layouts, including `app/(tabs)/` for the tab navigator. Reusable UI lives in `components/`, shared state in `store/usePlantStore.js`, and client-side inference/storage helpers in `utils/`. Static images and shipped model assets live under `assets/`, especially `assets/models/`. The Python training and export pipeline lives in `backend/`, with scripts such as `mobilevit.py`, `bert.py`, `mlp.py`, and `export_for_mobile.py`; notebooks and datasets are kept under `backend/notebooks/` and `backend/data/`.

## Build, Test, and Development Commands
Use `npm install` once at the repo root to install the Expo app dependencies. Run `npm start` to launch Expo, `npm run android` for a native Android build, and `npm run web` for browser testing. For backend work, create or activate a Python environment, then run `pip install -r backend/requirements.txt`. Rebuild mobile model assets from `backend/` with `python export_for_mobile.py`; this refreshes the ONNX files and tokenizer data in `assets/models/`.

## Coding Style & Naming Conventions
Follow the existing style: JavaScript/JSX uses 2-space indentation, double quotes, and semicolons; Python uses 4-space indentation and `snake_case`. Keep Expo Router screen filenames route-oriented and lowercase, such as `app/results.jsx`, while reusable React components use `PascalCase` like `components/PrimaryButton.jsx`. Prefer concise comments only where the control flow or model logic is non-obvious. No ESLint or Prettier config is committed, so match surrounding code carefully.

## Testing Guidelines
There is no automated test suite configured yet. Validate app changes by running `npm start` and exercising the affected flow in Expo or an emulator. Validate backend/model changes by running the relevant Python script directly and confirming regenerated files under `assets/models/`. If you add tests, keep them alongside the feature or under a dedicated `tests/` folder and use descriptive names such as `model_inference.test.js`.

## Commit & Pull Request Guidelines
Recent history mixes short feature notes (`home complete and scan page almost`) with script-focused commits. Prefer short, imperative messages that name the area changed, for example `scan: tighten camera permission flow` or `backend: export quantized text encoder`. Pull requests should summarize user-visible changes, list verification steps, link the related issue or task, and include screenshots or recordings for UI updates. Call out any large asset, model, or dataset changes explicitly.
