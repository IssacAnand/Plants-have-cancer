// utils/modelInference.js
//
// ─────────────────────────────────────────────────────────────────────────────
// TODO: replace stub with real PyTorch inference
//
// This file is currently STUBBED because react-native-pytorch-core is not yet
// compatible with React Native 0.81 and requires a native development build.
//
// When ready to use real inference:
//   1. Confirm react-native-pytorch-core supports RN 0.81:
//      https://www.npmjs.com/package/react-native-pytorch-core
//   2. Run: npm install react-native-pytorch-core
//   3. Replace this stub with the real implementation (see comments below)
//   4. Place your exported .ptl model at: assets/models/plant_model.ptl
//   5. Build with: npx expo run:android  or  npx expo run:ios
//      (Expo Go will NOT work with native modules)
// ─────────────────────────────────────────────────────────────────────────────
//
// ORIGINAL CONCEPT: On-Device AI with PyTorch Mobile
// ─────────────────────────────────────────────────────────────────────────────
// The real three-step inference pipeline:
//  [Camera Image URI]
//       ↓
//  1. PREPROCESS  — resize, crop, normalize the image into a tensor
//       ↓
//  2. INFERENCE   — feed tensor into model.forward() → get output tensor
//       ↓
//  3. POSTPROCESS — softmax → find highest probability → map to label
//       ↓
//  { disease, confidence, treatment }
// ─────────────────────────────────────────────────────────────────────────────

// ─── STUB IMPLEMENTATION ─────────────────────────────────────────────────────
//
// loadModel() immediately resolves true so the UI shows "Model Ready".
// analyzeLeaf() waits 2 seconds then returns a hardcoded mock result
// so the full navigation flow (scan → preview → processing → results) works.

/**
 * Simulates loading the PyTorch model.
 * Returns true immediately so the UI unlocks.
 */
export async function loadModel() {
  // Stub: no actual model loading
  console.log("[model] Running in STUB mode — no PyTorch model loaded");
  return true;
}

/**
 * Simulates plant disease inference on a captured image.
 * Returns a hardcoded mock result after a short delay.
 *
 * @param {string} imageUri  Local file URI from expo-camera
 * @returns {{ disease: string, confidence: number, treatment: string }}
 */
export async function analyzeLeaf(imageUri) {
  // Simulate inference time
  await new Promise((resolve) => setTimeout(resolve, 2000));

  return {
    plantName:  "Monsterra",
    disease:    "Light Blight with Sun Spots",
    confidence: 97,
    treatment:
      "Remove infected leaves immediately. Apply a copper-based fungicide every 7–10 days. Avoid overhead watering and ensure good air circulation.",
  };
}
