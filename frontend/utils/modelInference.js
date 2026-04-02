// utils/modelInference.js
//
// Fully offline plant disease inference using three ONNX Runtime sessions.
//
// Pipeline
// ────────
//   loadModel()
//     • image_backbone.onnx  — MobileViTv2_150 (pooled emb + spatial features)
//     • text_encoder.onnx    — agriculture-BERT INT8 ([CLS] embedding)
//     • mlp.onnx             — MultimodalMLP fusion head (89-class logits)
//
//   analyzeLeaf(imageUri, symptomText)
//     1. Resize image to 320×320 (expo-image-manipulator)
//     2. Decode JPEG → RGBA pixels (jpeg-js)
//     3. Normalise to float32 NCHW tensor (ImageNet stats)
//     4. Tokenise symptomText with bertTokenizer (WordPiece, BigInt64Array)
//     5. Run imageSession  → { img_emb, spatial_feat }
//     6. Run textSession   → { text_emb }
//     7. Run mlpSession    → { logits }   (softmax → argmax → class)
//     8. Build gradient-free CAM from spatial_feat → JPEG base64
//     9. Return { plantName, disease, confidence, treatment, heatmapUri }
//
// Requirements
// ────────────
//   npm install onnxruntime-react-native expo-image-manipulator jpeg-js expo-asset
//   Run with:  npx expo run:ios  OR  npx expo run:android
//   (Expo Go does NOT support native modules)

import { InferenceSession, Tensor } from "onnxruntime-react-native";
import { Asset } from "expo-asset";
import * as ImageManipulator from "expo-image-manipulator";
import jpeg from "jpeg-js";

import { tokenize } from "./bertTokenizer";

// ── Constants ─────────────────────────────────────────────────────────────────

const IMG_SIZE = 320;

// ImageNet normalisation — must match mobilevit.py get_transforms()
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];

// ── JSON assets (bundled by Metro at build time) ───────────────────────────────

const LABEL_MAP  = require("../assets/models/label_map.json");   // { "class name": int }
const TREATMENTS = require("../assets/models/treatments.json");  // { "class name": string }

// Invert label map once at module load: { int: "class name" }
const INDEX_TO_CLASS = Object.fromEntries(
  Object.entries(LABEL_MAP).map(([name, idx]) => [idx, name])
);

// ── Module-level ONNX sessions (initialised by loadModel) ────────────────────

let imageSession   = null;
let textSession    = null;
let mlpSession     = null;
let heatmapSession = null;  // optional 4th model for gradient-quality heatmaps

const BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BASE64_LOOKUP = Object.fromEntries(
  BASE64_ALPHABET.split("").map((char, index) => [char, index])
);

// ── loadModel ─────────────────────────────────────────────────────────────────

/**
 * Loads all three ONNX models from the app bundle into memory.
 * Call once from app/_layout.jsx on startup.
 *
 * @returns {Promise<boolean>}  true on success, false on any failure
 */
export async function loadModel() {
  try {
    // Helper: resolve an asset and create an InferenceSession from its local URI
    async function loadSession(requireId) {
      const asset = Asset.fromModule(requireId);
      await asset.downloadAsync();
      return InferenceSession.create(asset.localUri ?? asset.uri);
    }

    [imageSession, textSession, mlpSession, heatmapSession] = await Promise.all([
      loadSession(require("../assets/models/image_backbone.onnx")),
      loadSession(require("../assets/models/text_encoder.onnx")),
      loadSession(require("../assets/models/mlp.onnx")),
      loadSession(require("../assets/models/heatmap_generator.onnx")),
    ]);

    console.log("[model] All four ONNX sessions loaded");
    return true;
  } catch (err) {
    console.error("[model] Failed to load:", err);
    return false;
  }
}

function decodeBase64(base64) {
  const cleaned = (base64 ?? "").replace(/[^A-Za-z0-9+/=]/g, "");

  if (cleaned.length % 4 !== 0) {
    throw new Error("Invalid base64 input");
  }

  let outputIndex = 0;
  const output = new Uint8Array((cleaned.length / 4) * 3);

  for (let i = 0; i < cleaned.length; i += 4) {
    const c1 = cleaned[i];
    const c2 = cleaned[i + 1];
    const c3 = cleaned[i + 2];
    const c4 = cleaned[i + 3];

    const n1 = BASE64_LOOKUP[c1];
    const n2 = BASE64_LOOKUP[c2];
    const n3 = c3 === "=" ? 0 : BASE64_LOOKUP[c3];
    const n4 = c4 === "=" ? 0 : BASE64_LOOKUP[c4];

    if (
      n1 === undefined ||
      n2 === undefined ||
      (c3 !== "=" && n3 === undefined) ||
      (c4 !== "=" && n4 === undefined)
    ) {
      throw new Error("Invalid base64 input");
    }

    const chunk = (n1 << 18) | (n2 << 12) | (n3 << 6) | n4;
    output[outputIndex++] = (chunk >> 16) & 0xff;

    if (c3 !== "=") {
      output[outputIndex++] = (chunk >> 8) & 0xff;
    }

    if (c4 !== "=") {
      output[outputIndex++] = chunk & 0xff;
    }
  }

  return output.subarray(0, outputIndex);
}

function encodeBase64(bytes) {
  let output = "";

  for (let i = 0; i < bytes.length; i += 3) {
    const b1 = bytes[i];
    const b2 = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const b3 = i + 2 < bytes.length ? bytes[i + 2] : 0;

    const chunk = (b1 << 16) | (b2 << 8) | b3;

    output += BASE64_ALPHABET[(chunk >> 18) & 63];
    output += BASE64_ALPHABET[(chunk >> 12) & 63];
    output += i + 1 < bytes.length ? BASE64_ALPHABET[(chunk >> 6) & 63] : "=";
    output += i + 2 < bytes.length ? BASE64_ALPHABET[chunk & 63] : "=";
  }

  return output;
}

// ── Image pre-processing ──────────────────────────────────────────────────────

/**
 * Decodes a base64 JPEG string produced by expo-image-manipulator.
 * Returns a Uint8Array of RGBA pixels in row-major order.
 */
function decodeJpegBase64(base64) {
  return jpeg.decode(decodeBase64(base64), { useTArray: true }).data; // RGBA Uint8Array
}

/**
 * Resize and crop the captured photo to IMG_SIZE×IMG_SIZE, decode to pixels,
 * and return an ONNX float32 tensor shaped (1, 3, IMG_SIZE, IMG_SIZE).
 *
 * Preprocessing matches the training pipeline in mobilevit.py:
 *   torchvision.transforms.Resize(368)      — shorter side → 368, aspect preserved
 *   torchvision.transforms.CenterCrop(320)  — centre 320×320 crop
 *
 * @param {string} imageUri  file:// URI from expo-camera or expo-image-picker
 * @returns {Promise<Tensor>}
 */
async function preprocessImage(imageUri) {
  const RESIZE_SIZE = Math.round(IMG_SIZE * 1.15); // 368

  // Probe original dimensions so we can resize the *shorter* side to RESIZE_SIZE
  // while preserving aspect ratio — exactly what torchvision.Resize(int) does.
  const { width: origW, height: origH } = await ImageManipulator.manipulateAsync(
    imageUri, [], {}
  );

  const resizeW = origW <= origH
    ? RESIZE_SIZE
    : Math.round(origW * (RESIZE_SIZE / origH));
  const resizeH = origH <= origW
    ? RESIZE_SIZE
    : Math.round(origH * (RESIZE_SIZE / origW));

  // Centre-crop to IMG_SIZE × IMG_SIZE
  const cropX = Math.floor((resizeW - IMG_SIZE) / 2);
  const cropY = Math.floor((resizeH - IMG_SIZE) / 2);

  const resized = await ImageManipulator.manipulateAsync(
    imageUri,
    [
      { resize: { width: resizeW, height: resizeH } },
      { crop: { originX: cropX, originY: cropY, width: IMG_SIZE, height: IMG_SIZE } },
    ],
    { compress: 0.95, format: ImageManipulator.SaveFormat.JPEG, base64: true }
  );

  const rgbaData = decodeJpegBase64(resized.base64);

  // RGBA HWC → float32 NCHW with ImageNet normalisation
  const float32 = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  for (let h = 0; h < IMG_SIZE; h++) {
    for (let w = 0; w < IMG_SIZE; w++) {
      const srcBase = (h * IMG_SIZE + w) * 4;
      for (let c = 0; c < 3; c++) {
        float32[c * IMG_SIZE * IMG_SIZE + h * IMG_SIZE + w] =
          (rgbaData[srcBase + c] / 255 - MEAN[c]) / STD[c];
      }
    }
  }

  return new Tensor("float32", float32, [1, 3, IMG_SIZE, IMG_SIZE]);
}

// ── Heatmap from trained generator model ──────────────────────────────────────

/**
 * Convert the heatmap generator model output (Float32Array of [0,1] values)
 * into a jet-coloured JPEG data URI.
 *
 * @param {Float32Array} data  Flat (1 × 1 × size × size) output from ONNX
 * @param {number}       size  Output dimension (320)
 * @returns {string}           "data:image/jpeg;base64,…"
 */
function buildHeatmapFromModel(data, size) {
  const rgba = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const val = Math.max(0, Math.min(1, data[i]));
    const [r, g, b] = jetColor(val);
    rgba[i * 4]     = r;
    rgba[i * 4 + 1] = g;
    rgba[i * 4 + 2] = b;
    rgba[i * 4 + 3] = 255;
  }

  const encoded = jpeg.encode({ data: rgba, width: size, height: size }, 85);
  const bytes = new Uint8Array(encoded.data);
  return `data:image/jpeg;base64,${encodeBase64(bytes)}`;
}

// ── Heatmap generation (gradient-free mean-CAM) — fallback ────────────────────

/**
 * Maps t ∈ [0, 1] to an RGB triple using the jet colormap.
 *   0 → dark blue   0.5 → green   1 → dark red
 */
function jetColor(t) {
  const clamp = (x) => Math.max(0, Math.min(1, x));
  return [
    Math.round(clamp(1.5 - Math.abs(4 * t - 3)) * 255),
    Math.round(clamp(1.5 - Math.abs(4 * t - 2)) * 255),
    Math.round(clamp(1.5 - Math.abs(4 * t - 1)) * 255),
  ];
}

/**
 * Bilinear resize of a Float32Array from (srcW × srcH) to (dstW × dstH).
 */
function bilinearResize(src, srcW, srcH, dstW, dstH) {
  const dst = new Float32Array(dstW * dstH);
  for (let y = 0; y < dstH; y++) {
    for (let x = 0; x < dstW; x++) {
      const sx  = (x + 0.5) * (srcW / dstW) - 0.5;
      const sy  = (y + 0.5) * (srcH / dstH) - 0.5;
      const x0  = Math.max(0, Math.floor(sx));
      const x1  = Math.min(srcW - 1, x0 + 1);
      const y0  = Math.max(0, Math.floor(sy));
      const y1  = Math.min(srcH - 1, y0 + 1);
      const fx  = sx - x0;
      const fy  = sy - y0;
      dst[y * dstW + x] =
        (1 - fy) * ((1 - fx) * src[y0 * srcW + x0] + fx * src[y0 * srcW + x1]) +
           fy    * ((1 - fx) * src[y1 * srcW + x0] + fx * src[y1 * srcW + x1]);
    }
  }
  return dst;
}

/**
 * Build a jet-coloured heatmap from the backbone's spatial feature tensor.
 *
 * Algorithm (gradient-free mean-CAM):
 *   1. Average absolute activations across all C channels → (H, W) map
 *   2. Normalise to [0, 1]
 *   3. Bilinear upsample to 320×320
 *   4. Apply jet colormap → RGBA Uint8Array
 *   5. Encode as JPEG and return a data: URI
 *
 * @param {Float32Array} data   Raw tensor data from the ONNX output
 * @param {number[]}     dims   [batch, C, H, W]
 * @returns {string}            "data:image/jpeg;base64,…"
 */
function buildHeatmap(data, dims) {
  const [, C, H, W] = dims;

  // 1. Mean |activation| over channels
  const cam = new Float32Array(H * W);
  for (let c = 0; c < C; c++) {
    const base = c * H * W;
    for (let i = 0; i < H * W; i++) cam[i] += Math.abs(data[base + i]);
  }
  for (let i = 0; i < H * W; i++) cam[i] /= C;

  // 2. Normalise
  let lo = cam[0], hi = cam[0];
  for (const v of cam) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const range = hi - lo || 1;
  for (let i = 0; i < cam.length; i++) cam[i] = (cam[i] - lo) / range;

  // 3. Upsample to output display resolution
  const OUT = IMG_SIZE;
  const up  = bilinearResize(cam, W, H, OUT, OUT);

  // 4. Jet colormap → RGBA
  const rgba = new Uint8Array(OUT * OUT * 4);
  for (let i = 0; i < OUT * OUT; i++) {
    const [r, g, b] = jetColor(up[i]);
    rgba[i * 4]     = r;
    rgba[i * 4 + 1] = g;
    rgba[i * 4 + 2] = b;
    rgba[i * 4 + 3] = 255;
  }

  // 5. JPEG encode via jpeg-js (same library used for decode)
  const encoded = jpeg.encode({ data: rgba, width: OUT, height: OUT }, 85);

  // 6. Uint8Array → base64 string (chunked to avoid call-stack overflow)
  const bytes = new Uint8Array(encoded.data);
  return `data:image/jpeg;base64,${encodeBase64(bytes)}`;
}

// ── Post-processing helpers ───────────────────────────────────────────────────

/** Numerically stable softmax over a Float32Array. */
function softmax(logits) {
  const arr    = Array.from(logits);
  const maxVal = Math.max(...arr);
  const exps   = arr.map((x) => Math.exp(x - maxVal));
  const sum    = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/**
 * Parse a PlantWild class name (e.g. "tomato early blight") into
 * human-readable plant name and disease label.
 *   "apple black rot"  → { plantName: "Apple",  disease: "Black rot" }
 *   "tomato leaf"      → { plantName: "Tomato", disease: "Healthy" }
 */
function parseClassName(className) {
  const words     = (className ?? "unknown").trim().split(/\s+/);
  const plantName = words[0].charAt(0).toUpperCase() + words[0].slice(1);
  const rest      = words.slice(1).join(" ");
  const disease   = !rest || rest === "leaf"
    ? "Healthy"
    : rest.charAt(0).toUpperCase() + rest.slice(1);
  return { plantName, disease };
}

// ── analyzeLeaf ───────────────────────────────────────────────────────────────

/**
 * Runs the full multimodal inference pipeline on a captured image + text.
 *
 * @param {string} imageUri      Local file:// URI from expo-camera
 * @param {string} symptomText   User's symptom description (may be empty)
 * @returns {Promise<{
 *   plantName:   string,
 *   disease:     string,
 *   confidence:  number,   // 0–100
 *   treatment:   string,
 *   heatmapUri:  string,   // data:image/jpeg;base64,… for <Image source={{uri:…}} />
 * }>}
 */
export async function analyzeLeaf(imageUri, symptomText = "") {
  if (!imageSession || !textSession || !mlpSession || !heatmapSession) {
    throw new Error("Model not loaded — call loadModel() first");
  }

  // ── 1. Pre-process inputs in parallel ──────────────────────────────────────
  const { input_ids, attention_mask } = tokenize(symptomText);

  const [imageTensor] = await Promise.all([preprocessImage(imageUri)]);

  const inputIdsTensor      = new Tensor("int64", input_ids,      [1, 128]);
  const attentionMaskTensor = new Tensor("int64", attention_mask, [1, 128]);

  // ── 2. Run all three ONNX sessions ─────────────────────────────────────────
  const [imgOutputs, textOutputs] = await Promise.all([
    imageSession.run({ image: imageTensor }),
    textSession.run({ input_ids: inputIdsTensor, attention_mask: attentionMaskTensor }),
  ]);

  const imgEmb     = imgOutputs["img_emb"];
  const spatialFeat = imgOutputs["spatial_feat"];
  const textEmb    = textOutputs["text_emb"];

  const mlpOutputs = await mlpSession.run({
    img_emb:  imgEmb,
    text_emb: textEmb,
  });

  // ── 3. Classification ───────────────────────────────────────────────────────
  const logits       = mlpOutputs["logits"].data; // Float32Array (89,)
  const probs        = softmax(logits);
  const predictedIdx = probs.indexOf(Math.max(...probs));
  const confidence   = Math.round(probs[predictedIdx] * 100);

  const className            = INDEX_TO_CLASS[predictedIdx] ?? "unknown";
  const { plantName, disease } = parseClassName(className);
  const treatment = TREATMENTS[className]
    ?? "Consult a local agricultural expert for diagnosis and treatment advice.";

  // ── 4. Heatmap (via trained heatmap generator model) ────────────────────────
  let heatmapUri = null;

  try {
    const heatmapOutput = await heatmapSession.run({ spatial_feat: spatialFeat });
    // heatmapOutput["heatmap"] is (1, 1, 320, 320) with values in [0, 1]
    const heatmapTensor = heatmapOutput["heatmap"];
    heatmapUri = buildHeatmapFromModel(heatmapTensor.data, IMG_SIZE);
  } catch (err) {
    console.warn("[model] Heatmap generator failed, falling back to mean-CAM:", err);
    try {
      heatmapUri = buildHeatmap(spatialFeat.data, spatialFeat.dims);
    } catch (err2) {
      console.warn("[model] Mean-CAM fallback also failed:", err2);
    }
  }

  return { plantName, disease, confidence, treatment, heatmapUri };
}
