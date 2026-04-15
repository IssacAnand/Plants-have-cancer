// store/usePlantStore.js
//
// ─────────────────────────────────────────────────────────────────────────────
// CONCEPT: Zustand Global State
// ─────────────────────────────────────────────────────────────────────────────
//
// Imagine each screen in your app is a person in a building. Without Zustand,
// if Person A (Camera screen) takes a photo and Person C (Results screen) needs
// to see it, Person A would have to pass it to Person B (Preview) who passes
// it to Person C — this is called "prop-drilling" and gets messy fast.
//
// Zustand creates a shared bulletin board in the lobby. Person A pins the
// photo there, and Person C walks straight to the board to read it.
//
// HOW IT WORKS:
//   create((set, get) => ({ ... })) defines the store.
//   - `set` updates state:  set({ myValue: 123 })
//   - `get` reads state:    get().myValue
//   - Any component calls the hook:  const value = usePlantStore(s => s.myValue)
//   - React re-renders only that component when value changes.
// ─────────────────────────────────────────────────────────────────────────────

import { create } from "zustand";
import { saveRecentScan, getRecentScans } from "../utils/storage";

// ── Mock scan history shown before the user has done any real scans ───────────
const MOCK_SCANS = [
  {
    plantName:   "Sunflower",
    imageSource: require("../assets/sunflower.jpg"),
    imageUri:    null,
    disease:     "Light Blight with Sun Spots",
    confidence:  97,
    treatment:
      "Remove and destroy affected leaves immediately. Apply a copper-based fungicide spray every 7–10 days. Improve air circulation by spacing plants and avoid overhead watering to keep foliage dry.",
    date:        "2025-01-28T10:58:00.000Z",
    symptomText: "Yellow spots on leaves",
    isMock:      true,
  },
  {
    plantName:   "Monsterra",
    imageSource: require("../assets/monsterra.jpg"),
    imageUri:    null,
    disease:     "Root Rot",
    confidence:  85,
    treatment:
      "Repot the plant using well-draining soil. Trim any black or mushy roots before repotting. Reduce watering frequency and ensure the pot has drainage holes. Allow soil to dry between waterings.",
    date:        "2025-01-25T08:30:00.000Z",
    symptomText: "Wilting and yellowing leaves",
    isMock:      true,
  },
  {
    plantName:   "Bell Pepper",
    imageSource: require("../assets/bellpepper.jpg"),
    imageUri:    null,
    disease:     "Healthy Plant",
    confidence:  92,
    treatment:
      "No treatment needed. Continue regular watering and fertilisation. Monitor for pests or early disease signs. Ensure full sun exposure (6–8 hours daily) and consistent moisture.",
    date:        "2025-01-20T14:15:00.000Z",
    symptomText: "",
    isMock:      true,
  },
];

const usePlantStore = create((set, get) => ({

  // ── State ──────────────────────────────────────────────────────────────────

  /** Plants displayed on the My Farm home screen */
  plants: [
    {
      id:         "1",
      name:       "Monsterra",
      status:     "healthy",
      isFavourite: false,
      image:      require("../assets/monsterra.jpg"),
      bgColor:    "#c8f0d6",
      disease:    "Root Rot",
      confidence: 85,
      treatment:
        "Repot the plant using well-draining soil. Trim any black or mushy roots before repotting. Reduce watering frequency and ensure the pot has drainage holes. Allow soil to dry between waterings.",
    },
    {
      id:         "2",
      name:       "Sunflower",
      status:     "diseased",
      isFavourite: false,
      image:      require("../assets/sunflower.jpg"),
      bgColor:    "#fde68a",
      disease:    "Light Blight with Sun Spots",
      confidence: 97,
      treatment:
        "Remove and destroy affected leaves immediately. Apply a copper-based fungicide spray every 7–10 days. Improve air circulation by spacing plants and avoid overhead watering to keep foliage dry.",
    },
    {
      id:         "3",
      name:       "Bell Pepper",
      status:     "healthy",
      isFavourite: true,
      image:      require("../assets/bellpepper.jpg"),
      bgColor:    "#bbf7d0",
      disease:    "Healthy Plant",
      confidence: 92,
      treatment:
        "No treatment needed. Continue regular watering and fertilisation. Monitor for pests or early disease signs. Ensure full sun exposure (6–8 hours daily) and consistent moisture.",
    },
  ],

  /** URI (file path) of the photo the user just took with the camera */
  capturedImageUri: null,

  /** Plant name selected by the user on the preview screen */
  capturedPlantName: null,

  /** Symptom description typed by the user on the preview screen */
  capturedText: null,

  /**
   * The result returned by the AI model:
   * { disease: string, confidence: number, treatment: string }
   */
  analysisResult: null,

  /**
   * The scan the user tapped "View in Detail" on (from History or My Farm).
   * Null when viewing a fresh live scan result.
   */
  selectedScan: null,

  /** The 5 most recent scan records — seeded with mock data until real scans exist */
  recentScans: MOCK_SCANS,

  /**
   * Whether the ONNX model has been loaded into memory.
   * We track this so we can show a loading indicator and block the camera.
   */
  isModelLoaded: false,

  // ── Actions ────────────────────────────────────────────────────────────────

  /** Called by the camera / gallery flow after an image is selected. */
  setCapturedImage: (uri) => set({ capturedImageUri: uri }),

  /** Called by the preview screen when the user selects a plant name. */
  setCapturedPlantName: (name) => set({ capturedPlantName: name }),

  /** Called by the preview screen when the user taps "Get Diagnosis". */
  setCapturedText: (text) => set({ capturedText: text }),

  /** Called by the processing screen after inference completes. */
  setAnalysisResult: (result) => set({ analysisResult: result }),

  /** Sets the scan to display when navigating to results from History / My Farm. */
  setSelectedScan: (scan) => set({ selectedScan: scan }),

  /** Called in the root layout once the model file is loaded. */
  setModelLoaded: (loaded) => set({ isModelLoaded: loaded }),

  /**
   * Reads the stored scan history from AsyncStorage and puts it in state.
   * Falls back to MOCK_SCANS when AsyncStorage is empty (fresh install).
   */
  loadRecentScans: async () => {
    const scans = await getRecentScans();
    if (scans.length > 0) {
      set({ recentScans: scans });
    }
    // If AsyncStorage is empty, the initial MOCK_SCANS remain in state
  },

  /**
   * Saves a completed scan to AsyncStorage, then refreshes recentScans.
   * @param {{ plantName, imageUri, symptomText, disease, confidence, treatment, date }} scan
   */
  addScan: async (scan) => {
    await saveRecentScan(scan);
    await get().loadRecentScans();
  },

  /** Adds a newly scanned plant to the My Farm list. */
  addPlant: (plant) => set((state) => ({ plants: [...state.plants, plant] })),

  /** Resets all session state so the next scan starts clean. */
  resetSession: () => set({
    capturedImageUri:  null,
    capturedPlantName: null,
    capturedText:      null,
    analysisResult:    null,
    selectedScan:      null,
  }),
}));

export default usePlantStore;
