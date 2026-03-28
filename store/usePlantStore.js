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

const usePlantStore = create((set, get) => ({

  // ── State ──────────────────────────────────────────────────────────────────

  /** Plants displayed on the My Farm home screen */
  plants: [
    {
      id: "1",
      name: "Monsterra",
      status: "healthy",
      isFavourite: false,
      image: require("../assets/monsterra.jpg"),
      bgColor: "#c8f0d6",
    },
    {
      id: "2",
      name: "Sunflower",
      status: "diseased",
      isFavourite: false,
      image: require("../assets/sunflower.jpg"),
      bgColor: "#fde68a",
    },
    {
      id: "3",
      name: "Bell Pepper",
      status: "healthy",
      isFavourite: true,
      image: require("../assets/bellpepper.jpg"),
      bgColor: "#bbf7d0",
    },
  ],

  /** URI (file path) of the photo the user just took with the camera */
  capturedImageUri: null,

  /**
   * The result returned by the AI model:
   * { disease: string, confidence: number, treatment: string }
   */
  analysisResult: null,

  /** The 5 most recent scan records loaded from device storage */
  recentScans: [],

  /**
   * Whether the PyTorch model has been loaded into memory.
   * We track this so we can show a loading indicator on the home screen
   * and block the camera until the model is ready.
   */
  isModelLoaded: false,

  // ── Actions ────────────────────────────────────────────────────────────────

  /**
   * Called by the camera screen after the user takes a photo.
   * @param {string} uri  e.g. "file:///path/to/image.jpg"
   */
  setCapturedImage: (uri) => set({ capturedImageUri: uri }),

  /**
   * Called by the processing screen after inference completes.
   * @param {{ disease, confidence, treatment }} result
   */
  setAnalysisResult: (result) => set({ analysisResult: result }),

  /**
   * Called in the root layout (_layout.jsx) once the model file is loaded.
   */
  setModelLoaded: (loaded) => set({ isModelLoaded: loaded }),

  /**
   * Reads the stored scan history from AsyncStorage and puts it in state.
   * Call this when the app launches and when the History tab is focused.
   */
  loadRecentScans: async () => {
    const scans = await getRecentScans();
    set({ recentScans: scans });
  },

  /**
   * Saves a completed scan to AsyncStorage, then refreshes `recentScans`.
   * @param {{ imageUri, disease, confidence, treatment, date }} scan
   */
  addScan: async (scan) => {
    await saveRecentScan(scan);    // persist to device storage
    await get().loadRecentScans(); // update the in-memory list
  },

  /**
   * Adds a newly scanned plant to the My Farm list.
   * @param {{ id, name, status, image, bgColor }} plant
   */
  addPlant: (plant) => set((state) => ({ plants: [...state.plants, plant] })),

  /**
   * Resets photo + result so the next scan starts clean.
   * Call this when the user taps "New Scan" on the results screen.
   */
  resetSession: () => set({ capturedImageUri: null, analysisResult: null }),
}));

export default usePlantStore;
