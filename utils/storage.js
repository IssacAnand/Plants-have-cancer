// utils/storage.js
//
// ─────────────────────────────────────────────────────────────────────────────
// CONCEPT: AsyncStorage (Mobile Device Storage)
// ─────────────────────────────────────────────────────────────────────────────
//
// In web development you use `localStorage.setItem("key", "value")`.
// In React Native there is no localStorage, so we use AsyncStorage instead.
//
// Key differences from web localStorage:
//   1. ASYNCHRONOUS — every operation returns a Promise, so we use async/await.
//      This is because reading/writing to storage is an I/O operation that takes
//      a tiny moment; the app keeps running while it happens.
//   2. STRINGS ONLY — values must be strings. Objects must be JSON.stringify()'d
//      before saving and JSON.parse()'d after reading.
//   3. PERSISTENT — data survives app restarts (until the user uninstalls or
//      you explicitly delete it).
// ─────────────────────────────────────────────────────────────────────────────

import AsyncStorage from "@react-native-async-storage/async-storage";

/** The key under which the scan array is stored */
const SCANS_KEY = "plant_recent_scans";

/** Maximum number of scans to keep */
const MAX_SCANS = 5;

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Save a new scan result to the device.
 * Automatically trims the list to MAX_SCANS entries (oldest are dropped).
 *
 * @param {{
 *   plantName:  string,   - user-facing name of the plant (e.g. "Sunflower")
 *   imageUri:   string,   - local file path of the captured image
 *   disease:    string,   - detected disease name
 *   confidence: number,   - 0–100 confidence percentage
 *   treatment:  string,   - recommended treatment text
 *   date:       string,   - ISO date string, e.g. new Date().toISOString()
 * }} scan
 */
export async function saveRecentScan(scan) {
  try {
    // 1. Load whatever is already stored
    const existing = await getRecentScans();

    // 2. Prepend the newest scan so index 0 is always the most recent
    //    .slice(0, MAX_SCANS) drops anything beyond the 5th entry
    const updated = [scan, ...existing].slice(0, MAX_SCANS);

    // 3. Convert the array to a JSON string and store it
    await AsyncStorage.setItem(SCANS_KEY, JSON.stringify(updated));
  } catch (err) {
    // Never crash the app over a storage failure — just log and continue
    console.error("[storage] saveRecentScan failed:", err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Retrieve the list of recent scans from the device.
 * Returns an empty array if nothing is stored yet.
 *
 * @returns {Promise<Array>}
 */
export async function getRecentScans() {
  try {
    const raw = await AsyncStorage.getItem(SCANS_KEY);

    // getItem returns null when the key doesn't exist yet
    return raw ? JSON.parse(raw) : [];
  } catch (err) {
    console.error("[storage] getRecentScans failed:", err);
    return [];
  }
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Delete all stored scans (useful for a "Clear History" button).
 */
export async function clearAllScans() {
  try {
    await AsyncStorage.removeItem(SCANS_KEY);
  } catch (err) {
    console.error("[storage] clearAllScans failed:", err);
  }
}
