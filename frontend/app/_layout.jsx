// app/_layout.jsx
//
// ─────────────────────────────────────────────────────────────────────────────
// CONCEPT: Expo Router Root Layout
// ─────────────────────────────────────────────────────────────────────────────
//
// Expo Router uses file-based routing — just like Next.js for web.
// The folder structure inside /app becomes your navigation structure:
//
//   app/_layout.jsx          → wraps EVERY screen (root shell)
//   app/(tabs)/_layout.jsx   → wraps the bottom-tab screens
//   app/(tabs)/index.jsx     → the Home tab  (route: "/")
//   app/results.jsx          → a full-screen stack route (route: "/results")
//
// _layout.jsx files are NEVER shown as screens themselves — they define
// how child screens are arranged (Stack, Tabs, Drawer, etc.)
//
// This ROOT layout does three things:
//   1. Sets up a Stack navigator (the base navigation container)
//   2. Hides the default header on all screens
//   3. Loads the PyTorch model as soon as the app starts
// ─────────────────────────────────────────────────────────────────────────────

import { useEffect } from "react";
import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";

import { loadModel } from "../utils/modelInference";
import usePlantStore from "../store/usePlantStore";

export default function RootLayout() {
  // Pull the setter from our Zustand store
  const setModelLoaded = usePlantStore((state) => state.setModelLoaded);
  const loadRecentScans = usePlantStore((state) => state.loadRecentScans);

  useEffect(() => {
    // Run once when the app first mounts
    async function bootstrap() {
      const appStartTime = Date.now();
      console.log("[bootstrap] useEffect hook fired, calling bootstrap");
      console.log("[bootstrap] Starting model and history load...");
      
      // Kick off the model load in the background
      console.log("[bootstrap] Calling loadModel()");
      const loadStart = Date.now();
      const success = await loadModel();
      const loadElapsed = Date.now() - loadStart;
      console.log("[bootstrap] loadModel() returned:", success, `(${loadElapsed}ms)`);
      setModelLoaded(success);

      // Pre-load scan history so the History tab is ready
      console.log("[bootstrap] Calling loadRecentScans()");
      await loadRecentScans();
      console.log("[bootstrap] loadRecentScans() completed");
      
      const totalElapsed = Date.now() - appStartTime;
      console.log(`[bootstrap] ✅ Bootstrap complete in ${totalElapsed}ms`);
    }

    bootstrap().catch((err) => {
      console.error("[bootstrap] bootstrap() threw:", err?.message ?? err, err?.stack ?? "");
    });
  }, []); // empty array = run once on mount

  return (
    <>
      {/* StatusBar controls the top bar (time, battery, etc.) */}
      <StatusBar style="dark" />

      {/*
        Stack is the navigation container.
        screenOptions applies to ALL child screens.
        headerShown: false hides the built-in navigation header so each screen
        can draw its own custom header (or none at all).
      */}
      <Stack screenOptions={{ headerShown: false }} initialRouteName="index">
        {/* Root index — redirects immediately to /splash */}
        <Stack.Screen name="index" options={{ animation: "none" }} />

        {/* Splash — shown first; navigates to (tabs) on CTA tap */}
        <Stack.Screen name="splash" options={{ animation: "fade" }} />

        {/* The (tabs) group routes to app/(tabs)/_layout.jsx */}
        <Stack.Screen name="(tabs)" />

        {/* These are full-screen "pushed" routes — they slide in over the tabs */}
        <Stack.Screen name="camera"     />
        <Stack.Screen name="preview"    />
        <Stack.Screen name="processing" />
        <Stack.Screen name="results"    />
      </Stack>
    </>
  );
}
