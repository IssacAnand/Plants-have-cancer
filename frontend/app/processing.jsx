// app/processing.jsx  — Processing / Analyzing Screen
//
// ─────────────────────────────────────────────────────────────────────────────
// CONCEPT: useEffect for async side effects
// ─────────────────────────────────────────────────────────────────────────────
//
// You cannot use `await` directly at the top level of a component.
// The correct pattern for async work after a component mounts is:
//
//   useEffect(() => {
//     async function doWork() { ... }
//     doWork();
//   }, []);
//
// This screen:
//  1. Mounts → shows the spinner immediately
//  2. useEffect fires → runs analyzeLeaf() in the background
//  3. On success → saves result to Zustand + navigates to Results
//  4. On error → navigates back (or shows an error state)
// ─────────────────────────────────────────────────────────────────────────────

import { useEffect, useRef } from "react";
import { View, Text, ActivityIndicator, SafeAreaView, Animated } from "react-native";
import { useRouter } from "expo-router";

import { analyzeLeaf } from "../utils/modelInference";
import usePlantStore from "../store/usePlantStore";

export default function ProcessingScreen() {
  const router = useRouter();

  // Grab what we need from the store
  const capturedImageUri  = usePlantStore((state) => state.capturedImageUri);
  const capturedPlantName = usePlantStore((state) => state.capturedPlantName);
  const capturedText      = usePlantStore((state) => state.capturedText);
  const isModelLoaded     = usePlantStore((state) => state.isModelLoaded);
  const setAnalysisResult = usePlantStore((state) => state.setAnalysisResult);
  const addScan           = usePlantStore((state) => state.addScan);

  // Animated value for the pulsing circle
  const pulse      = useRef(new Animated.Value(1)).current;
  const hasStarted = useRef(false);

  // ── Pulse animation ──────────────────────────────────────────────────────
  useEffect(() => {
    // Animated.loop repeats the sequence forever
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1.15, duration: 800, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 1.00, duration: 800, useNativeDriver: true }),
      ])
    ).start();
  }, []);

  // ── Run inference ────────────────────────────────────────────────────────
  useEffect(() => {
    if (!capturedImageUri) {
      // Shouldn't happen, but guard anyway
      router.replace("/(tabs)/scan");
      return;
    }

    if (!isModelLoaded || hasStarted.current) {
      return;
    }

    hasStarted.current = true;

    const plantText = capturedPlantName?.trim();
    const symptomText = capturedText?.trim();
    const combinedText = plantText
      ? `${plantText} plant${symptomText ? `. Symptoms: ${symptomText}` : ""}`
      : (symptomText ?? "");

    console.log("[processing] Inference text:", combinedText);

    async function runInference() {
      try {
        // analyzeLeaf() runs three ONNX sessions (image + text + MLP)
        const result = await analyzeLeaf(capturedImageUri, combinedText, plantText ?? "");

        // 1. Store the result so the Results screen can read it
        setAnalysisResult(result);

        // 2. Persist to history (AsyncStorage via Zustand action)
        await addScan({
          plantName:   result.plantName,
          imageUri:    capturedImageUri,
          symptomText: combinedText,
          disease:     result.disease,
          confidence:  result.confidence,
          treatment:   result.treatment,
          date:        new Date().toISOString(),
        });

        // 3. Navigate to the Results screen
        //    replace() so the user can't go Back to this loading screen
        router.replace("/results");

      } catch (err) {
        console.error("Inference error:", err);
        // TODO: show a proper error toast; for now go back to camera
        router.replace("/(tabs)/scan");
      }
    }

    runInference();
  }, [addScan, capturedImageUri, capturedPlantName, capturedText, isModelLoaded, router, setAnalysisResult]);

  // ── UI ────────────────────────────────────────────────────────────────────
  return (
    <SafeAreaView className="flex-1 bg-surface items-center justify-center px-8">
      {/* Pulsing circle */}
      <Animated.View
        style={{ transform: [{ scale: pulse }] }}
        className="w-40 h-40 rounded-full bg-green-100 items-center justify-center mb-8"
      >
        <ActivityIndicator size="large" color="#22C55E" />
      </Animated.View>

      <Text className="text-2xl font-bold text-gray-800 text-center">
        Processing
      </Text>

      <Text className="text-gray-500 text-center mt-3 text-base">
        {isModelLoaded ? "Analyzing leaf image and symptoms..." : "Loading AI model..."}
      </Text>

      <Text className="text-gray-400 text-center mt-1 text-sm">
        (Should be fast)
      </Text>

      {/* Step indicators */}
      <View className="mt-10 w-full bg-white rounded-2xl p-5 shadow-sm">
        {[
          { label: "Loading image",          done: true  },
          { label: "Preparing plant name",   done: Boolean(capturedPlantName?.trim()) },
          { label: "Preparing symptom text", done: Boolean(capturedText?.trim()) },
          { label: "Running AI model",       done: false },
          { label: "Preparing results",      done: false },
        ].map((step, i) => (
          <View key={i} className="flex-row items-center py-2">
            <View
              className={`w-5 h-5 rounded-full mr-3 items-center justify-center ${
                step.done ? "bg-primary" : "bg-gray-200"
              }`}
            >
              {step.done && <Text className="text-white text-xs">✓</Text>}
            </View>
            <Text className={`text-sm ${step.done ? "text-gray-700" : "text-gray-400"}`}>
              {step.label}
            </Text>
          </View>
        ))}
      </View>
    </SafeAreaView>
  );
}
