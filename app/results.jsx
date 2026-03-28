// app/results.jsx  — Results Screen
//
// ─────────────────────────────────────────────────────────────────────────────
// CONCEPT: Composing data from multiple Zustand slices
// ─────────────────────────────────────────────────────────────────────────────
//
// This screen needs BOTH the image URI and the analysis result.
// We pull each one separately from the store — Zustand only re-renders
// the component when those specific values change.
//
// resetSession() clears both values so the next scan starts clean.
// ─────────────────────────────────────────────────────────────────────────────

import { View, Text, Image, ScrollView, SafeAreaView, TouchableOpacity } from "react-native";
import { useRouter } from "expo-router";

import usePlantStore from "../store/usePlantStore";
import PrimaryButton from "../components/PrimaryButton";

export default function ResultsScreen() {
  const router = useRouter();

  const capturedImageUri  = usePlantStore((s) => s.capturedImageUri);
  const analysisResult    = usePlantStore((s) => s.analysisResult);
  const resetSession      = usePlantStore((s) => s.resetSession);

  // Guard: if we somehow arrive here with no result, go home
  if (!analysisResult) {
    router.replace("/(tabs)/index");
    return null;
  }

  const { disease, confidence, treatment } = analysisResult;

  // Colour the confidence text based on how sure the model is
  const confidenceColor =
    confidence >= 80 ? "#16A34A" :   // green  — high confidence
    confidence >= 60 ? "#D97706" :   // amber  — medium
                       "#DC2626";    // red    — low

  function handleNewScan() {
    resetSession();              // clear photo + result from global state
    router.replace("/(tabs)/scan"); // go straight to camera
  }

  function handleHome() {
    resetSession();
    router.replace("/(tabs)/index");
  }

  return (
    <SafeAreaView className="flex-1 bg-surface">
      {/* ── Header ── */}
      <View className="flex-row items-center px-6 py-4 border-b border-gray-100 bg-white">
        <TouchableOpacity onPress={handleHome} className="mr-4">
          <Text className="text-2xl">←</Text>
        </TouchableOpacity>
        <Text className="text-xl font-bold text-gray-800 flex-1 text-center">
          Results
        </Text>
        <View className="w-8" />
      </View>

      <ScrollView
        className="flex-1"
        contentContainerStyle={{ padding: 20 }}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Leaf photo thumbnail ── */}
        {capturedImageUri && (
          <Image
            source={{ uri: capturedImageUri }}
            className="w-full h-48 rounded-2xl bg-gray-100 mb-4"
            resizeMode="cover"
          />
        )}

        {/* ── Result card ── */}
        <View className="bg-white rounded-2xl p-5 shadow-sm mb-4">
          {/* Disease name */}
          <Text className="text-xl font-bold" style={{ color: "#F97316" }}>
            {disease}
          </Text>

          {/* Confidence */}
          <Text className="text-base font-semibold mt-1" style={{ color: confidenceColor }}>
            {confidence}% Confidence
          </Text>

          {/* Divider */}
          <View className="h-px bg-gray-100 my-4" />

          {/* Treatment section */}
          <Text className="text-gray-500 text-xs uppercase tracking-widest mb-2 font-semibold">
            Recommended Action
          </Text>
          <Text className="text-gray-700 text-base leading-relaxed">
            {treatment}
          </Text>
        </View>

        {/* ── Confidence meter ── */}
        <View className="bg-white rounded-2xl p-5 shadow-sm mb-4">
          <Text className="text-gray-500 text-xs uppercase tracking-widest mb-3 font-semibold">
            Detection Confidence
          </Text>
          {/* Track */}
          <View className="h-3 bg-gray-100 rounded-full overflow-hidden">
            {/* Fill — width is a percentage of confidence */}
            <View
              className="h-full rounded-full"
              style={{
                width: `${confidence}%`,
                backgroundColor: confidenceColor,
              }}
            />
          </View>
          <Text className="text-right text-sm font-bold mt-1" style={{ color: confidenceColor }}>
            {confidence}%
          </Text>
        </View>

        {/* ── Offline badge ── */}
        <View className="flex-row items-center bg-green-50 px-4 py-3 rounded-2xl border border-green-100 mb-6">
          <Text className="text-xl mr-2">🛡</Text>
          <View>
            <Text className="text-green-700 font-semibold text-sm">Offline Mode Active</Text>
            <Text className="text-green-600 text-xs">Analysis performed on-device</Text>
          </View>
        </View>

        {/* ── Actions ── */}
        <PrimaryButton label="TAKE NEW SCAN" onPress={handleNewScan} />
        <View className="h-3" />
        <PrimaryButton label="GO HOME" onPress={handleHome} outline />
      </ScrollView>
    </SafeAreaView>
  );
}
