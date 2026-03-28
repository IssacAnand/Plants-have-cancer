// app/preview.jsx  — Image Preview Screen
//
// ─────────────────────────────────────────────────────────────────────────────
// CONCEPT: Reading from Zustand store
// ─────────────────────────────────────────────────────────────────────────────
//
// This screen never touches the camera — it just reads the URI that the
// Scan screen saved to the Zustand store via setCapturedImage().
//
// Pattern: usePlantStore(selector)
//   The selector function picks ONLY the values this screen needs.
//   Zustand will only re-render this component when those specific values change.
//
// router.replace() vs router.push():
//   push  → adds to the history stack (user can go back)
//   replace → replaces the current route (user cannot go back to preview
//             once they've confirmed the photo — avoids a confusing back stack)
// ─────────────────────────────────────────────────────────────────────────────

import { View, Text, Image, SafeAreaView } from "react-native";
import { useRouter } from "expo-router";

import usePlantStore from "../store/usePlantStore";
import PrimaryButton from "../components/PrimaryButton";

export default function PreviewScreen() {
  const router = useRouter();

  // Read values from the global store
  const capturedImageUri  = usePlantStore((state) => state.capturedImageUri);
  const setCapturedImage  = usePlantStore((state) => state.setCapturedImage);

  function handleUsePhoto() {
    // Navigate to the processing screen which runs inference
    // replace() so the user can't hit Back and come back here after analysing
    router.replace("/processing");
  }

  function handleRetake() {
    // Clear the stored image and go back to the camera
    setCapturedImage(null);
    router.back();
  }

  return (
    <SafeAreaView className="flex-1 bg-white">
      {/* ── Header ── */}
      <View className="px-6 py-4 border-b border-gray-100">
        <Text className="text-xl font-bold text-gray-800 text-center">
          Image Preview
        </Text>
      </View>

      {/* ── Photo ── */}
      <View className="flex-1 items-center justify-center px-6">
        {capturedImageUri ? (
          <Image
            source={{ uri: capturedImageUri }}
            className="w-full rounded-2xl bg-gray-100"
            style={{ aspectRatio: 3 / 4 }}  // portrait crop
            resizeMode="cover"
          />
        ) : (
          // Fallback in case we arrive here with no image (shouldn't happen)
          <View className="w-full h-96 rounded-2xl bg-gray-100 items-center justify-center">
            <Text className="text-gray-400 text-4xl">🌿</Text>
            <Text className="text-gray-400 mt-2">No image captured</Text>
          </View>
        )}
      </View>

      {/* ── Action buttons ── */}
      <View className="px-8 pb-10 gap-y-3">
        <PrimaryButton
          label="USE PHOTO"
          onPress={handleUsePhoto}
          disabled={!capturedImageUri}
        />
        <PrimaryButton
          label="RETAKE"
          onPress={handleRetake}
          outline
        />
      </View>
    </SafeAreaView>
  );
}
