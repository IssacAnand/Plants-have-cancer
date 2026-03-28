// app/camera.jsx — Live Camera Screen
// (Previously lived at app/(tabs)/scan.jsx)

import { useState, useRef } from "react";
import { View, Text, TouchableOpacity, SafeAreaView } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useRouter } from "expo-router";

import usePlantStore from "../store/usePlantStore";

export default function CameraScreen() {
  const router    = useRouter();
  const cameraRef = useRef(null);

  const [facing,   setFacing]   = useState("back");
  const [isTaking, setIsTaking] = useState(false);

  const [permission, requestPermission] = useCameraPermissions();

  const setCapturedImage = usePlantStore((state) => state.setCapturedImage);

  // ── Permission not yet determined ──────────────────────────────────────────
  if (!permission) {
    return <View className="flex-1 bg-black" />;
  }

  // ── Permission denied ──────────────────────────────────────────────────────
  if (!permission.granted) {
    return (
      <SafeAreaView className="flex-1 bg-gray-900 items-center justify-center px-8">
        <Text className="text-white text-5xl mb-6">📷</Text>
        <Text className="text-white text-xl font-bold text-center">
          Camera Access Needed
        </Text>
        <Text className="text-gray-400 text-center mt-3 mb-8">
          We need camera access to photograph your plant leaves for disease detection.
        </Text>
        <TouchableOpacity
          onPress={requestPermission}
          className="bg-primary px-8 py-4 rounded-full"
        >
          <Text className="text-white font-bold text-base">Grant Permission</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  // ── Take a photo ───────────────────────────────────────────────────────────
  async function handleCapture() {
    if (!cameraRef.current || isTaking) return;

    setIsTaking(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      setCapturedImage(photo.uri);
      router.push("/preview");
    } catch (err) {
      console.error("Capture failed:", err);
    } finally {
      setIsTaking(false);
    }
  }

  // ── Render camera ──────────────────────────────────────────────────────────
  return (
    <View className="flex-1 bg-black">
      <CameraView ref={cameraRef} className="flex-1" facing={facing}>
        {/* ── Top controls ── */}
        <SafeAreaView className="flex-row justify-between items-center px-6 pt-2">
          <TouchableOpacity
            onPress={() => router.back()}
            className="w-10 h-10 items-center justify-center"
          >
            <Text className="text-white text-2xl">✕</Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => setFacing((f) => (f === "back" ? "front" : "back"))}
            className="w-10 h-10 items-center justify-center"
          >
            <Text className="text-white text-2xl">🔄</Text>
          </TouchableOpacity>
        </SafeAreaView>

        {/* ── Targeting guide overlay ── */}
        <View className="flex-1 items-center justify-center">
          <View
            className="w-64 h-64 border-2 border-white rounded-2xl"
            style={{ opacity: 0.6 }}
          />
          <Text className="text-white text-sm mt-4 opacity-70">
            Frame the leaf within the box
          </Text>
        </View>

        {/* ── Capture button ── */}
        <View className="pb-12 items-center">
          <TouchableOpacity
            onPress={handleCapture}
            disabled={isTaking}
            className="w-20 h-20 rounded-full border-4 border-white items-center justify-center"
            style={{ backgroundColor: isTaking ? "#ffffff88" : "#ffffff" }}
          >
            <View className="w-14 h-14 rounded-full bg-white border-2 border-gray-200" />
          </TouchableOpacity>
          <Text className="text-white text-xs mt-3 opacity-60">
            {isTaking ? "Capturing…" : "Tap to capture"}
          </Text>
        </View>
      </CameraView>
    </View>
  );
}
