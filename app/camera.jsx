// app/camera.jsx — Live Camera Screen

import { useState, useRef } from "react";
import { View, Text, TouchableOpacity, SafeAreaView, Image, PanResponder } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useRouter } from "expo-router";
import { RotateCcw } from "lucide-react-native";

import usePlantStore from "../store/usePlantStore";

export default function CameraScreen() {
  const router    = useRouter();
  const cameraRef = useRef(null);

  const [isTaking, setIsTaking] = useState(false);
  const [zoom,     setZoom]     = useState(0);

  // Refs to track pinch state without triggering re-renders on every move
  const zoomRef         = useRef(0);
  const lastDistanceRef = useRef(null);

  const [permission, requestPermission] = useCameraPermissions();

  const capturedImageUri = usePlantStore((s) => s.capturedImageUri);
  const setCapturedImage = usePlantStore((s) => s.setCapturedImage);
  const resetSession     = usePlantStore((s) => s.resetSession);

  // ── Pinch-to-zoom via PanResponder ────────────────────────────────────────
  const pinchResponder = useRef(
    PanResponder.create({
      // Only activate when two fingers are on screen
      onStartShouldSetPanResponder:         (e) => e.nativeEvent.touches.length === 2,
      onMoveShouldSetPanResponder:          (e) => e.nativeEvent.touches.length === 2,
      onStartShouldSetPanResponderCapture:  (e) => e.nativeEvent.touches.length === 2,
      onMoveShouldSetPanResponderCapture:   (e) => e.nativeEvent.touches.length === 2,

      onPanResponderMove: (e) => {
        const touches = e.nativeEvent.touches;
        if (touches.length !== 2) return;

        const dx = touches[0].pageX - touches[1].pageX;
        const dy = touches[0].pageY - touches[1].pageY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (lastDistanceRef.current !== null) {
          const delta  = distance - lastDistanceRef.current;
          // Sensitivity: 0.004 per pixel of finger spread
          const newZoom = Math.min(1, Math.max(0, zoomRef.current + delta * 0.004));
          zoomRef.current = newZoom;
          setZoom(newZoom);
        }
        lastDistanceRef.current = distance;
      },

      onPanResponderRelease:   () => { lastDistanceRef.current = null; },
      onPanResponderTerminate: () => { lastDistanceRef.current = null; },
    })
  ).current;

  // ── Permission not yet determined ──────────────────────────────────────────
  if (!permission) {
    return <View style={{ flex: 1, backgroundColor: "#000" }} />;
  }

  // ── Permission denied ──────────────────────────────────────────────────────
  if (!permission.granted) {
    return (
      <SafeAreaView
        style={{
          flex: 1, backgroundColor: "#111",
          alignItems: "center", justifyContent: "center",
          paddingHorizontal: 32,
        }}
      >
        <Text style={{ color: "#fff", fontSize: 48, marginBottom: 24 }}>📷</Text>
        <Text style={{ color: "#fff", fontSize: 20, fontWeight: "bold", textAlign: "center" }}>
          Camera Access Needed
        </Text>
        <Text style={{ color: "#9ca3af", textAlign: "center", marginTop: 12, marginBottom: 32 }}>
          We need camera access to photograph your plant leaves for disease detection.
        </Text>
        <TouchableOpacity
          onPress={requestPermission}
          style={{
            backgroundColor: "#08AF4E",
            paddingHorizontal: 32, paddingVertical: 16,
            borderRadius: 50,
          }}
        >
          <Text style={{ color: "#fff", fontWeight: "bold", fontSize: 16 }}>
            Grant Permission
          </Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  // ── Capture ────────────────────────────────────────────────────────────────
  async function handleCapture() {
    if (!cameraRef.current || isTaking) return;
    setIsTaking(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ quality: 0.8 });
      setCapturedImage(photo.uri);
    } catch (err) {
      console.error("Capture failed:", err);
    } finally {
      setIsTaking(false);
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <View style={{ flex: 1, backgroundColor: "#000" }} {...pinchResponder.panHandlers}>
      <CameraView ref={cameraRef} style={{ flex: 1 }} facing="back" zoom={zoom}>

        {/* Spacer */}
        <View style={{ flex: 1 }} />

        {/* ── Bottom controls ── */}
        <SafeAreaView>
          <View
            style={{
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-between",
              paddingHorizontal: 48,
              paddingBottom: 32,
              paddingTop: 8,
            }}
          >
            {/* Last captured photo thumbnail — tap to proceed */}
            <TouchableOpacity
              onPress={() => capturedImageUri && router.push("/preview")}
              activeOpacity={capturedImageUri ? 0.7 : 1}
            >
              {capturedImageUri ? (
                <Image
                  source={{ uri: capturedImageUri }}
                  style={{
                    width: 56, height: 56, borderRadius: 10,
                    borderWidth: 2, borderColor: "#fff",
                  }}
                  resizeMode="cover"
                />
              ) : (
                <View
                  style={{
                    width: 56, height: 56, borderRadius: 10,
                    borderWidth: 2, borderColor: "#ffffff44",
                    backgroundColor: "#ffffff11",
                  }}
                />
              )}
            </TouchableOpacity>

            {/* Shutter button */}
            <TouchableOpacity
              onPress={handleCapture}
              disabled={isTaking}
              style={{
                width: 72, height: 72, borderRadius: 36,
                borderWidth: 4, borderColor: "#fff",
                alignItems: "center", justifyContent: "center",
              }}
            >
              <View
                style={{
                  width: 56, height: 56, borderRadius: 28,
                  backgroundColor: isTaking ? "#ffffff88" : "#ffffff",
                }}
              />
            </TouchableOpacity>

            {/* Retake — clears the captured image */}
            <TouchableOpacity
              onPress={resetSession}
              style={{
                width: 56, height: 56,
                alignItems: "center", justifyContent: "center",
              }}
            >
              <RotateCcw size={32} color="#ffffff" />
            </TouchableOpacity>
          </View>
        </SafeAreaView>

      </CameraView>
    </View>
  );
}
