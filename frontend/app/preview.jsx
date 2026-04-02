// app/preview.jsx — Diagnosis Preview Screen

import { useState } from "react";
import {
  View,
  Text,
  Image,
  SafeAreaView,
  TextInput,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import { useRouter } from "expo-router";
import {
  useFonts,
  Poppins_600SemiBold,
  Poppins_400Regular,
} from "@expo-google-fonts/poppins";

import usePlantStore from "../store/usePlantStore";

const GREEN       = "#08AF4E";
const TITLE_COLOR = "#561111";

export default function PreviewScreen() {
  const router = useRouter();

  const [fontsLoaded] = useFonts({ Poppins_600SemiBold, Poppins_400Regular });

  // Symptom text — will be passed to the model once integration is complete
  const [symptomText, setSymptomText] = useState("");

  const capturedImageUri = usePlantStore((s) => s.capturedImageUri);
  const setCapturedText  = usePlantStore((s) => s.setCapturedText);
  const isModelLoaded    = usePlantStore((s) => s.isModelLoaded);

  function handleGetDiagnosis() {
    if (!capturedImageUri || !isModelLoaded) return;
    setCapturedText(symptomText);
    router.replace("/processing");
  }

  const canSubmit = Boolean(capturedImageUri) && isModelLoaded;

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <SafeAreaView style={{ flex: 1, backgroundColor: "#ffffff" }}>
        <ScrollView
          contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 24 }}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* ── Title ── */}
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
              fontSize: 32,
              color: TITLE_COLOR,
              paddingTop: 24,
              marginBottom: 16,
            }}
          >
            Diagnosis
          </Text>

          {/* ── Captured photo ── */}
          {capturedImageUri ? (
            <Image
              source={{ uri: capturedImageUri }}
              style={{
                width: "100%",
                aspectRatio: 8 / 10,
                borderRadius: 16,
                backgroundColor: "#f3f4f6",
              }}
              resizeMode="cover"
            />
          ) : (
            <View
              style={{
                width: "100%",
                aspectRatio: 6 / 10,
                borderRadius: 16,
                backgroundColor: "#f3f4f6",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Text style={{ fontSize: 40 }}>🌿</Text>
              <Text
                style={{
                  fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
                  color: "#9ca3af",
                  marginTop: 8,
                }}
              >
                No image captured
              </Text>
            </View>
          )}

          {/* ── Symptom prompt ── */}
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
              fontSize: 17,
              color: TITLE_COLOR,
              marginTop: 20,
              marginBottom: 10,
            }}
          >
            What's bothering your plant today?
          </Text>

          {/* ── Symptom text input ── */}
          <TextInput
            style={{
              backgroundColor: "#f3f4f6",
              borderRadius: 12,
              padding: 14,
              fontSize: 14,
              fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
              color: "#374151",
              minHeight: 120,
              textAlignVertical: "top",
            }}
            placeholder="Describe what you are noticing..."
            placeholderTextColor="#9ca3af"
            multiline
            value={symptomText}
            onChangeText={setSymptomText}
          />
        </ScrollView>

        {/* ── Get Diagnosis button — pinned to bottom ── */}
        <View style={{ paddingHorizontal: 20, paddingBottom: 24, paddingTop: 8 }}>
          <TouchableOpacity
            onPress={handleGetDiagnosis}
            activeOpacity={canSubmit ? 0.85 : 1}
            disabled={!canSubmit}
            style={{
              backgroundColor: canSubmit ? GREEN : "#9ca3af",
              borderRadius: 10,
              paddingVertical: 16,
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Text
              style={{
                fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
                color: "#ffffff",
                fontSize: 18,
              }}
            >
              {!capturedImageUri
                ? "Capture Image First"
                : isModelLoaded
                  ? "Get Diagnosis"
                  : "Loading Model..."}
            </Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </KeyboardAvoidingView>
  );
}
