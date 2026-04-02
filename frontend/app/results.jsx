// app/results.jsx — Diagnosis Results Screen

import { useState } from "react";
import {
  View,
  Text,
  Image,
  ScrollView,
  TouchableOpacity,
  SafeAreaView,
} from "react-native";
import { useRouter } from "expo-router";
import {
  useFonts,
  Poppins_600SemiBold,
  Poppins_400Regular,
} from "@expo-google-fonts/poppins";
import { Plus } from "lucide-react-native";

import usePlantStore from "../store/usePlantStore";

const GREEN       = "#08AF4E";
const TITLE_COLOR = "#561111";
const TABS        = ["Diagnosis", "Explainability View", "Treatment"];

const LOREM =
  "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

export default function ResultsScreen() {
  const router = useRouter();

  const [activeTab,  setActiveTab]  = useState("Diagnosis");
  const [fontsLoaded] = useFonts({ Poppins_600SemiBold, Poppins_400Regular });

  const capturedImageUri = usePlantStore((s) => s.capturedImageUri);
  const analysisResult   = usePlantStore((s) => s.analysisResult);
  const addPlant         = usePlantStore((s) => s.addPlant);
  const resetSession     = usePlantStore((s) => s.resetSession);

  // Use real result when available, placeholder otherwise
  const result = analysisResult ?? {
    plantName:  "Monsterra",
    disease:    "Light Blight with Sun Spots",
    confidence: 97,
  };

  function handleAddToFarm() {
    addPlant({
      id:          Date.now().toString(),
      name:        result.plantName ?? "Unknown Plant",
      status:      "diseased",
      isFavourite: false,
      image:       capturedImageUri ? { uri: capturedImageUri } : null,
      bgColor:     "#fde68a",
    });
    resetSession();
    router.replace("/(tabs)/index");
  }

  return (
    <View style={{ flex: 1, backgroundColor: "#ffffff" }}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingBottom: 100 }}
        bounces={false}
      >
        {/* ── Full-width photo, edge to edge ── */}
        {capturedImageUri ? (
          <Image
            source={{ uri: capturedImageUri }}
            style={{ width: "100%", aspectRatio: 9 / 10 }}
            resizeMode="cover"
          />
        ) : (
          <View
            style={{
              width: "100%", aspectRatio: 9 / 10,
              backgroundColor: "#d1fae5",
              alignItems: "center", justifyContent: "center",
            }}
          >
            <Text style={{ fontSize: 64 }}>🌿</Text>
          </View>
        )}

        {/* ── Tabs — mirrors the category strip in index.jsx ── */}
        <View
          style={{
            flexDirection: "row",
            borderBottomWidth: 0.5,
            borderBottomColor: "#e5e7eb",
          }}
        >
          {TABS.map((tab) => {
            const active = activeTab === tab;
            return (
              <TouchableOpacity
                key={tab}
                onPress={() => setActiveTab(tab)}
                style={{ flex: 1, alignItems: "center", justifyContent: "center" }}
                activeOpacity={0.7}
              >
                {/* Underline lives on the inner View so it only spans the text */}
                <View
                  style={{
                    paddingHorizontal: active ? 4 : 0,
                    paddingBottom: 6,
                    paddingTop: 12,
                    borderBottomWidth: active ? 2 : 0,
                    borderBottomColor: GREEN,
                  }}
                >
                  <Text
                    style={{
                      fontFamily: fontsLoaded
                        ? active ? "Poppins_600SemiBold" : "Poppins_400Regular"
                        : undefined,
                      fontSize: 11,
                      color: active ? GREEN : "#9ca3af",
                    }}
                  >
                    {tab}
                  </Text>
                </View>
              </TouchableOpacity>
            );
          })}
        </View>

        {/* ── Tab content ── */}
        <View style={{ padding: 15 }}>

          {activeTab === "Diagnosis" && (
            <>
              <Text
                style={{
                  fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
                  fontSize: 14,
                  color: TITLE_COLOR,
                  marginBottom: 8,
                }}
              >
                {result.plantName ?? "Unknown Plant"}
              </Text>

              <Text
                style={{
                  fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
                  fontSize: 15,
                  color: "#1f2937",
                  marginBottom: 4,
                }}
              >
                Predicted Disease: {result.disease}
              </Text>

              <Text
                style={{
                  fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
                  fontSize: 15,
                  color: GREEN,
                  marginBottom: 16,
                }}
              >
                Confidence: {result.confidence}%
              </Text>

              <Text
                style={{
                  fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
                  fontSize: 13,
                  color: "#374151",
                  lineHeight: 22,
                }}
              >
                {LOREM}
              </Text>
            </>
          )}

          {activeTab === "Explainability View" && (
            analysisResult?.heatmapUri ? (
              <>
                <Text
                  style={{
                    fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
                    fontSize: 13,
                    color: "#374151",
                    lineHeight: 20,
                    marginBottom: 12,
                  }}
                >
                  Areas highlighted in red had the highest neural activation —
                  these are the regions the model focused on when identifying
                  the disease. Blue areas had low activation.
                </Text>
                <Image
                  source={{ uri: analysisResult.heatmapUri }}
                  style={{
                    width: "100%",
                    aspectRatio: 1,
                    borderRadius: 12,
                  }}
                  resizeMode="cover"
                />
              </>
            ) : (
              <Text
                style={{
                  fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
                  fontSize: 13,
                  color: "#9ca3af",
                  textAlign: "center",
                  marginTop: 48,
                }}
              >
                Heatmap not available for this scan.
              </Text>
            )
          )}

          {activeTab === "Treatment" && (
            <Text
              style={{
                fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
                fontSize: 13,
                color: "#374151",
                lineHeight: 22,
              }}
            >
              {analysisResult?.treatment
                ?? "Treatment recommendations not available."}
            </Text>
          )}

        </View>
      </ScrollView>

      {/* ── Add to My Farm — fixed at bottom ── */}
      <SafeAreaView
        style={{
          position: "absolute", bottom: 0, left: 0, right: 0,
          backgroundColor: "#ffffff",
          paddingHorizontal: 40,
          paddingTop: 10,
          paddingBottom: 10,
        }}
      >
        <TouchableOpacity
          onPress={handleAddToFarm}
          activeOpacity={0.85}
          style={{
            backgroundColor: GREEN,
            borderRadius: 10,
            paddingVertical: 16,
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "center",
            gap: 10,
          }}
        >
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
              color: "#ffffff",
              fontSize: 17,
            }}
          >
            Add to My Farm
          </Text>
          <View
            style={{
              width: 26, height: 26, borderRadius: 13,
              borderWidth: 2, borderColor: "#ffffff",
              alignItems: "center", justifyContent: "center",
            }}
          >
            <Plus size={14} color="#ffffff" />
          </View>
        </TouchableOpacity>
      </SafeAreaView>
    </View>
  );
}
