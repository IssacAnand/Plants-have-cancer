// app/(tabs)/scan.jsx — Diagnosis Screen

import { useEffect } from "react";
import {
  View,
  Text,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  Image,
  Alert,
} from "react-native";
import { useRouter } from "expo-router";
import * as ImagePicker from "expo-image-picker";

import {
  useFonts,
  Poppins_600SemiBold,
  Poppins_400Regular,
} from "@expo-google-fonts/poppins";
import { Camera, Images, SlidersHorizontal, Leaf } from "lucide-react-native";

import usePlantStore from "../../store/usePlantStore";

const GREEN       = "#08AF4E";
const TITLE_COLOR = "#561111";

// ── Mock data shown when there are no real scans yet ──────────────────────────
const MOCK_IMAGE = require("../../assets/sunflower.jpg");

const MOCK_SCANS = [
  {
    id:          "mock-1",
    plantName:   "Sunflower",
    imageSource: MOCK_IMAGE,
    disease:     "Light Blight with Sun Spots",
    confidence:  97,
    date:        "2025-01-28T10:58:00.000Z",
  },
  {
    id:          "mock-2",
    plantName:   "Sunflower",
    imageSource: MOCK_IMAGE,
    disease:     "Light Blight with Sun Spots",
    confidence:  97,
    date:        "2025-01-28T10:58:00.000Z",
  },
];

// ─────────────────────────────────────────────────────────────────────────────

export default function DiagnosisScreen() {
  const router = useRouter();

  const [fontsLoaded] = useFonts({ Poppins_600SemiBold, Poppins_400Regular });

  const recentScans      = usePlantStore((s) => s.recentScans);
  const loadRecentScans  = usePlantStore((s) => s.loadRecentScans);
  const setSelectedScan  = usePlantStore((s) => s.setSelectedScan);
  const setCapturedImage = usePlantStore((s) => s.setCapturedImage);

  useEffect(() => {
    loadRecentScans();
  }, []);

  const scans = recentScans.length > 0 ? recentScans : MOCK_SCANS;

  async function handleGalleryPick() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") {
      Alert.alert(
        "Permission needed",
        "Please allow access to your photo library to upload an image."
      );
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsEditing: false,
      quality: 0.8,
    });
    if (!result.canceled) {
      setCapturedImage(result.assets[0].uri);
      router.push("/preview");
    }
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#ffffff" }}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingBottom: 40 }}
      >
        {/* ── Title ── */}
        <View style={{ paddingHorizontal: 20, paddingTop: 24, paddingBottom: 4 }}>
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
              fontSize: 32,
              color: TITLE_COLOR,
            }}
          >
            Diagnosis
          </Text>
        </View>

        {/* ── Hero ── */}
        <View style={{ alignItems: "center", paddingVertical: 28, paddingHorizontal: 20 }}>
          {/* TODO: replace with real asset when provided */}
          <PlantIconPlaceholder />

          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
              fontSize: 20,
              color: GREEN,
              textAlign: "center",
              marginTop: 18,
              lineHeight: 30,
            }}
          >
            Check the health{"\n"}of your plants!
          </Text>

          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
              fontSize: 14,
              color: "#561111",
              textAlign: "center",
              marginTop: 4,
            }}
          >
            take/upload a photo and get a diagnosis
          </Text>
        </View>

        {/* ── Action buttons — gallery (left) + scan (right) ── */}
        <View style={{ paddingHorizontal: 20, marginBottom: 32, flexDirection: "row", gap: 12 }}>
          {/* Gallery button */}
          <TouchableOpacity
            onPress={handleGalleryPick}
            activeOpacity={0.85}
            style={{
              flex: 1,
              backgroundColor: "#f3f4f6",
              borderRadius: 10,
              paddingVertical: 16,
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "center",
              gap: 10,
            }}
          >
            <Images size={24} color="#374151" />
            <Text
              style={{
                fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
                color: "#374151",
                fontSize: 20,
              }}
            >
              Gallery
            </Text>
          </TouchableOpacity>

          {/* Scan button */}
          <TouchableOpacity
            onPress={() => router.push("/camera")}
            activeOpacity={0.85}
            style={{
              flex: 1,
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
                fontSize: 20,
              }}
            >
              Scan
            </Text>
            <Camera size={24} color="#ffffff" />
          </TouchableOpacity>
        </View>

        {/* ── Scan History header ── */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            justifyContent: "space-between",
            paddingHorizontal: 20,
            marginBottom: 14,
          }}
        >
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
              fontSize: 22,
              color: TITLE_COLOR,
            }}
          >
            Scan History
          </Text>
          <SlidersHorizontal size={22} color="#9ca3af" />
        </View>

        {/* ── History cards ── */}
        {scans.map((scan, idx) => (
          <HistoryCard
            key={scan.id ?? idx}
            scan={scan}
            fontsLoaded={fontsLoaded}
            onViewDetail={() => {
              setSelectedScan(scan);
              router.push("/results");
            }}
          />
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Placeholder — swap this out once the real asset is provided
// ─────────────────────────────────────────────────────────────────────────────
function PlantIconPlaceholder() {
  return (
    <View
      style={{
        width: 96,
        height: 96,
        borderRadius: 48,
        backgroundColor: "#FEF9C3",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Leaf size={48} color="#E8B800" />
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Single history card
// ─────────────────────────────────────────────────────────────────────────────
function HistoryCard({ scan, fontsLoaded, onViewDetail }) {
  const date = new Date(scan.date);

  // "28th January 2025, 4:28PM GST+8"
  const day    = date.getDate();
  const suffix = ordinal(day);
  const month  = date.toLocaleString("en-US", { month: "long" });
  const year   = date.getFullYear();
  const time   = date
    .toLocaleString("en-US", { hour: "numeric", minute: "2-digit", hour12: true })
    .replace(" ", "");                        // "4:28PM"
  const tz     = date
    .toLocaleString("en-US", { timeZoneName: "short" })
    .split(", ")[1]?.replace("GMT", "GST") ?? "";

  const formattedDate = `${day}${suffix} ${month} ${year}, ${time} ${tz}`.trim();

  return (
    <View
      style={{
        flexDirection:   "row",
        backgroundColor: "#ffffff",
        borderRadius:    16,
        marginHorizontal: 16,
        marginBottom:    12,
        padding:         12,
        shadowColor:     "#000",
        shadowOpacity:   0.06,
        shadowRadius:    8,
        shadowOffset:    { width: 0, height: 2 },
        elevation:       2,
        borderWidth:     1,
        borderColor:     "#f0f0f0",
      }}
    >
      {/* Thumbnail — imageSource for local requires, imageUri for camera URIs */}
      {scan.imageSource || scan.imageUri ? (
        <Image
          source={scan.imageSource ?? { uri: scan.imageUri }}
          style={{ width: 80, height: 80, borderRadius: 10 }}
          resizeMode="cover"
        />
      ) : (
        <View
          style={{
            width: 80, height: 80, borderRadius: 10,
            backgroundColor: "#f3f4f6",
          }}
        />
      )}

      {/* Text content */}
      <View style={{ flex: 1, marginLeft: 12 }}>
        <Text
          style={{
            fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
            fontSize:   16,
            color:      "#1f2937",
          }}
        >
          {scan.plantName ?? "Unknown Plant"}
        </Text>

        <Text
          style={{
            fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
            fontSize:   11,
            color:      "#9ca3af",
            marginBottom: 4,
          }}
        >
          {formattedDate}
        </Text>

        <Text
          style={{
            fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
            fontSize:   12,
            color:      "#374151",
          }}
        >
          <Text style={{ fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined }}>
            Disease Predicted:{" "}
          </Text>
          {scan.disease}
        </Text>

        <Text
          style={{
            fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
            fontSize:   12,
            color:      "#374151",
            marginTop:  2,
          }}
        >
          <Text style={{ fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined }}>
            Confidence:{" "}
          </Text>
          {scan.confidence}%
        </Text>

        {/* View in Detail */}
        <TouchableOpacity
          onPress={onViewDetail}
          style={{
            marginTop:         8,
            borderWidth:       1,
            borderColor:       "#9ca3af",
            borderRadius:      6,
            paddingVertical:   3,
            paddingHorizontal: 10,
            alignSelf:         "flex-start",
          }}
          activeOpacity={0.7}
        >
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
              fontSize:   11,
              color:      "#4b5563",
            }}
          >
            View in Detail
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function ordinal(n) {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return s[(v - 20) % 10] ?? s[v] ?? s[0];
}
