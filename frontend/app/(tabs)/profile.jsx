// app/(tabs)/profile.jsx — My Profile Screen

import { View, Text, ScrollView, TouchableOpacity, Alert } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import {
  ChevronRight,
  Clock,
  Sprout,
  FileSearch,
  Cpu,
  Package,
  HardDrive,
  Trash2,
  HelpCircle,
  Info,
  Leaf,
} from "lucide-react-native";
import {
  useFonts,
  Poppins_600SemiBold,
  Poppins_400Regular,
} from "@expo-google-fonts/poppins";

import usePlantStore from "../../store/usePlantStore";
import { clearAllScans } from "../../utils/storage";

const GREEN       = "#08AF4E";
const TITLE_COLOR = "#561111";

// ── ProfileRow ────────────────────────────────────────────────────────────────
function ProfileRow({ icon: Icon, iconBg, iconColor, label, labelColor, rightLabel, onPress, isLast }) {
  const Wrapper = onPress ? TouchableOpacity : View;

  return (
    <Wrapper
      onPress={onPress}
      activeOpacity={0.6}
      style={{
        flexDirection: "row",
        alignItems: "center",
        paddingVertical: 14,
        borderBottomWidth: isLast ? 0 : 1,
        borderBottomColor: "#f3f4f6",
      }}
    >
      <View
        style={{
          width: 36,
          height: 36,
          borderRadius: 10,
          backgroundColor: iconBg,
          alignItems: "center",
          justifyContent: "center",
          marginRight: 14,
        }}
      >
        <Icon size={18} color={iconColor} strokeWidth={1.8} />
      </View>

      <Text
        style={{
          flex: 1,
          fontSize: 15,
          color: labelColor ?? "#374151",
          fontFamily: "Poppins_400Regular",
        }}
      >
        {label}
      </Text>

      {rightLabel ? (
        <Text style={{ fontSize: 13, color: "#9ca3af", fontFamily: "Poppins_400Regular" }}>
          {rightLabel}
        </Text>
      ) : onPress ? (
        <ChevronRight size={18} color="#d1d5db" strokeWidth={2} />
      ) : null}
    </Wrapper>
  );
}

// ── SectionLabel ──────────────────────────────────────────────────────────────
function SectionLabel({ title }) {
  return (
    <Text
      style={{
        fontSize: 13,
        color: TITLE_COLOR,
        fontFamily: "Poppins_600SemiBold",
        marginBottom: 10,
        marginTop: 6,
      }}
    >
      {title}
    </Text>
  );
}

// ── Card ──────────────────────────────────────────────────────────────────────
function Card({ children, style }) {
  return (
    <View
      style={[
        {
          backgroundColor: "#ffffff",
          borderRadius: 16,
          paddingHorizontal: 16,
          borderWidth: 1,
          borderColor: "#f0f0f0",
          shadowColor: "#000",
          shadowOpacity: 0.06,
          shadowRadius: 8,
          shadowOffset: { width: 0, height: 2 },
          elevation: 2,
          marginBottom: 20,
        },
        style,
      ]}
    >
      {children}
    </View>
  );
}

// ── StatCell ──────────────────────────────────────────────────────────────────
function StatCell({ value, label, fontsLoaded }) {
  return (
    <View style={{ flex: 1, alignItems: "center" }}>
      <Text
        style={{
          fontSize: 26,
          fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
          color: GREEN,
        }}
      >
        {value}
      </Text>
      <Text
        style={{
          fontSize: 11,
          color: "#9ca3af",
          textAlign: "center",
          marginTop: 2,
          fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
        }}
      >
        {label}
      </Text>
    </View>
  );
}

// ── Screen ────────────────────────────────────────────────────────────────────
export default function ProfileScreen() {
  const router          = useRouter();
  const isModelLoaded   = usePlantStore((s) => s.isModelLoaded);
  const recentScans     = usePlantStore((s) => s.recentScans);
  const plants          = usePlantStore((s) => s.plants);
  const setSelectedScan = usePlantStore((s) => s.setSelectedScan);
  const loadRecentScans = usePlantStore((s) => s.loadRecentScans);

  const [fontsLoaded] = useFonts({ Poppins_600SemiBold, Poppins_400Regular });

  const uniqueDiseases = new Set(
    recentScans.filter((s) => s.disease).map((s) => s.disease)
  ).size;

  function handleLastScan() {
    if (!recentScans.length) {
      Alert.alert("No Scans Yet", "Complete a scan first to view results here.");
      return;
    }
    setSelectedScan(recentScans[0]);
    router.push("/results");
  }

  function handleClearHistory() {
    Alert.alert(
      "Clear Scan History",
      "This will permanently delete all saved scans. This cannot be undone.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Clear",
          style: "destructive",
          onPress: async () => {
            await clearAllScans();
            await loadRecentScans();
          },
        },
      ]
    );
  }

  function handleHowToUse() {
    Alert.alert(
      "How to Use Farmpals",
      "1. Tap the Scan tab and press the camera button.\n\n" +
        "2. Take a clear photo of the affected leaf.\n\n" +
        "3. Optionally describe the symptoms in the text box.\n\n" +
        "4. Wait while the AI analyses your plant.\n\n" +
        "5. View the diagnosis and recommended treatment.",
      [{ text: "Got it" }]
    );
  }

  function handleAbout() {
    Alert.alert(
      "About Farmpals",
      "Farmpals is an offline-first plant health detector powered by a multimodal AI pipeline (vision + text).\n\n" +
        "It can identify 89 plant disease classes using your phone's camera — no internet required.\n\n" +
        "Version 1.0.0",
      [{ text: "Close" }]
    );
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#ffffff" }}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 40 }}
      >
        {/* ── Page title ── */}
        <View style={{ paddingTop: 24, paddingBottom: 20 }}>
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
              fontSize: 32,
              color: TITLE_COLOR,
              lineHeight: 40,
            }}
          >
            My Profile
          </Text>
        </View>

        {/* ── Avatar card ── */}
        <Card style={{ paddingVertical: 24, alignItems: "center", marginBottom: 20 }}>
          <View
            style={{
              width: 72,
              height: 72,
              borderRadius: 36,
              backgroundColor: GREEN,
              alignItems: "center",
              justifyContent: "center",
              marginBottom: 12,
            }}
          >
            <Leaf size={34} color="#fff" strokeWidth={1.8} />
          </View>
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
              fontSize: 17,
              color: "#1f2937",
            }}
          >
            Farmpals User
          </Text>
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
              fontSize: 13,
              color: "#9ca3af",
              marginTop: 3,
            }}
          >
            Plant Health Detector
          </Text>
        </Card>

        {/* ── Stats strip ── */}
        <View
          style={{
            flexDirection: "row",
            backgroundColor: "#f3f4f6",
            borderRadius: 16,
            paddingVertical: 16,
            marginBottom: 28,
          }}
        >
          <StatCell value={recentScans.length} label={"Total\nScans"} fontsLoaded={fontsLoaded} />
          <View style={{ width: 1, backgroundColor: "#e5e7eb" }} />
          <StatCell value={uniqueDiseases} label={"Diseases\nFound"} fontsLoaded={fontsLoaded} />
          <View style={{ width: 1, backgroundColor: "#e5e7eb" }} />
          <StatCell value={plants.length} label={"Plants on\nFarm"} fontsLoaded={fontsLoaded} />
        </View>

        {/* ── My Activity ── */}
        <SectionLabel title="My Activity" />
        <Card>
          <ProfileRow
            icon={Clock}
            iconBg="#eff6ff"
            iconColor="#3b82f6"
            label="Scan History"
            onPress={() => router.push("/(tabs)/scan")}
          />
          <ProfileRow
            icon={Sprout}
            iconBg="#f0fdf4"
            iconColor={GREEN}
            label="My Farm"
            onPress={() => router.push("/(tabs)")}
          />
          <ProfileRow
            icon={FileSearch}
            iconBg="#fdf4ff"
            iconColor="#a855f7"
            label="Last Scan Result"
            onPress={handleLastScan}
            isLast
          />
        </Card>

        {/* ── App Info ── */}
        <SectionLabel title="App Info" />
        <Card>
          <ProfileRow
            icon={Cpu}
            iconBg="#fff7ed"
            iconColor="#f97316"
            label="Model Status"
            rightLabel={isModelLoaded ? "Loaded" : "Loading…"}
          />
          <ProfileRow
            icon={Package}
            iconBg="#f8fafc"
            iconColor="#64748b"
            label="App Version"
            rightLabel="1.0.0"
          />
          <ProfileRow
            icon={HardDrive}
            iconBg="#f0fdf4"
            iconColor="#16a34a"
            label="Scan Storage"
            rightLabel={`${recentScans.length} / 5`}
            isLast
          />
        </Card>

        {/* ── Data ── */}
        <SectionLabel title="Data" />
        <Card>
          <ProfileRow
            icon={Trash2}
            iconBg="#fef2f2"
            iconColor="#ef4444"
            label="Clear Scan History"
            labelColor="#ef4444"
            onPress={handleClearHistory}
            isLast
          />
        </Card>

        {/* ── Help ── */}
        <SectionLabel title="Help" />
        <Card>
          <ProfileRow
            icon={HelpCircle}
            iconBg="#fffbeb"
            iconColor="#f59e0b"
            label="How to Use"
            onPress={handleHowToUse}
          />
          <ProfileRow
            icon={Info}
            iconBg="#eff6ff"
            iconColor="#3b82f6"
            label="About"
            onPress={handleAbout}
            isLast
          />
        </Card>

        {/* ── Offline status pill ── */}
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            backgroundColor: "#f3f4f6",
            borderRadius: 50,
            paddingVertical: 10,
            paddingHorizontal: 16,
          }}
        >
          <View
            style={{
              width: 8,
              height: 8,
              borderRadius: 4,
              backgroundColor: isModelLoaded ? GREEN : "#f59e0b",
              marginRight: 10,
            }}
          />
          <Text
            style={{
              flex: 1,
              fontSize: 13,
              color: "#374151",
              fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
            }}
          >
            {isModelLoaded
              ? "AI model loaded — all analysis runs on-device."
              : "AI model loading, please wait…"}
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
