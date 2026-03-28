// app/(tabs)/profile.jsx  — Profile Screen
//
// A simple profile/info screen. In a real app you'd connect this to
// an authentication system (e.g. Supabase, Firebase Auth).
// For now it shows app info and a model status indicator.

import { View, Text, SafeAreaView, ScrollView } from "react-native";
import usePlantStore from "../../store/usePlantStore";

// ── A small settings row component (local to this file) ──────────────────────
function InfoRow({ label, value, valueColor = "text-gray-600" }) {
  return (
    <View className="flex-row items-center justify-between py-3.5 border-b border-gray-50">
      <Text className="text-gray-500 text-sm">{label}</Text>
      <Text className={`text-sm font-medium ${valueColor}`}>{value}</Text>
    </View>
  );
}

export default function ProfileScreen() {
  const isModelLoaded = usePlantStore((s) => s.isModelLoaded);
  const recentScans   = usePlantStore((s) => s.recentScans);

  return (
    <SafeAreaView className="flex-1 bg-surface">
      {/* ── Header ── */}
      <View className="px-6 py-4 border-b border-gray-100 bg-white">
        <Text className="text-xl font-bold text-gray-800">Profile</Text>
      </View>

      <ScrollView className="flex-1" contentContainerStyle={{ padding: 20 }}>
        {/* ── Avatar & name ── */}
        <View className="items-center py-6 bg-white rounded-2xl shadow-sm mb-4">
          <View className="w-20 h-20 rounded-full bg-green-100 items-center justify-center mb-3">
            <Text style={{ fontSize: 44 }}>👨‍🌾</Text>
          </View>
          <Text className="text-gray-800 text-lg font-bold">Farmer User</Text>
          <Text className="text-gray-400 text-sm mt-1">Plant Health Detector</Text>
        </View>

        {/* ── Stats ── */}
        <View className="flex-row gap-x-3 mb-4">
          <View className="flex-1 bg-white rounded-2xl p-4 shadow-sm items-center">
            <Text className="text-3xl font-bold text-primary">{recentScans.length}</Text>
            <Text className="text-gray-400 text-xs mt-1 text-center">Total{"\n"}Scans</Text>
          </View>
          <View className="flex-1 bg-white rounded-2xl p-4 shadow-sm items-center">
            <Text className="text-3xl font-bold text-primary">5</Text>
            <Text className="text-gray-400 text-xs mt-1 text-center">Diseases{"\n"}Detectable</Text>
          </View>
          <View className="flex-1 bg-white rounded-2xl p-4 shadow-sm items-center">
            <Text className="text-3xl font-bold text-primary">100%</Text>
            <Text className="text-gray-400 text-xs mt-1 text-center">Offline{"\n"}Capable</Text>
          </View>
        </View>

        {/* ── App info ── */}
        <View className="bg-white rounded-2xl px-4 shadow-sm mb-4">
          <Text className="text-gray-400 text-xs uppercase tracking-widest pt-4 pb-2 font-semibold">
            App Info
          </Text>
          <InfoRow label="App Version"    value="1.0.0" />
          <InfoRow label="AI Model"       value="PyTorch Lite (.ptl)" />
          <InfoRow
            label="Model Status"
            value={isModelLoaded ? "✅ Loaded" : "⏳ Loading…"}
            valueColor={isModelLoaded ? "text-green-600" : "text-amber-500"}
          />
          <InfoRow label="Storage"        value="AsyncStorage (local)" />
          <InfoRow label="History Limit"  value="5 scans" />
        </View>

        {/* ── Offline badge ── */}
        <View className="bg-green-50 border border-green-100 rounded-2xl p-4 flex-row items-center">
          <Text className="text-3xl mr-3">🛡</Text>
          <View className="flex-1">
            <Text className="text-green-700 font-semibold">Offline Mode Active</Text>
            <Text className="text-green-600 text-sm mt-0.5">
              All analysis happens on-device. No data is sent to any server.
            </Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
