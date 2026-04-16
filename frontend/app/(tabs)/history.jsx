// app/(tabs)/history.jsx  — Scan History Screen
//
// ─────────────────────────────────────────────────────────────────────────────
// CONCEPT: useFocusEffect — run code every time a tab becomes active
// ─────────────────────────────────────────────────────────────────────────────
//
// useEffect with [] only runs ONCE when the component first mounts.
// But with tab navigation, components mount ONCE and stay mounted.
// If the user scans a leaf and then taps the History tab, the list
// won't update because the component already mounted before the scan.
//
// useFocusEffect() fires every time the screen comes into focus —
// perfect for refreshing data when switching tabs.
//
// Pattern:
//   useFocusEffect(
//     useCallback(() => { doWork(); }, [deps])
//   );
// The useCallback is required by useFocusEffect to avoid infinite loops.
// ─────────────────────────────────────────────────────────────────────────────

import { useCallback } from "react";
import {
  View, Text, FlatList, SafeAreaView,
  TouchableOpacity, Alert
} from "react-native";
import { useFocusEffect, useRouter } from "expo-router";

import usePlantStore from "../../store/usePlantStore";
import ScanCard from "../../components/ScanCard";
import { clearAllScans } from "../../utils/storage";

export default function HistoryScreen() {
  const router          = useRouter();
  const recentScans     = usePlantStore((s) => s.recentScans);
  const loadRecentScans = usePlantStore((s) => s.loadRecentScans);
  const setSelectedScan = usePlantStore((s) => s.setSelectedScan);

  // Refresh the list every time this tab gains focus
  useFocusEffect(
    useCallback(() => {
      loadRecentScans();
    }, [])
  );

  async function handleClearHistory() {
    Alert.alert(
      "Clear History",
      "This will delete all 5 recent scans. Are you sure?",
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

  // ── Empty state ────────────────────────────────────────────────────────────
  function EmptyState() {
    return (
      <View className="flex-1 items-center justify-center py-20">
        <Text className="text-5xl mb-4">🌱</Text>
        <Text className="text-gray-500 text-lg font-semibold">No scans yet</Text>
        <Text className="text-gray-400 text-sm mt-2 text-center px-8">
          Your last 5 scans will appear here after you photograph a leaf.
        </Text>
      </View>
    );
  }

  return (
    <SafeAreaView className="flex-1 bg-surface">
      {/* ── Header ── */}
      <View className="flex-row items-center justify-between px-6 py-4 border-b border-gray-100 bg-white">
        <Text className="text-xl font-bold text-gray-800">Recent Scans</Text>
        {recentScans.length > 0 && (
          <TouchableOpacity onPress={handleClearHistory}>
            <Text className="text-red-400 text-sm font-medium">Clear</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* ── Subtitle ── */}
      <View className="px-6 pt-4 pb-2">
        <Text className="text-gray-400 text-sm">
          Showing {recentScans.length} of 5 most recent scans
        </Text>
      </View>

      {/* ── List ── */}
      <FlatList
        data={recentScans}
        keyExtractor={(item, index) => `${item.date}-${index}`}
        renderItem={({ item }) => (
          <ScanCard
            imageUri={item.imageUri}
            imageSource={item.imageSource}
            plantName={item.plantName}
            disease={item.disease}
            confidence={item.confidence}
            date={item.date}
            onViewDetail={() => {
              setSelectedScan(item);
              router.push("/results");
            }}
          />
        )}
        contentContainerStyle={{ padding: 16, flexGrow: 1 }}
        ListEmptyComponent={<EmptyState />}
        showsVerticalScrollIndicator={false}
      />
    </SafeAreaView>
  );
}
