// app/(tabs)/index.jsx — My Farm Screen

import { useState } from "react";
import {
  View,
  Text,
  TextInput,
  FlatList,
  Image,
  TouchableOpacity,
  SafeAreaView,
} from "react-native";
import {
  useFonts,
  Poppins_600SemiBold,
  Poppins_400Regular,
} from "@expo-google-fonts/poppins";
import { useRouter } from "expo-router";
import { Search } from "lucide-react-native";

import usePlantStore from "../../store/usePlantStore";

const CATEGORIES = ["All", "Healthy", "Diseased", "Favourites"];

const ACTIVE_GREEN = "#08AF4E";
const TITLE_COLOR  = "#561111";

// ─── Screen ───────────────────────────────────────────────────────────────────
export default function HomeScreen() {
  const router = useRouter();

  const [search, setSearch]                 = useState("");
  const [activeCategory, setActiveCategory] = useState("All");

  const [fontsLoaded] = useFonts({ Poppins_600SemiBold, Poppins_400Regular });

  const plants          = usePlantStore((s) => s.plants);
  const setSelectedScan = usePlantStore((s) => s.setSelectedScan);

  // Filter by search text and active category tab
  const filtered = plants.filter((p) => {
    const matchesSearch = p.name.toLowerCase().includes(search.toLowerCase());

    const matchesCategory =
      activeCategory === "All" ||
      (activeCategory === "Healthy"    && p.status === "healthy")  ||
      (activeCategory === "Diseased"   && p.status === "diseased") ||
      (activeCategory === "Favourites" && p.isFavourite);

    return matchesSearch && matchesCategory;
  });

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#ffffff" }}>
      {/* ── Title ── */}
      <View style={{ paddingHorizontal: 20, paddingTop: 24, paddingBottom: 12 }}>
        <Text
          style={{
            fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
            fontSize: 32,
            color: TITLE_COLOR,
            lineHeight: 40,
          }}
        >
          My Farm
        </Text>
      </View>

      {/* ── Search bar ── */}
      <View style={{ paddingHorizontal: 20, marginBottom: 16 }}>
        <View
          style={{
            flexDirection: "row",
            alignItems: "center",
            backgroundColor: "#f3f4f6",
            borderRadius: 50,
            paddingHorizontal: 14,
            paddingVertical: 10,
          }}
        >
          <Search size={18} color="#9ca3af" />
          <TextInput
            style={{
              flex: 1,
              marginLeft: 8,
              fontSize: 15,
              color: "#374151",
              fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
            }}
            placeholder="Search plants…"
            placeholderTextColor="#9ca3af"
            value={search}
            onChangeText={setSearch}
            returnKeyType="search"
          />
        </View>
      </View>

      {/* ── Category strip ── */}
      <View style={{ marginBottom: 16 }}>
        <View
          style={{
            flexDirection: "row",
            borderBottomWidth: 0.5,
            borderBottomColor: "#e5e7eb",
          }}
        >
          {CATEGORIES.map((cat) => {
            const active = activeCategory === cat;
            return (
              <TouchableOpacity
                key={cat}
                onPress={() => setActiveCategory(cat)}
                style={{ flex: 1, alignItems: "center", justifyContent: "center" }}
                activeOpacity={0.7}
              >
                <View
                  style={{
                    paddingHorizontal: active ? 4 : 0,
                    paddingBottom: 6,
                    borderBottomWidth: active ? 2 : 0,
                    borderBottomColor: ACTIVE_GREEN,
                  }}
                >
                  <Text
                    style={{
                      fontFamily: fontsLoaded
                        ? active
                          ? "Poppins_600SemiBold"
                          : "Poppins_400Regular"
                        : undefined,
                      fontSize: 14,
                      color: active ? ACTIVE_GREEN : "#9ca3af",
                    }}
                  >
                    {cat}
                  </Text>
                </View>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* ── Plant grid ── */}
      <FlatList
        data={filtered}
        keyExtractor={(item) => item.id}
        numColumns={2}
        contentContainerStyle={{ paddingHorizontal: 16, paddingBottom: 24 }}
        columnWrapperStyle={{ justifyContent: "space-between" }}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <View style={{ flex: 1, alignItems: "center", marginTop: 60 }}>
            <Text
              style={{
                fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
                color: "#9ca3af",
                fontSize: 15,
              }}
            >
              No plants found.
            </Text>
          </View>
        }
        renderItem={({ item }) => (
          <PlantCard
            plant={item}
            fontsLoaded={fontsLoaded}
            onPress={() => {
              setSelectedScan({
                plantName:   item.name,
                imageSource: item.image,
                imageUri:    null,
                disease:     item.disease ?? (item.status === "healthy" ? "Healthy Plant" : "Disease Detected"),
                confidence:  item.confidence ?? null,
                treatment:   item.treatment ?? null,
              });
              router.push("/results");
            }}
          />
        )}
      />
    </SafeAreaView>
  );
}

// ─── Plant card ───────────────────────────────────────────────────────────────
function PlantCard({ plant, fontsLoaded, onPress }) {
  const isHealthy  = plant.status === "healthy";
  const badgeColor = isHealthy ? ACTIVE_GREEN : "#ef4444";

  return (
    <TouchableOpacity
      onPress={onPress}
      activeOpacity={0.85}
      style={{ width: "48%", marginBottom: 18 }}
    >
      {/* Plant image */}
      <View
        style={{
          width: "100%",
          aspectRatio: 1,
          borderRadius: 14,
          backgroundColor: plant.bgColor ?? "#e5e7eb",
          overflow: "hidden",
        }}
      >
        {plant.image ? (
          <Image
            source={plant.image}
            style={{ width: "100%", height: "100%" }}
            resizeMode="cover"
          />
        ) : null}
      </View>

      {/* Name + status badge */}
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          marginTop: 6,
          flexWrap: "wrap",
          gap: 6,
        }}
      >
        <Text
          style={{
            fontFamily: fontsLoaded ? "Poppins_600SemiBold" : undefined,
            fontSize: 14,
            color: "#561111",
          }}
        >
          {plant.name}
        </Text>

        <View
          style={{
            borderWidth: 1,
            borderColor: badgeColor,
            borderRadius: 5,
            paddingHorizontal: 8,
            paddingVertical: 1,
          }}
        >
          <Text
            style={{
              fontFamily: fontsLoaded ? "Poppins_400Regular" : undefined,
              fontSize: 11,
              color: badgeColor,
            }}
          >
            {isHealthy ? "Healthy" : "Diseased"}
          </Text>
        </View>
      </View>
    </TouchableOpacity>
  );
}
