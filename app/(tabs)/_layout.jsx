// app/(tabs)/_layout.jsx

import { Tabs } from "expo-router";
import { View } from "react-native";
import { House, ScanLine, User } from "lucide-react-native";

const ACTIVE   = "#08AF4E";
const INACTIVE = "#999999";

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: false,
        tabBarStyle: {
          backgroundColor: "#ffffff",
          borderTopWidth: 1,
          borderTopColor: "#eeeeee",
          height: 75,
          paddingBottom: 12,
          paddingTop: 16,
        },
        tabBarActiveTintColor:   ACTIVE,
        tabBarInactiveTintColor: INACTIVE,
      }}
    >
      {/* Home Tab */}
      <Tabs.Screen
        name="index"
        options={{
          tabBarIcon: ({ focused }) => (
            <House size={34} color={focused ? ACTIVE : INACTIVE} />
          ),
        }}
      />

      {/* Scan Tab — ScanLine inside gray circle */}
      <Tabs.Screen
        name="scan"
        options={{
          tabBarIcon: ({ focused }) => (
            <View
              style={{
                width: 52,
                height: 52,
                borderRadius: 26,
                backgroundColor: "#E8E8E8",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <ScanLine size={34} color={focused ? ACTIVE : INACTIVE} />
            </View>
          ),
        }}
      />

      {/* History — hidden from tab bar but kept as a route */}
      <Tabs.Screen
        name="history"
        options={{ href: null }}
      />

      {/* Profile Tab */}
      <Tabs.Screen
        name="profile"
        options={{
          tabBarIcon: ({ focused }) => (
            <User size={34} color={focused ? ACTIVE : INACTIVE} />
          ),
        }}
      />
    </Tabs>
  );
}
