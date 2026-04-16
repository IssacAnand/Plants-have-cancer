// components/SplashScreen.jsx
//
// Splash screen — first thing the user sees on launch.
// Layout: Stack composition — upper content block + bottom-anchored oversized mascot.

import { useEffect } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  StatusBar,
} from "react-native";

import { useFonts } from "expo-font";
import {
  MontserratAlternates_800ExtraBold,
} from "@expo-google-fonts/montserrat-alternates";
import {
  Poppins_400Regular,
  Poppins_600SemiBold,
} from "@expo-google-fonts/poppins";

const { width, height } = Dimensions.get("window");

// Mascot is oversized — spans full screen width, bottom half clipped by screen edge
const MASCOT_SIZE = width * 1.4;
// How much of the mascot to clip below the bottom edge (shows upper body / head)
const MASCOT_CLIP = MASCOT_SIZE * 0.30;

/**
 * @param {object}   props
 * @param {function} props.onPress  - Called when the user taps "Let's Go!"
 */
export default function SplashScreen({ onPress }) {
  const [fontsLoaded, fontsError] = useFonts({
    MontserratAlternates_800ExtraBold,
    Poppins_400Regular,
    Poppins_600SemiBold,
  });

  useEffect(() => {
    if (fontsError) {
      console.warn("[splash] Font loading failed; using fallback system fonts.", fontsError);
    }
  }, [fontsError]);

  return (
    <View style={styles.root}>
      {/* Hide status bar for edge-to-edge rendering */}
      <StatusBar hidden />

      {/* ── Upper content block ───────────────────────────────────────────── */}
      <View style={styles.contentBlock}>
        <Text style={styles.brandName}>Farm</Text>
        <Text style={styles.brandName}>pals</Text>

        <Text style={styles.tagline}>
          your one stop smart plant{"\n"}care app for your farm
        </Text>

        <TouchableOpacity
          style={styles.button}
          onPress={onPress}
          activeOpacity={0.85}
        >
          <Text style={styles.buttonLabel}>Let's Go!</Text>
        </TouchableOpacity>
      </View>

      {/* ── Mascot — oversized, bottom-anchored, lower half clipped ─────── */}
      <View style={styles.mascotWrapper} pointerEvents="none">
        <Image
          source={require("../assets/corn.png")}
          style={styles.mascotImage}
          resizeMode="contain"
        />
      </View>
    </View>
  );
}

// ─── Design tokens ────────────────────────────────────────────────────────────
const BRAND_YELLOW = "#E5BD04";
const BRAND_GREEN  = "#08AF4E";

const styles = StyleSheet.create({
  // Stack container — fills screen, clips overflow so mascot bottom is hidden
  root: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    overflow: "hidden",
  },

  // Centered upper block — title, subtitle, button tightly grouped
  contentBlock: {
    position: "absolute",
    top: height * 0.27,
    left: 0,
    right: 0,
    alignItems: "center",
    paddingHorizontal: 32,
  },

  brandName: {
    fontFamily: "MontserratAlternates_800ExtraBold",
    fontSize: 64,
    color: BRAND_YELLOW,
    textAlign: "center",
    letterSpacing: 0,
    marginTop: -20,
  },

  tagline: {
    fontFamily: "Poppins_400Regular",
    fontSize: 14,
    color: "#792020",
    textAlign: "center",
    marginTop: 10,
    lineHeight: 19,
  },

  button: {
    marginTop: 18,
    backgroundColor: BRAND_GREEN,
    paddingVertical: 10,
    paddingHorizontal: 60,
    borderRadius: 10,
  },

  buttonLabel: {
    fontFamily: "Poppins_600SemiBold",
    fontSize: 14,
    color: "#FFFFFF",
    letterSpacing: 0.3,
  },

  // Mascot pinned to bottom — positioned so lower half sits off-screen
  mascotWrapper: {
    position: "absolute",
    bottom: -MASCOT_CLIP,
    left: (width - MASCOT_SIZE) / 2,
    width: MASCOT_SIZE,
    height: MASCOT_SIZE,
  },

  mascotImage: {
    width: "100%",
    height: "100%",
  },
});
