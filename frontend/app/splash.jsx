// app/splash.jsx  — Route: /splash
//
// Wraps the SplashScreen component and navigates to the main app
// when the user taps "Let's Go!".

import { useRouter } from "expo-router";
import SplashScreen from "../components/SplashScreen";

export default function SplashRoute() {
  const router = useRouter();

  function handleLetsGo() {
    // Replace so the user can't swipe back to the splash
    router.replace("/(tabs)");
  }

  return <SplashScreen onPress={handleLetsGo} />;
}
