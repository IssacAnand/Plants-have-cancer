// app/index.jsx
//
// Root entry point — immediately redirects to the splash screen.
// Expo Router resolves native apps to "/" on launch, so without this file
// it falls through to (tabs)/index.jsx, skipping the splash entirely.

import { Redirect } from "expo-router";

export default function Index() {
  return <Redirect href="/splash" />;
}
