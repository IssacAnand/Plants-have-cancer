// tailwind.config.js
//
// CONCEPT: This file tells Tailwind CSS two things:
//   1. WHERE to look for your className usage (the `content` array)
//   2. HOW to extend or customise the default design system (`theme.extend`)
//
// NativeWind reads this config so the same Tailwind classes you know from
// web development work inside React Native components via the `className` prop.

/** @type {import('tailwindcss').Config} */
module.exports = {
  // Scan these file patterns for Tailwind class names
  content: [
    "./app/**/*.{js,jsx}",
    "./components/**/*.{js,jsx}",
  ],

  theme: {
    extend: {
      colors: {
        // Brand colours — use as: className="bg-primary text-primary" etc.
        primary:       "#22C55E",  // Main green (matches the app's green buttons)
        "primary-dark":"#16A34A",  // Darker green for pressed states
        surface:       "#F0FAF0",  // Soft green-tinted page background
        card:          "#FFFFFF",  // White cards
        disease:       "#F97316",  // Orange — disease name text
        confidence:    "#22C55E",  // Green — confidence percentage
      },
      fontFamily: {
        // Add custom fonts here after running expo-font, e.g.:
        // sans: ["Inter_400Regular"],
      },
    },
  },

  plugins: [],
};
