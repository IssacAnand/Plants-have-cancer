// babel.config.js
//
// CONCEPT: Babel is a "translator" that converts modern JavaScript (and JSX)
// into code that React Native can understand and run.
//
// babel-preset-expo handles all the standard Expo transforms.
// "nativewind/babel" is a PLUGIN that scans your className props and converts
// Tailwind class strings into actual React Native StyleSheet objects at build time.

module.exports = function (api) {
  api.cache(true);
  return {
    presets: [
      ["babel-preset-expo", { jsxImportSource: "nativewind" }],
      "nativewind/babel",
    ],
  };
};
