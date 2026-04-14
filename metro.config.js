// metro.config.js
// Registers .onnx files as bundled assets so require('../assets/models/plant_classifier.onnx')
// works at runtime in a native dev build.

const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push("onnx");

module.exports = config;
