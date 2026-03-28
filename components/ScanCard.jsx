// components/ScanCard.jsx
//
// A card component that displays a single scan record in the History screen.
// Receives all its data as props — it doesn't touch any global state directly.

import { View, Text, Image } from "react-native";

/**
 * @param {object} props
 * @param {string} props.imageUri    - Local URI of the scanned leaf photo
 * @param {string} props.disease     - Detected disease name
 * @param {number} props.confidence  - 0–100
 * @param {string} props.date        - ISO date string
 */
export default function ScanCard({ imageUri, disease, confidence, date }) {
  // Format the date to something readable, e.g. "Mar 8, 2025"
  const formattedDate = new Date(date).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  // Pick a colour for the confidence badge
  const confidenceColor =
    confidence >= 80 ? "text-green-600" :
    confidence >= 60 ? "text-yellow-600" :
                       "text-red-500";

  return (
    <View className="flex-row items-center bg-white rounded-2xl p-3 mb-3 shadow-sm border border-gray-100">
      {/* Thumbnail */}
      <Image
        source={{ uri: imageUri }}
        className="w-16 h-16 rounded-xl bg-gray-100"
        resizeMode="cover"
      />

      {/* Text info */}
      <View className="flex-1 ml-3">
        <Text className="text-gray-800 font-semibold text-base" numberOfLines={1}>
          {disease}
        </Text>
        <Text className={`text-sm font-medium mt-0.5 ${confidenceColor}`}>
          {confidence}% Confidence
        </Text>
        <Text className="text-gray-400 text-xs mt-1">{formattedDate}</Text>
      </View>

      {/* Arrow indicator */}
      <Text className="text-gray-300 text-xl ml-2">›</Text>
    </View>
  );
}
