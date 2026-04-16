// components/ScanCard.jsx
//
// A card component that displays a single scan record in the History screen.
// Receives all its data as props — it doesn't touch any global state directly.

import { View, Text, Image, TouchableOpacity } from "react-native";

/**
 * @param {object}   props
 * @param {string}   props.imageUri      - Local URI of the scanned leaf photo
 * @param {*}        props.imageSource   - require() asset (takes precedence over imageUri)
 * @param {string}   props.plantName     - User-facing plant name
 * @param {string}   props.disease       - Detected disease name
 * @param {number}   props.confidence    - 0–100
 * @param {string}   props.date          - ISO date string
 * @param {function} props.onViewDetail  - Called when "View in Detail" is tapped
 */
export default function ScanCard({ imageUri, imageSource, plantName, disease, confidence, date, onViewDetail }) {
  const imageDisplaySource = imageSource ?? (imageUri ? { uri: imageUri } : null);

  // Format the date to something readable, e.g. "Mar 8, 2025"
  const formattedDate = new Date(date).toLocaleDateString("en-US", {
    month: "short",
    day:   "numeric",
    year:  "numeric",
  });

  // Pick a colour for the confidence badge
  const confidenceColor =
    confidence >= 80 ? "text-green-600" :
    confidence >= 60 ? "text-yellow-600" :
                       "text-red-500";

  return (
    <View className="bg-white rounded-2xl p-3 mb-3 shadow-sm border border-gray-100">
      <View className="flex-row items-center">
        {/* Thumbnail */}
        {imageDisplaySource ? (
          <Image
            source={imageDisplaySource}
            className="w-16 h-16 rounded-xl bg-gray-100"
            resizeMode="cover"
          />
        ) : (
          <View className="w-16 h-16 rounded-xl bg-green-100 items-center justify-center">
            <Text className="text-3xl">🌿</Text>
          </View>
        )}

        {/* Text info */}
        <View className="flex-1 ml-3">
          {plantName ? (
            <Text className="text-gray-500 text-xs font-medium mb-0.5">{plantName}</Text>
          ) : null}
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

      {/* View in Detail button */}
      {onViewDetail ? (
        <TouchableOpacity
          onPress={onViewDetail}
          activeOpacity={0.7}
          className="mt-3 border border-gray-200 rounded-lg py-2 items-center"
        >
          <Text className="text-gray-600 text-sm font-medium">View in Detail</Text>
        </TouchableOpacity>
      ) : null}
    </View>
  );
}
