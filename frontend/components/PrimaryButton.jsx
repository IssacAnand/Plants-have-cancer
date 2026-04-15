// components/PrimaryButton.jsx
//
// ─────────────────────────────────────────────────────────────────────────────
// CONCEPT: Reusable Components & Props
// ─────────────────────────────────────────────────────────────────────────────
//
// A component is a function that returns JSX (the HTML-like syntax).
// Props are the "arguments" you pass to a component from its parent.
//
// Instead of copy-pasting the same green button everywhere, we define it ONCE
// and customise it via props:
//
//   <PrimaryButton label="Get Started" onPress={handlePress} />
//   <PrimaryButton label="Use Photo"   onPress={handleUse}   outline />
//
// The `outline` prop switches between a filled and outlined style.
// ─────────────────────────────────────────────────────────────────────────────

import { TouchableOpacity, Text, ActivityIndicator } from "react-native";

/**
 * @param {object} props
 * @param {string}   props.label     - Button text
 * @param {function} props.onPress   - Tap handler
 * @param {boolean}  [props.outline] - If true, renders as outlined/secondary button
 * @param {boolean}  [props.loading] - If true, shows a spinner instead of label
 * @param {boolean}  [props.disabled]
 * @param {string}   [props.className] - Extra NativeWind classes
 */
export default function PrimaryButton({
  label,
  onPress,
  outline   = false,
  loading   = false,
  disabled  = false,
  className = "",
}) {
  const isDisabled = disabled || loading;

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={isDisabled}
      activeOpacity={0.8}
      className={`
        w-full py-4 rounded-full items-center justify-center
        ${outline
          ? "border-2 border-primary bg-transparent"
          : "bg-primary"
        }
        ${isDisabled ? "opacity-50" : "opacity-100"}
        ${className}
      `}
    >
      {loading ? (
        <ActivityIndicator color={outline ? "#22C55E" : "#ffffff"} />
      ) : (
        <Text
          className={`text-base font-bold tracking-wide ${
            outline ? "text-primary" : "text-white"
          }`}
        >
          {label}
        </Text>
      )}
    </TouchableOpacity>
  );
}
