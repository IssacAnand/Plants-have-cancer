const fs = require("fs");
const path = require("path");

const cmakePath = path.join(
  __dirname,
  "..",
  "node_modules",
  "onnxruntime-react-native",
  "android",
  "CMakeLists.txt"
);

if (!fs.existsSync(cmakePath)) {
  console.log("[postinstall] onnxruntime-react-native CMakeLists.txt not found, skipping");
  process.exit(0);
}

let content = fs.readFileSync(cmakePath, "utf8");
let updated = content;

if (!updated.includes("set(CMAKE_OBJECT_PATH_MAX 128)")) {
  updated = updated.replace(
    "cmake_minimum_required(VERSION 3.9.0)\n",
    "cmake_minimum_required(VERSION 3.9.0)\n\n# Keep generated object paths short enough for Windows/Ninja builds in deep workspaces.\nset(CMAKE_OBJECT_PATH_MAX 128)\n"
  );
}

updated = updated.replace(
  /file\(TO_CMAKE_PATH\s*\n\s*"\$\{NODE_MODULES_DIR\}\/react-native\/ReactCommon\/jsi\/jsi\/jsi\.cpp" libPath\)/,
  'file(TO_CMAKE_PATH\n     "${NODE_MODULES_DIR}/react-native/ReactCommon/jsi/jsi/jsi.cpp" reactNativeJsiPath)\n\nset(ORT_CPP_STAGING_DIR "${CMAKE_CURRENT_BINARY_DIR}/ort_cpp")\nfile(MAKE_DIRECTORY ${ORT_CPP_STAGING_DIR})\nconfigure_file(${reactNativeJsiPath} ${ORT_CPP_STAGING_DIR}/jsi.cpp COPYONLY)\nconfigure_file(../cpp/JsiMain.cpp ${ORT_CPP_STAGING_DIR}/JsiMain.cpp COPYONLY)\nconfigure_file(../cpp/InferenceSessionHostObject.cpp ${ORT_CPP_STAGING_DIR}/InferenceSessionHostObject.cpp COPYONLY)\nconfigure_file(../cpp/JsiUtils.cpp ${ORT_CPP_STAGING_DIR}/JsiUtils.cpp COPYONLY)\nconfigure_file(../cpp/SessionUtils.cpp ${ORT_CPP_STAGING_DIR}/SessionUtils.cpp COPYONLY)\nconfigure_file(../cpp/TensorUtils.cpp ${ORT_CPP_STAGING_DIR}/TensorUtils.cpp COPYONLY)'
);

updated = updated.replace(
  /add_library\(\s*\n\s*onnxruntimejsi SHARED\s*\n\s*\$\{libPath\}\s*\n\s*src\/main\/cpp\/cpp-adapter\.cpp\s*\n\s*\.\.\/cpp\/JsiMain\.cpp\s*\n\s*\.\.\/cpp\/InferenceSessionHostObject\.cpp\s*\n\s*\.\.\/cpp\/JsiUtils\.cpp\s*\n\s*\.\.\/cpp\/SessionUtils\.cpp\s*\n\s*\.\.\/cpp\/TensorUtils\.cpp\)/,
  "add_library(\n  onnxruntimejsi SHARED\n  ${ORT_CPP_STAGING_DIR}/jsi.cpp\n  src/main/cpp/cpp-adapter.cpp\n  ${ORT_CPP_STAGING_DIR}/JsiMain.cpp\n  ${ORT_CPP_STAGING_DIR}/InferenceSessionHostObject.cpp\n  ${ORT_CPP_STAGING_DIR}/JsiUtils.cpp\n  ${ORT_CPP_STAGING_DIR}/SessionUtils.cpp\n  ${ORT_CPP_STAGING_DIR}/TensorUtils.cpp)"
);

if (updated === content) {
  console.log("[postinstall] onnxruntime-react-native CMake workaround already applied");
  process.exit(0);
}

fs.writeFileSync(cmakePath, updated);
console.log("[postinstall] applied onnxruntime-react-native CMake workaround");
