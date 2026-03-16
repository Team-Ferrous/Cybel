#include <cstdint>

extern "C" std::int32_t anvil_backend_qwen35_marker() {
    return 1;
}

extern "C" const char* anvil_backend_qwen35_name() {
#ifdef ANVIL_BACKEND_NAME
    return ANVIL_BACKEND_NAME;
#else
    return "qwen35";
#endif
}

extern "C" const char* anvil_backend_qwen35_build_id() {
#ifdef ANVIL_NATIVE_BUILD_ID
    return ANVIL_NATIVE_BUILD_ID;
#else
    return "";
#endif
}

extern "C" std::int32_t anvil_backend_qwen35_abi_version() {
#ifdef ANVIL_NATIVE_BACKEND_ABI_VERSION
    return static_cast<std::int32_t>(ANVIL_NATIVE_BACKEND_ABI_VERSION);
#else
    return static_cast<std::int32_t>(1);
#endif
}
