#include <cstdint>

extern "C" {

const char* anvil_native_split_layout() {
#ifdef ANVIL_NATIVE_SPLIT_LAYOUT
    return ANVIL_NATIVE_SPLIT_LAYOUT;
#else
    return "kernels/runtime_core/backends/compat";
#endif
}

const char* anvil_native_public_load_target() {
#ifdef ANVIL_NATIVE_PUBLIC_LOAD_TARGET
    return ANVIL_NATIVE_PUBLIC_LOAD_TARGET;
#else
    return "libanvil_native_ops.so";
#endif
}

const char* anvil_native_runtime_core_target() {
#ifdef ANVIL_NATIVE_RUNTIME_CORE_TARGET
    return ANVIL_NATIVE_RUNTIME_CORE_TARGET;
#else
    return "libanvil_runtime_core.so";
#endif
}

const char* anvil_native_compat_alias_csv() {
#ifdef ANVIL_NATIVE_COMPAT_ALIAS_CSV
    return ANVIL_NATIVE_COMPAT_ALIAS_CSV;
#else
    return "libanvil_native_ops.so,libfast_attention.so,libcoconut_bridge.so";
#endif
}

std::int32_t anvil_native_split_abi_version() {
#ifdef ANVIL_NATIVE_SPLIT_ABI_VERSION
    return static_cast<std::int32_t>(ANVIL_NATIVE_SPLIT_ABI_VERSION);
#else
    return static_cast<std::int32_t>(1);
#endif
}

const char* anvil_native_isa_baseline() {
#ifdef ANVIL_NATIVE_ISA_BASELINE
    return ANVIL_NATIVE_ISA_BASELINE;
#else
    return "host";
#endif
}

const char* anvil_native_optional_isa_leaves() {
#ifdef ANVIL_NATIVE_OPTIONAL_ISA_LEAVES
    return ANVIL_NATIVE_OPTIONAL_ISA_LEAVES;
#else
    return "";
#endif
}

}  // extern "C"
