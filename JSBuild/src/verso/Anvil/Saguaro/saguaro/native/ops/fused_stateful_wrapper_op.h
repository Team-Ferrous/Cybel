// saguaro.native/ops/fused_stateful_wrapper_op.h
// Stateful wrapper helpers for SSM state management
#ifndef SAGUARO_NATIVE_OPS_FUSED_STATEFUL_WRAPPER_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_STATEFUL_WRAPPER_OP_H_
#include <cstdint>
#include <algorithm>
namespace saguaro { namespace ops {
inline void stateful_reset(float* state, int64_t size) {
    std::fill(state, state + size, 0.0f);
}
inline void stateful_copy(const float* src, float* dst, int64_t size) {
    std::copy(src, src + size, dst);
}
}} // namespace
#endif
