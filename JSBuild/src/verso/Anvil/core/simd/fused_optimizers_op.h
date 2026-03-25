// highnoon/_native/ops/fused_optimizers_op.h
// Custom optimizer helpers
#ifndef HIGHNOON_NATIVE_OPS_FUSED_OPTIMIZERS_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_OPTIMIZERS_OP_H_
#include <cstdint>
#include <cmath>
namespace highnoon { namespace ops {

// Fused AdamW update
inline void optimizer_adamw_update(
    float* param, float* m, float* v,
    const float* grad, int64_t size,
    float lr, float beta1, float beta2, float eps, float wd, int64_t t) {
    
    float bc1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    float bc2 = 1.0f - std::pow(beta2, static_cast<float>(t));
    
    for (int64_t i = 0; i < size; ++i) {
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        param[i] = param[i] - lr * (m_hat / (std::sqrt(v_hat) + eps) + wd * param[i]);
    }
}

}} // namespace
#endif
