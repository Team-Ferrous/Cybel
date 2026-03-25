// highnoon/_native/ops/fused_qgan_op.h
// Quantum GAN helpers
#ifndef HIGHNOON_NATIVE_OPS_FUSED_QGAN_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_QGAN_OP_H_
#include <cstdint>
#include <cmath>
namespace highnoon { namespace ops {
// Wasserstein distance approximation
inline float qgan_wasserstein(const float* real, const float* fake, int64_t size) {
    float dist = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        dist += std::abs(real[i] - fake[i]);
    }
    return dist / size;
}
}} // namespace
#endif
