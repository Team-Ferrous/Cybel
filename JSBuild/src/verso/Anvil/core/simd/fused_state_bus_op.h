// highnoon/_native/ops/fused_state_bus_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file fused_state_bus_op.h
 * @brief Helper utilities for the State Bus fused op.
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_STATE_BUS_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_STATE_BUS_OP_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace highnoon {
namespace ops {

inline float statebus_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline void statebus_sigmoid_inplace(float* data, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        data[i] = statebus_sigmoid(data[i]);
    }
}

inline void statebus_softmax(float* data, int64_t size) {
    if (size <= 0) {
        return;
    }
    float max_val = data[0];
    for (int64_t i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    if (sum <= 0.0f) {
        return;
    }
    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < size; ++i) {
        data[i] *= inv_sum;
    }
}

inline float statebus_dot(const float* a, const float* b, int64_t size) {
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_STATE_BUS_OP_H_
