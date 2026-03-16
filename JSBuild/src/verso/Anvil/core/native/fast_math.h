#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace anvil_fast_math {

inline float fast_exp_scalar(float x) {
    // Robust saturation to avoid NaN/Inf from exponent bit construction on
    // extreme negatives (observed in flash-attention correction terms).
    if (x <= -80.0f) {
        return 0.0f;
    }
    if (x >= 88.0f) {
        return std::numeric_limits<float>::infinity();
    }
    x = std::min(88.0f, std::max(-80.0f, x));
    const float log2e = 1.44269504088896341f;
    const float ln2 = 0.6931471805599453f;
    const float fx = std::floor(x * log2e);
    const float r = x - fx * ln2;
    const float p =
        (((((0.008301359601318836f * r + 0.04165734723210335f) * r
            + 0.16666552424430847f) * r + 0.49999985098838806f) * r + 1.0f) * r + 1.0f);
    const int n = static_cast<int>(fx);
    if (n < -126) {
        return 0.0f;
    }
    if (n > 127) {
        return std::numeric_limits<float>::infinity();
    }
    std::uint32_t bits = static_cast<std::uint32_t>(n + 127) << 23;
    float pow2n = 0.0f;
    std::memcpy(&pow2n, &bits, sizeof(float));
    return p * pow2n;
}

#ifdef __AVX2__
inline __m256 v_expf(__m256 x) {
    // Adapted from ik_llama.cpp (MIT): AVX2 exp approximation with robust range behavior.
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(0x1.7f7d1cp-20f),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x)
    );
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(
        _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1)))
    );
    const __m256i c = _mm256_castps_si256(
        _mm256_cmp_ps(
            _mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
            _mm256_set1_ps(126),
            _CMP_GT_OQ
        )
    );
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_set1_ps(0x1.0e4020p-7f),
                b,
                _mm256_set1_ps(0x1.573e2ep-5f)
            ),
            u,
            _mm256_fmadd_ps(
                _mm256_set1_ps(0x1.555e66p-3f),
                b,
                _mm256_set1_ps(0x1.fffdb6p-2f)
            )
        ),
        u,
        _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b)
    );
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) {
        return _mm256_fmadd_ps(j, k, k);
    }
    const __m256i g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x82000000u)
    );
    const __m256 s1 =
        _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    const __m256i d = _mm256_castps_si256(
        _mm256_cmp_ps(
            _mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
            _mm256_set1_ps(192),
            _CMP_GT_OQ
        )
    );
    return _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(
                    _mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)
                ),
                _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k))
            )
        )
    );
}

inline __m256 v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    const __m256 exp_neg_x = v_expf(neg_x);
    return _mm256_div_ps(x, _mm256_add_ps(one, exp_neg_x));
}

// Adapted from ik_llama.cpp (MIT) — iqk_utils.h
inline __m256 v_tanh(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 exp_two_x = v_expf(_mm256_mul_ps(x, _mm256_set1_ps(2.0f)));
    const __m256 res = _mm256_div_ps(
        _mm256_sub_ps(exp_two_x, one),
        _mm256_add_ps(exp_two_x, one)
    );
    const __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(10.0f), _CMP_GT_OQ);
    return _mm256_or_ps(_mm256_and_ps(mask, one), _mm256_andnot_ps(mask, res));
}

// Adapted from ik_llama.cpp (MIT) — iqk_utils.h
inline __m256 v_gelu(__m256 x, __m256 c1, __m256 c2) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(10.0f), _CMP_GT_OQ);
    __m256 arg = _mm256_add_ps(one, _mm256_mul_ps(_mm256_mul_ps(x, x), c1));
    arg = _mm256_mul_ps(arg, _mm256_mul_ps(x, c2));
    const __m256 exp_arg = v_expf(arg);
    const __m256 gelu = _mm256_mul_ps(
        x, _mm256_div_ps(exp_arg, _mm256_add_ps(exp_arg, one))
    );
    return _mm256_or_ps(_mm256_and_ps(mask, x), _mm256_andnot_ps(mask, gelu));
}
#endif

}  // namespace anvil_fast_math
