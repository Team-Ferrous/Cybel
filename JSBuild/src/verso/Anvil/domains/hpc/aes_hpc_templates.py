"""AES HPC templates used to scaffold compliant generated code."""

ALIGNED_SIMD_KERNEL = """
#include <immintrin.h>
#include <omp.h>

// AES-HPC-2: explicit alignment contract for AVX2 loads and stores
struct alignas(32) AlignedBuffer {
    float data[8];
};

float simd_scale_sum(const float* input, float* output, float scale, int n) {
    float total = 0.0f;
    const int vectorized_end = (n / 8) * 8;

    #pragma omp parallel for schedule(static) \\
        shared(output, input, vectorized_end) firstprivate(scale) reduction(+:total)
    for (int i = 0; i < vectorized_end; i += 8) {
        __m256 v = _mm256_load_ps(&input[i]);
        __m256 s = _mm256_set1_ps(scale);
        __m256 r = _mm256_mul_ps(v, s);
        _mm256_store_ps(&output[i], r);

        alignas(32) float lane_sum[8];
        _mm256_store_ps(lane_sum, r);
        for (int j = 0; j < 8; ++j) {
            total += lane_sum[j];
        }
    }

    for (int i = vectorized_end; i < n; ++i) {
        output[i] = input[i] * scale;
        total += output[i];
    }

    return total;
}

// AES-HPC-4: scalar reference oracle is mandatory for parity tests.
float scalar_scale_sum(const float* input, float* output, float scale, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * scale;
        total += output[i];
    }
    return total;
}

// Evidence scaffold: scalar-vs-vector parity bundle for every SIMD kernel.
struct EquivalenceEvidence {
    const char* trace_id;
    const char* evidence_bundle_id;
    float scalar_total;
    float simd_total;
    float abs_error;
};
"""
