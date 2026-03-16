#include <cstdint>
#include <complex>
#include <vector>
#include "../hnn_simd_common.h"
#include "../circular_conv_op.h"
#include "../hyperdimensional_embedding_op.h"

extern "C" {
    // 1. Basic SIMD Primivites
    void simd_exp_inplace(float* data, int64_t size) {
        highnoon::ops::simd_exp_inplace(data, size);
    }
    void simd_log_inplace(float* data, int64_t size) {
        highnoon::ops::simd_log_inplace(data, size);
    }
    void simd_sigmoid_inplace(float* data, int64_t size) {
        highnoon::ops::simd_sigmoid_inplace(data, size);
    }

    void simd_hadamard_product(const float* a, const float* b, float* out, int64_t size) {
        highnoon::ops::simd_hadamard_product(a, b, out, size);
    }

    void simd_hadamard_product_batched(
        const float* a, const float* b, float* out,
        int batch, int seq, int hd_dim) {
        
        int64_t seq_stride = hd_dim;
        int64_t batch_stride = (int64_t)seq * hd_dim;
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < seq; ++j) {
                const float* a_ptr = a + i * batch_stride + j * seq_stride;
                const float* b_ptr = b + j * seq_stride;  // b is shared across batches
                float* out_ptr = out + i * batch_stride + j * seq_stride;
                
                highnoon::ops::simd_hadamard_product(a_ptr, b_ptr, out_ptr, hd_dim);
            }
        }
    }
    
    // Helper to find next power of 2
    inline int next_pow2(int n) {
        if (n <= 1) return 1;
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    // 2. Circular Convolution
    void circular_convolution_batched(
        const float* a,
        const float* b,
        float* out,
        int batch, int seq, int hd_dim) {
        
        int n_pow2 = next_pow2(hd_dim);
        
        if (n_pow2 == hd_dim) {
            // Standard path for power-of-2
            highnoon::ops::circular_convolution_batched(a, b, out, batch, seq, hd_dim);
        } else {
            // Padded path for non-power-of-2 (O(D log D))
            // We use linear convolution via padding to next power of 2
            // For holographic binding, we want to maintain the same "meaning"
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < batch; ++i) {
                for (int j = 0; j < seq; ++j) {
                    std::vector<float> a_pad(n_pow2, 0.0f);
                    std::vector<float> b_pad(n_pow2, 0.0f);
                    std::vector<float> out_pad(n_pow2, 0.0f);
                    
                    int64_t batch_stride = (int64_t)seq * hd_dim;
                    int64_t seq_stride = hd_dim;
                    
                    std::memcpy(a_pad.data(), a + i * batch_stride + j * seq_stride, hd_dim * sizeof(float));
                    std::memcpy(b_pad.data(), b + j * seq_stride, hd_dim * sizeof(float));
                    
                    highnoon::ops::circular_convolution_inplace(a_pad.data(), b_pad.data(), out_pad.data(), n_pow2);
                    
                    std::memcpy(out + i * batch_stride + j * seq_stride, out_pad.data(), hd_dim * sizeof(float));
                }
            }
        }
    }

    // 3. Holographic Bundling
    void holographic_bundle(
        const int* token_ids, const float* base_vectors,
        const float* position_keys, const float* amplitudes,
        float* output, int batch, int seq, int vocab_size,
        int hd_dim, int model_dim) {
        
        hsmn::hqe::HQEConfig config;
        config.hd_dim = hd_dim;
        config.model_dim = model_dim;
        
        hsmn::hqe::HolographicBundle(token_ids, base_vectors, position_keys, 
                                     amplitudes, output, config, batch, seq, vocab_size);
    }
    
    // 4. Scratch Pool
    float* scratch_get(int64_t size, int slot) {
        return g_path_scratch.get(size, slot);
    }
    
    void scratch_clear() {
        g_path_scratch.clear();
    }

    // =========================================================================
    // 5. SSM/Mamba Layer Operations for Sparse Forward Pass
    // =========================================================================
}

#include "../fused_mamba_op.h"

extern "C" {
    /**
     * @brief Full SSM scan for sequence processing.
     * SIMD-optimized with AVX2/AVX512/NEON support.
     */
    void ssm_scan(
        const float* x,         // [batch, seq, d_inner]
        const float* A_log,     // [d_inner, state_dim]
        const float* dt,        // [batch, seq, d_inner]
        const float* B,         // [batch, seq, state_dim]
        const float* C,         // [batch, seq, state_dim]
        const float* D,         // [d_inner]
        float* output,          // [batch, seq, d_inner]
        float* h_final,         // [batch, d_inner, state_dim] or nullptr
        int batch, int seq, int d_inner, int state_dim) {
        
        highnoon::ops::mamba_ssm_scan(
            x, A_log, dt, B, C, D, output, h_final,
            batch, seq, d_inner, state_dim
        );
    }

    /**
     * @brief Parallel SSM scan with chunked processing.
     * Uses 2-pass algorithm for better parallelism.
     */
    void ssm_scan_parallel(
        const float* x, const float* A_log, const float* dt,
        const float* B, const float* C, const float* D,
        float* output, float* h_final,
        int batch, int seq, int d_inner, int state_dim, int chunk_size) {
        
        highnoon::ops::mamba_parallel_ssm_scan(
            x, A_log, dt, B, C, D, output, h_final,
            batch, seq, d_inner, state_dim, chunk_size
        );
    }

    /**
     * @brief Depthwise 1D convolution with causal padding.
     * Used in Mamba pre-processing.
     */
    void depthwise_conv1d(
        const float* input,     // [batch, seq, channels]
        const float* filter,    // [kernel_size, 1, channels]
        const float* bias,      // [channels]
        float* output,          // [batch, seq, channels]
        int batch, int seq, int channels, int kernel_size) {
        
        highnoon::ops::mamba_depthwise_conv1d(
            input, filter, bias, output,
            batch, seq, channels, kernel_size
        );
    }

    /**
     * @brief SiLU activation (x * sigmoid(x)).
     */
    void silu_inplace(float* data, int64_t size) {
        highnoon::ops::mamba_silu_inplace(data, size);
    }

    /**
     * @brief Gated output: out = y * silu(z).
     */
    void gated_output(const float* y, const float* z, float* out, int64_t size) {
        highnoon::ops::mamba_gated_output(y, z, out, size);
    }
}
