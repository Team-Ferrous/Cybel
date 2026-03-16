#include <cstdint>
#include <complex>
#include <vector>
#include <cstring>
#include <cmath>

// Common SIMD headers
#include "hnn_simd_common.h"
#include "common/perf_utils.h"
#include "common/tensor_stream_pool.h"

// Namespace Bridge: Saguaro -> HighNoon / HSMN
// Many kernels expect saguaro::ops utilities to be available in hsmn::ops or highnoon::ops
namespace hsmn {
    namespace ops {
        using namespace saguaro::ops;
    }
}
namespace highnoon {
    namespace ops {
        using namespace saguaro::ops;
    }
}

// Now include kernels which depend on the bridge
#include "circular_conv_op.h"
#include "hyperdimensional_embedding_op.h"
#include "fused_mamba_op.h"
#include "unified_attention_op.h"
#include "q_ssm_gating_op.h"

// OpenMP
#if defined(_OPENMP)
#include <omp.h>
#endif

extern "C" {
    // =========================================================================
    // 1. Basic SIMD Primitives
    // =========================================================================
    
    void simd_exp_inplace(float* data, int64_t size) {
        highnoon::ops::simd_exp_inplace(data, size);
    }
    void simd_log_inplace(float* data, int64_t size) {
        highnoon::ops::simd_log_inplace(data, size);
    }
    void simd_sigmoid_inplace(float* data, int64_t size) {
        highnoon::ops::simd_sigmoid_inplace(data, size);
    }
    void simd_softmax_inplace(float* data, int64_t size) {
        highnoon::ops::simd_softmax_inplace(data, size);
    }
    void simd_silu_inplace(float* data, int64_t size) {
        highnoon::ops::simd_silu_inplace(data, size);
    }
    void simd_gelu_inplace(float* data, int64_t size) {
        highnoon::ops::simd_gelu_inplace(data, size);
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
    
    float simd_dot_product(const float* a, const float* b, int64_t size) {
        return hsmn::attention::simd::dot_product(a, b, size);
    }
    
    float simd_cosine_similarity(const float* a, const float* b, int64_t size) {
        // Simple implementation reusing dot_product
        float dot = hsmn::attention::simd::dot_product(a, b, size);
        float norm_a = std::sqrt(hsmn::attention::simd::dot_product(a, a, size));
        float norm_b = std::sqrt(hsmn::attention::simd::dot_product(b, b, size));
        return dot / (norm_a * norm_b + 1e-8f);
    }
    
    // =========================================================================
    // 2. Circular Convolution
    // =========================================================================

    // Helper to find next power of 2
    inline int next_pow2(int n) {
        if (n <= 1) return 1;
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }

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
            // Padded path for non-power-of-2
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
    
    // =========================================================================
    // 3. Holographic Bundling
    // =========================================================================

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
    
    // Scratch Pool
    float* scratch_get(int64_t size, int slot) {
        return g_path_scratch.get(size, slot);
    }
    
    void scratch_clear() {
        g_path_scratch.clear();
    }

    // =========================================================================
    // 4. SSM/Mamba Operations
    // =========================================================================

    void ssm_scan(
        const float* x, const float* A_log, const float* dt,
        const float* B, const float* C, const float* D,
        float* output, float* h_final,
        int batch, int seq, int d_inner, int state_dim) {
        
        highnoon::ops::mamba_ssm_scan(
            x, A_log, dt, B, C, D, output, h_final,
            batch, seq, d_inner, state_dim
        );
    }

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

    void depthwise_conv1d(
        const float* input, const float* filter, const float* bias,
        float* output, int batch, int seq, int channels, int kernel_size) {
        
        highnoon::ops::mamba_depthwise_conv1d(
            input, filter, bias, output,
            batch, seq, channels, kernel_size
        );
    }

    void gated_output(const float* y, const float* z, float* out, int64_t size) {
        highnoon::ops::mamba_gated_output(y, z, out, size);
    }

    // =========================================================================
    // 5. Unified Attention & QSSM (NEW)
    // =========================================================================

    void simd_flash_attention_forward(
        const float* Q, const float* K, const float* V,
        float* output,
        int batch, int heads, int head_dim, int seq_len, int kv_seq_len,
        float scale, bool causal
    ) {
        hsmn::attention::UnifiedAttentionConfig config;
        config.mode = hsmn::attention::AttentionMode::FLASH;
        config.batch_size = batch;
        config.num_heads = heads;
        config.head_dim = head_dim;
        config.seq_len = seq_len;
        config.kv_seq_len = kv_seq_len;
        config.scale = scale;
        config.causal = causal;
        
        hsmn::attention::FlashAttentionForward(Q, K, V, config, output);
    }
    
    void simd_qssm_forward(
        const float* input, float* state, const float* vqc_params, float* output,
        int batch, int seq_len, int input_dim, int state_dim,
        int vqc_qubits, int vqc_layers
    ) {
        hsmn::qssm::QSSMConfig config;
        config.input_dim = input_dim;
        config.state_dim = state_dim;
        config.vqc_qubits = vqc_qubits;
        config.vqc_layers = vqc_layers;
        config.use_born_rule = true;
        
        hsmn::qssm::QSSMForward(input, state, vqc_params, output, config, batch, seq_len);
    }
}

// =============================================================================
// UNITY BUILD IMPLEMENTATIONS
// Include implementation files here to ensure they inherit namespace bridges
// =============================================================================

#include "unified_attention_op.cc"
#include "common/tensor_stream_pool.cc"
