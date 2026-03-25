#include "qsg_mps_entangle.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

namespace saguaro {
namespace ops {

// Helper to compute Von Neumann entropy of a singular value spectrum
// S are singular values from SVD
// Entropy = - sum(p * log(p)) where p = s^2 / sum(s^2)
inline float compute_entropy(const std::vector<float>& s) {
    float sum_sq = 0.0f;
    for (float val : s) {
        sum_sq += val * val;
    }
    
    if (sum_sq < 1e-9f) return 0.0f;
    
    float entropy = 0.0f;
    for (float val : s) {
        float p = (val * val) / sum_sq;
        if (p > 1e-9f) {
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

void qsg_mps_context_entangle(
    const float* embeddings,
    const float* site_weights,
    float* context_out,
    float* entropy_out,
    int batch_size,
    int seq_len,
    int embedding_dim,
    int bond_dim,
    int phys_dim
) {
    // O(N·χ²) MPS Contraction Kernel
    // Implements left-to-right state evolution S_t = Contract(S_{t-1}, A[x_t])
    
    // Constants for entropy calculation
    const float kEpsilon = 1e-9f;

    // Parallelize over batch dimension
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        
        // --- 1. Initialize State Vector |ψ_0> ---
        // State vector size is bond_dim (chi)
        // We use std::vector for memory safety, though raw pointers would be slightly faster allocation
        // In highly optimized loops, we'd use a thread-local scratchpad.
        std::vector<float> state(bond_dim, 0.0f);
        std::vector<float> next_state(bond_dim, 0.0f);
        state[0] = 1.0f; // Initial state |0...0> logic in MPS usually starts with [1, 0, ..., 0]

        // Pointers for this batch
        const float* b_emb = embeddings + (size_t)b * seq_len * embedding_dim;
        const float* b_sites = site_weights + (size_t)b * seq_len * bond_dim * phys_dim * bond_dim;
        float* b_out = context_out + (size_t)b * seq_len * embedding_dim;
        float* b_entro = (entropy_out) ? entropy_out + (size_t)b * (seq_len - 1) : nullptr;

        for (int t = 0; t < seq_len; ++t) {
            // --- 2. MPS Contraction Step ---
            // Tensor contraction: NewState[j] = sum_{i, p} State[i] * SiteTensor[i, p, j] * InputProj[p]
            // However, our input is an embedding vector, not a discrete token selector.
            // In "Uniform MPS for Probabilistic Modeling", we often project input to a "physical" vector.
            // Here, we treat 'phys_dim' as a small latent space that the embedding maps to.
            // Optimizing: We assume the embedding has ALREADY been projected to [phys_dim] weights 
            // OR we treat the site tensor as conditional on the input.
            // 
            // SIMPLIFICATION for QSG Context: 
            // The python layer projects input -> [L, bond*phys*bond] which effectively gives us the 
            // A matrices directly for each time step. So 'site_weights' actually contains A[t].
            // A[t] shape: [bond_dim (left), phys_dim (feature), bond_dim (right)]
            // But wait, the python code projects to [B, L, bond*spatial*bond].
            // So at time t, we have a concrete tensor A[t] of shape [chi, d, chi].
            // But we don't have a discrete input 'x_t' selecting a slice.
            // Instead, this IS the tensor for this time step (generated hypernetwork-style or just learned weights).
            // We just need to contract the "physical" leg. But wait, if it's generated from input, 
            // the 'physical' leg is effectively already handled or implied.
            //
            // Let's look at the shape: [bond_left, spatial_bond, bond_right].
            // Standard MPS update: S_next = S_prev * A[x_t]
            // Since A is already verified to be specific to this time step (via hypernet or learned),
            // we just contract. But what about the 'spatial_bond' (phys) dim?
            // Usually we trace it out or project it.
            // For "context entanglement", we often want to aggregate information.
            // We fully contract the state through the tensor.
            //
            // State Update: S_{t+1}[k] = \sum_{i, j} S_t[i] * A[t][i, j, k]
            // We sum over j (physical/spatial) effectively "seeing" all spatial features.
            
            const float* A_t = b_sites + (size_t)t * bond_dim * phys_dim * bond_dim;
            
            // Clear next state
            std::fill(next_state.begin(), next_state.end(), 0.0f);

            // Contraction S[i] * A[i, j, k] -> next_state[k]
            // This is Matrix-Vector multiplication if we view A as [chi, chi] (summed over phys)
            
            // Pre-sum A over physical dimension to get effective transition matrix T[i, k]
            // Optimization: Do this on the fly to avoid large temporary buffers
            
            for (int i = 0; i < bond_dim; ++i) {
                float s_val = state[i];
                if (std::abs(s_val) < 1e-9f) continue; // Skip zero states

                for (int j = 0; j < phys_dim; ++j) {
                    const float* A_slice = A_t + (i * phys_dim + j) * bond_dim;
                    
                    // Vectorized accumulation into next_state
                    // next_state[k] += s_val * A[i, j, k]
                    
                    int k = 0;
                    #if defined(__AVX2__)
                    __m256 s_vec = _mm256_set1_ps(s_val);
                    for (; k + 8 <= bond_dim; k += 8) {
                        __m256 n_vec = _mm256_loadu_ps(&next_state[k]);
                        __m256 a_vec = _mm256_loadu_ps(&A_slice[k]);
                        n_vec = _mm256_fmadd_ps(s_vec, a_vec, n_vec);
                        _mm256_storeu_ps(&next_state[k], n_vec);
                    }
                    #elif defined(__ARM_NEON)
                    float32x4_t s_vec = vdupq_n_f32(s_val);
                    for (; k + 4 <= bond_dim; k += 4) {
                        float32x4_t n_vec = vld1q_f32(&next_state[k]);
                        float32x4_t a_vec = vld1q_f32(&A_slice[k]);
                        n_vec = vmlaq_f32(n_vec, s_vec, a_vec);
                        vst1q_f32(&next_state[k], n_vec);
                    }
                    #endif
                    for (; k < bond_dim; ++k) {
                        next_state[k] += s_val * A_slice[k];
                    }
                }
            }

            // Norm preservation (critical for stable MPS execution without canonical form)
            float norm_sq = 0.0f;
            for (float v : next_state) norm_sq += v * v;
            if (norm_sq > 1e-9f) {
                float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (int k = 0; k < bond_dim; ++k) next_state[k] *= inv_norm;
            }

            // Update state
            state = next_state;

            // --- 3. Output Projection ---
            // Combine Original Embedding + Entangled State -> Output
            // O[d] = E[d] + (State[d % chi] mapped)
            
            const float* emb_ptr = b_emb + t * embedding_dim;
            float* out_ptr = b_out + t * embedding_dim;
            
            // Vectorized addition
            int d = 0;
            #if defined(__AVX2__)
            for (; d + 8 <= embedding_dim; d += 8) {
                __m256 emb_vec = _mm256_loadu_ps(&emb_ptr[d]);
                // Map state cyclically (simple projection)
                // Efficient gathering is tricky, so we reconstruct a vector or just scalar loop for mapping
                // For speed, let's assume we just add. For better ML, we'd use a dense layer (done in Python usually)
                // Here we essentially emulate a residual connection S -> Out.
                // We'll compute the values and load.
                float tmp[8];
                for(int z=0; z<8; ++z) tmp[z] = state[(d+z) % bond_dim];
                __m256 state_vec = _mm256_loadu_ps(tmp);
                
                __m256 res = _mm256_add_ps(emb_vec, state_vec);
                _mm256_storeu_ps(&out_ptr[d], res);
            }
            #endif
            for (; d < embedding_dim; ++d) {
                out_ptr[d] = emb_ptr[d] + state[d % bond_dim];
            }

            // --- 4. Entropy Calculation ---
            // Compute Von Neumann entropy of the bond
            // S = -sum(s^2 * log(s^2)) (simplified for pure states in normalized vector base)
            // Just computing entropy of the probability distribution of the state vector amplitudes
            if (t < seq_len - 1 && b_entro != nullptr) {
                float entropy = 0.0f;
                // state is already normalized L2, so p_k = state[k]^2 sums to 1
                for (float v : state) {
                    float p = v * v;
                    if (p > kEpsilon) {
                        entropy -= p * std::log(p);
                    }
                }
                b_entro[t] = entropy;
            }
        }
    }
}

} // namespace ops
} // namespace saguaro
